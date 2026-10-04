"""
TRAINING SCRIPT — run manually in Google Colab only.
GitHub Actions never calls this file; it only runs predict.py.

Full pipeline:
1. Check walk-forward retrain trigger (calendar + drift) — informational only
2. Fetch HSI + cross-market + macro (Stock Connect, GARCH, ADR) data
3. Feature engineering (candlesticks, technicals, CCASS, news, macro)
4. Next-day labeling (close/high/low returns)
5. Optuna hyperparameter search (Purged K-Fold objective)
6. Expanding-window walk-forward validation report (using tuned params)
7. Train final Ensemble model (LightGBM + RandomForest + Ridge) for close
8. Train standalone quantile models (q10/q50/q90) for close band
9. Train direct High / Low regressors
10. Train Meta-Labeling (confidence) model
11. Train per-stock bottom-up models (close q50)
12. Save everything + metrics.json + mark_trained_today()
"""

import json
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (MODEL_CLOSE_Q10_PATH, MODEL_CLOSE_Q50_PATH, MODEL_CLOSE_Q90_PATH,
                     MODEL_HIGH_PATH, MODEL_LOW_PATH, FEATURE_LIST_PATH, METRICS_PATH,
                     STOCK_MODEL_DIR, HSI_TICKER, US_FUTURES_TICKER, VIX_TICKER,
                     LGB_PARAMS)
from data_sources import fetch_with_fallback
from features import build_features, get_numeric_feature_columns
from labeling import build_nextday_labels, LABEL_COLUMNS
from confidence import build_meta_labels, train_meta_model
from news_sentiment import get_daily_market_sentiment, get_stock_sentiment
from ccass_scraper import get_ccass_change
from stock_universe import get_universe
from ensemble_model import HSIEnsembleModel
from optuna_tuning import run_optuna_search
from walk_forward import should_retrain, mark_trained_today, expanding_window_validation


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_labeled_hsi_dataset():
    """Fetches HSI + cross-market data, builds features, applies next-day labeling."""
    us_futures = fetch_with_fallback(US_FUTURES_TICKER)
    vix = fetch_with_fallback(VIX_TICKER)
    market_sentiment = get_daily_market_sentiment()

    raw = fetch_with_fallback(HSI_TICKER, stooq_ticker="^hsi")
    feat_df = build_features(raw, us_futures=us_futures, vix=vix,
                              ccass_change=0.0, market_sentiment=market_sentiment,
                              stock_sentiment=0.0, ticker=HSI_TICKER, include_macro=True)
    labeled_df = build_nextday_labels(feat_df)
    return labeled_df


# ---------------------------------------------------------------------------
# HSI-level model training
# ---------------------------------------------------------------------------

def train_hsi_models():
    print("=== [Step 1] Checking walk-forward retrain trigger (informational) ===")
    retrain_status = should_retrain()
    print(json.dumps(retrain_status, indent=2))

    print("=== [Step 2] Building HSI dataset (features + next-day labels) ===")
    labeled_df = build_labeled_hsi_dataset()

    feature_cols = get_numeric_feature_columns(labeled_df, exclude=LABEL_COLUMNS)
    X = labeled_df[feature_cols]
    y_close = labeled_df["next_close_return"]
    y_high = labeled_df["next_high_return"]
    y_low = labeled_df["next_low_return"]

    print(f"Dataset ready: {len(X)} samples, {len(feature_cols)} features")

    print("=== [Step 3] Optuna hyperparameter search (close target) ===")
    best_params = run_optuna_search(X, y_close)

    print("=== [Step 4] Expanding-window walk-forward validation (tuned params) ===")
    def model_fn(X_tr, y_tr):
        m = lgb.LGBMRegressor(objective="regression", **best_params)
        m.fit(X_tr, y_tr)
        return m

    wf_report = expanding_window_validation(X, y_close, model_fn, n_windows=5)
    print("Walk-forward validation report:")
    print(json.dumps(wf_report, indent=2))

    print("=== [Step 5] Training Ensemble model (close) ===")
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y_close.iloc[:split_idx], y_close.iloc[split_idx:]

    ensemble = HSIEnsembleModel(lgb_params=best_params)
    ensemble_report = ensemble.fit(X_train, y_train, X_val, y_val)
    print("Ensemble weights:", ensemble_report["weights"])
    ensemble.save(prefix="hsi")

    print("=== [Step 6] Training quantile models (close q10/q50/q90) ===")
    def train_quantile(y, alpha):
        params = {k: v for k, v in best_params.items()}
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
        m.fit(X, y)
        return m

    model_close_q10 = train_quantile(y_close, 0.1)
    model_close_q50 = train_quantile(y_close, 0.5)
    model_close_q90 = train_quantile(y_close, 0.9)

    model_close_q10.booster_.save_model(str(MODEL_CLOSE_Q10_PATH))
    model_close_q50.booster_.save_model(str(MODEL_CLOSE_Q50_PATH))
    model_close_q90.booster_.save_model(str(MODEL_CLOSE_Q90_PATH))

    print("=== [Step 7] Training direct High / Low regressors ===")
    model_high = lgb.LGBMRegressor(objective="regression", **best_params)
    model_high.fit(X, y_high)
    model_high.booster_.save_model(str(MODEL_HIGH_PATH))

    model_low = lgb.LGBMRegressor(objective="regression", **best_params)
    model_low.fit(X, y_low)
    model_low.booster_.save_model(str(MODEL_LOW_PATH))

    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_cols, f)

    print("=== [Step 8] Training Meta-Labeling (confidence) model ===")
    primary_pred = model_close_q50.predict(X)
    meta_labels = build_meta_labels(pd.Series(primary_pred, index=X.index), y_close)
    train_meta_model(X, meta_labels)

    return {
        "best_hyperparams": best_params,
        "ensemble_report": ensemble_report,
        "walk_forward_report": wf_report,
        "retrain_trigger_status": retrain_status,
        "n_samples": int(len(X)),
        "n_features": len(feature_cols),
        "date_range": {"start": str(labeled_df.index.min()), "end": str(labeled_df.index.max())},
    }


# ---------------------------------------------------------------------------
# Per-stock bottom-up model training
# ---------------------------------------------------------------------------

def train_stock_models():
    print("=== Training per-stock bottom-up models ===")
    universe = get_universe()
    stock_metrics = {}

    for ticker, weight in universe.items():
        try:
            print(f"Training {ticker} (weight={weight:.4f})")
            stooq_code = ticker.replace(".HK", "").zfill(5) + ".hk"
            keywords = [ticker.split(".")[0]]

            raw = fetch_with_fallback(ticker, stooq_ticker=stooq_code)
            ccass_change = get_ccass_change(ticker.split(".")[0])
            stock_sentiment = get_stock_sentiment(keywords)

            feat_df = build_features(raw, ccass_change=ccass_change,
                                      stock_sentiment=stock_sentiment, ticker=ticker,
                                      include_macro=False)
            labeled_df = build_nextday_labels(feat_df)

            if len(labeled_df) < 100:
                print(f"Skipping {ticker}: insufficient data ({len(labeled_df)} rows)")
                continue

            feature_cols = get_numeric_feature_columns(labeled_df, exclude=LABEL_COLUMNS)
            X = labeled_df[feature_cols]
            y = labeled_df["next_close_return"]

            model = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
            model.fit(X, y)

            model_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_close_q50.txt"
            model.booster_.save_model(str(model_path))

            feat_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_features.json"
            with open(feat_path, "w") as f:
                json.dump(feature_cols, f)

            stock_metrics[ticker] = {"weight": weight, "n_samples": int(len(X))}
        except Exception as e:
            print(f"Failed training {ticker}: {e}")
            continue

    return stock_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    hsi_metrics = train_hsi_models()
    stock_metrics = train_stock_models()

    metrics = {
        "trained_at": datetime.datetime.utcnow().isoformat(),
        "hsi_model": hsi_metrics,
        "stock_models": stock_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    mark_trained_today()

    print("=== Training complete ===")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
