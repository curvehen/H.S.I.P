"""
TRAINING SCRIPT — run manually in Google Colab only.
GitHub Actions never calls this file; it only runs predict.py.

Pipeline:
1. Check walk-forward trigger (calendar + drift) -- informational, does not block manual run
2. Fetch HSI + cross-market + macro (Stock Connect, GARCH) + intraday data
3. Feature engineering (candlesticks, technicals, CCASS, news, macro, intraday)
4. Triple-barrier labeling
5. Optuna hyperparameter search (Purged K-Fold objective)
6. Expanding-window walk-forward validation report
7. Train final Ensemble model (LightGBM + RandomForest + Ridge) on full dataset
8. Train standalone quantile models (q10/q50/q90) for band estimation
9. Train meta-labeling model
10. Per-stock bottom-up models
11. Save everything + metrics.json + mark_trained_today()
"""

import json
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (MODEL_Q10_PATH, MODEL_Q50_PATH, MODEL_Q90_PATH,
                     FEATURE_LIST_PATH, METRICS_PATH, STOCK_MODEL_DIR,
                     HSI_TICKER, US_FUTURES_TICKER, VIX_TICKER,
                     BARRIER_HOLDING_DAYS)
from data_sources import fetch_with_fallback
from features import build_features
from labeling import triple_barrier_labels
from meta_labeling import build_meta_labels, train_meta_model
from ccass_scraper import get_ccass_change
from news_sentiment import get_daily_market_sentiment, get_stock_sentiment
from stock_universe import get_universe
from ensemble_model import HSIEnsembleModel
from optuna_tuning import run_optuna_search, load_best_params
from walk_forward import should_retrain, mark_trained_today, expanding_window_validation


def get_numeric_features(df):
    exclude = ["barrier_label", "barrier_return", "source", "is_stale"]
    cols = [c for c in df.columns if c not in exclude]
    return df[cols].select_dtypes(include=[np.number])


def build_labeled_dataset(ticker, stooq_ticker=None, us_futures=None, vix=None,
                           ccass_change=0.0, market_sentiment=0.0, stock_kw=None):
    raw = fetch_with_fallback(ticker, stooq_ticker)
    stock_sentiment = get_stock_sentiment(stock_kw) if stock_kw else 0.0
    feat_df = build_features(raw, us_futures, vix, ccass_change, market_sentiment,
                              stock_sentiment, ticker=ticker)
    labeled_df = triple_barrier_labels(feat_df)
    return labeled_df


def train_hsi_model():
    print("=== [1/3] Checking walk-forward retrain trigger (informational) ===")
    retrain_status = should_retrain()
    print(json.dumps(retrain_status, indent=2))

    print("=== [2/3] Fetching data + building features ===")
    us_futures = fetch_with_fallback(US_FUTURES_TICKER)
    vix = fetch_with_fallback(VIX_TICKER)
    market_sentiment = get_daily_market_sentiment()

    labeled_df = build_labeled_dataset(
        HSI_TICKER, stooq_ticker="^hsi",
        us_futures=us_futures, vix=vix,
        ccass_change=0.0, market_sentiment=market_sentiment,
    )

    X = get_numeric_features(labeled_df)
    y = labeled_df["barrier_return"]

    print(f"Dataset ready: {len(X)} samples, {len(X.columns)} features")

    print("=== [3/3] Optuna hyperparameter search ===")
    best_params = run_optuna_search(X, y)

    # --- Walk-forward validation using tuned params ---
    def model_fn(X_tr, y_tr):
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, verbosity=-1, **best_params)
        m.fit(X_tr, y_tr)
        return m

    wf_report = expanding_window_validation(X, y, model_fn)
    print("Walk-forward validation report:")
    print(json.dumps(wf_report, indent=2))

    # --- Train final Ensemble model on full data (last 20% held out for weighting) ---
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    ensemble = HSIEnsembleModel(lgb_params=best_params)
    ensemble_report = ensemble.fit(X_train, y_train, X_val, y_val)
    print("Ensemble weights:", ensemble_report["weights"])
    ensemble.save(prefix="hsi")

    # --- Train standalone quantile models for band estimation (q10/q50/q90) ---
    def train_quantile(alpha):
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, verbosity=-1, **best_params)
        m.fit(X, y)
        return m

    model_q10 = train_quantile(0.1)
    model_q50 = train_quantile(0.5)
    model_q90 = train_quantile(0.9)

    model_q10.booster_.save_model(str(MODEL_Q10_PATH))
    model_q50.booster_.save_model(str(MODEL_Q50_PATH))
    model_q90.booster_.save_model(str(MODEL_Q90_PATH))

    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(list(X.columns), f)

    # --- Meta-labeling ---
    primary_pred = model_q50.predict(X)
    meta_labels = build_meta_labels(pd.Series(primary_pred, index=X.index), y)
    train_meta_model(X, meta_labels)

    return {
        "best_hyperparams": best_params,
        "ensemble_report": ensemble_report,
        "walk_forward_report": wf_report,
        "n_samples": int(len(X)),
        "n_features": len(X.columns),
        "date_range": {"start": str(labeled_df.index.min()), "end": str(labeled_df.index.max())},
        "retrain_trigger_status": retrain_status,
    }


def train_stock_models():
    print("=== Training per-stock bottom-up models ===")
    universe = get_universe()
    stock_metrics = {}

    for ticker, weight in universe.items():
        try:
            print(f"Training {ticker} (weight={weight:.4f})")
            stooq_code = ticker.replace(".HK", "").zfill(5) + ".hk"
            keywords = [ticker.split(".")[0]]
            labeled_df = build_labeled_dataset(ticker, stooq_ticker=stooq_code, stock_kw=keywords)

            if len(labeled_df) < 100:
                print(f"Skipping {ticker}: insufficient data ({len(labeled_df)} rows)")
                continue

            X = get_numeric_features(labeled_df)
            y = labeled_df["barrier_return"]

            params = load_best_params()

                        for alpha, suffix in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
                model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, verbosity=-1, **params)
                model.fit(X, y)
                model_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_{suffix}.txt"
                model.booster_.save_model(str(model_path))

            feat_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_features.json"
            with open(feat_path, "w") as f:
                json.dump(list(X.columns), f)

            stock_metrics[ticker] = {"weight": weight, "n_samples": int(len(X))}
        except Exception as e:
            print(f"Failed training {ticker}: {e}")
            continue

    return stock_metrics


def main():
    hsi_metrics = train_hsi_model()
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
