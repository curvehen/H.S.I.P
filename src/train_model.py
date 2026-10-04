"""
TRAINING SCRIPT — run manually in Google Colab only.
GitHub Actions never calls this file; it only runs predict.py.

Pipeline:
1. Fetch HSI + cross-market data (multi-source fallback)
2. Feature engineering (candlesticks, technicals, CCASS, news, macro)
3. Next-day labeling (close/high/low returns)
4. Purged K-Fold validated Ensemble model (primary point estimate for close)
5. Quantile models for close (q10/q50/q90) + direct high/low regressors
6. Meta-confidence model
7. Per-stock bottom-up models (close q50 only, for aggregation)
8. Save everything + metrics.json
"""

import json
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

from config import (MODEL_CLOSE_Q10_PATH, MODEL_CLOSE_Q50_PATH, MODEL_CLOSE_Q90_PATH,
                     MODEL_HIGH_PATH, MODEL_LOW_PATH, FEATURE_LIST_PATH, METRICS_PATH,
                     STOCK_MODEL_DIR, HSI_TICKER, US_FUTURES_TICKER, VIX_TICKER,
                     PURGE_EMBARGO_DAYS, LGB_PARAMS, LAST_TRAIN_DATE_PATH)
from data_sources import fetch_with_fallback
from features import build_features, get_numeric_feature_columns
from labeling import build_nextday_labels, LABEL_COLUMNS
from confidence import build_meta_labels, train_meta_model
from news_sentiment import get_daily_market_sentiment, get_stock_sentiment
from stock_universe import get_universe
from ensemble_model import HSIEnsembleModel


def purged_kfold_indices(n_samples, n_splits=5, embargo=PURGE_EMBARGO_DAYS):
    kf = KFold(n_splits=n_splits, shuffle=False)
    for train_idx, test_idx in kf.split(np.arange(n_samples)):
        test_start, test_end = test_idx.min(), test_idx.max()
        purge_mask = ~((train_idx >= test_start - embargo) & (train_idx <= test_end + embargo))
        yield train_idx[purge_mask], test_idx


def build_labeled_hsi_dataset():
    us_futures = fetch_with_fallback(US_FUTURES_TICKER)
    vix = fetch_with_fallback(VIX_TICKER)
    market_sentiment = get_daily_market_sentiment()

    raw = fetch_with_fallback(HSI_TICKER, stooq_ticker="^hsi")
    feat_df = build_features(raw, us_futures=us_futures, vix=vix,
                              ccass_change=0.0, market_sentiment=market_sentiment,
                              stock_sentiment=0.0, ticker=HSI_TICKER)
    labeled_df = build_nextday_labels(feat_df)
    return labeled_df


def train_hsi_models():
    print("=== Building HSI dataset ===")
    labeled_df = build_labeled_hsi_dataset()

    feature_cols = get_numeric_feature_columns(labeled_df, exclude=LABEL_COLUMNS)
    X = labeled_df[feature_cols]
    y_close = labeled_df["next_close_return"]
    y_high = labeled_df["next_high_return"]
    y_low = labeled_df["next_low_return"]

    print(f"Dataset ready: {len(X)} samples, {len(feature_cols)} features")

    # ---- Purged K-Fold validation (close target) ----
    print("=== Purged K-Fold validation (close target) ===")
    rmses = []
    for train_idx, test_idx in purged_kfold_indices(len(X)):
        m = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
        m.fit(X.iloc[train_idx], y_close.iloc[train_idx])
        preds = m.predict(X.iloc[test_idx])
        rmses.append(float(np.sqrt(np.mean((preds - y_close.iloc[test_idx]) ** 2))))
    avg_rmse = float(np.mean(rmses))
    print(f"Avg Purged K-Fold RMSE (next_close_return): {avg_rmse:.5f}")

    # ---- Ensemble model for close (primary point estimate) ----
    print("=== Training Ensemble model (close) ===")
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y_close.iloc[:split_idx], y_close.iloc[split_idx:]

    ensemble = HSIEnsembleModel(lgb_params=LGB_PARAMS)
    ensemble_report = ensemble.fit(X_train, y_train, X_val, y_val)
    print("Ensemble weights:", ensemble_report["weights"])
    ensemble.save(prefix="hsi")

    # ---- Quantile models for close band (q10/q50/q90) ----
    print("=== Training quantile models (close q10/q50/q90) ===")
    def train_quantile(y, alpha):
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha,
                               **{k: v for k, v in LGB_PARAMS.items()})
        m.fit(X, y)
        return m

    model_close_q10 = train_quantile(y_close, 0.1)
    model_close_q50 = train_quantile(y_close, 0.5)
    model_close_q90 = train_quantile(y_close, 0.9)

    model_close_q10.booster_.save_model(str(MODEL_CLOSE_Q10_PATH))
    model_close_q50.booster_.save_model(str(MODEL_CLOSE_Q50_PATH))
    model_close_q90.booster_.save_model(str(MODEL_CLOSE_Q90_PATH))

    # ---- Direct High / Low regressors ----
    print("=== Training High/Low regressors ===")
    model_high = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
    model_high.fit(X, y_high)
    model_high.booster_.save_model(str(MODEL_HIGH_PATH))

    model_low = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
    model_low.fit(X, y_low)
    model_low.booster_.save_model(str(MODEL_LOW_PATH))

    # ---- Save feature column list (critical: predict.py must use same order) ----
    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_cols, f)

    # ---- Meta-confidence model ----
    print("=== Training meta-confidence model ===")
    primary_pred = model_close_q50.predict(X)
    meta_labels = build_meta_labels(pd.Series(primary_pred, index=X.index), y_close)
    train_meta_model(X, meta_labels)

    return {
        "avg_purged_kfold_rmse_close": avg_rmse,
        "ensemble_report": ensemble_report,
        "n_samples": int(len(X)),
        "n_features": len(feature_cols),
        "date_range": {"start": str(labeled_df.index.min()), "end": str(labeled_df.index.max())},
    }


def train_stock_models():
    """Per-stock bottom-up models (close-return q50 only, used for HSI aggregation)."""
    print("=== Training per-stock bottom-up models ===")
    from ccass_scraper import get_ccass_change

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

    LAST_TRAIN_DATE_PATH.write_text(datetime.date.today().isoformat())

    print("=== Training complete ===")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()

