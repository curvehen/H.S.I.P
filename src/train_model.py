"""
TRAINING SCRIPT — run manually in Google Colab only.
GitHub Actions never calls this file; it only runs predict.py.

Pipeline:
1. Fetch HSI + cross-market data (multi-source fallback)
2. Fetch per-stock data for HSI constituents (bottom-up)
3. Feature engineering (candlesticks, technicals, CCASS, news)
4. Triple-barrier labeling
5. Purged K-Fold LightGBM quantile training (HSI-level model)
6. Meta-labeling model training
7. Per-stock quantile models (bottom-up aggregation)
8. Save all models + metrics.json
"""

import json
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

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


def purged_kfold_indices(n_samples, n_splits=5, embargo=BARRIER_HOLDING_DAYS):
    """Purged K-Fold: removes train samples near test-fold boundary to prevent leakage."""
    kf = KFold(n_splits=n_splits, shuffle=False)
    for train_idx, test_idx in kf.split(np.arange(n_samples)):
        test_start, test_end = test_idx.min(), test_idx.max()
        purge_mask = ~((train_idx >= test_start - embargo) & (train_idx <= test_end + embargo))
        yield train_idx[purge_mask], test_idx


def train_quantile_model(X, y, alpha):
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=alpha, n_estimators=500,
        learning_rate=0.03, max_depth=5, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, verbosity=-1,
    )
    model.fit(X, y)
    return model


def build_labeled_dataset(ticker, stooq_ticker=None, us_futures=None, vix=None,
                           ccass_change=0.0, market_sentiment=0.0, stock_kw=None):
    raw = fetch_with_fallback(ticker, stooq_ticker)
    stock_sentiment = get_stock_sentiment(stock_kw) if stock_kw else 0.0
    feat_df = build_features(raw, us_futures, vix, ccass_change, market_sentiment, stock_sentiment)
    labeled_df = triple_barrier_labels(feat_df)
    return labeled_df


def get_numeric_features(df):
    exclude = ["barrier_label", "barrier_return", "source", "is_stale"]
    cols = [c for c in df.columns if c not in exclude]
    return df[cols].select_dtypes(include=[np.number])


def train_hsi_model():
    print("=== Training HSI-level model ===")
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

    rmses = []
    for train_idx, test_idx in purged_kfold_indices(len(X)):
        m = train_quantile_model(X.iloc[train_idx], y.iloc[train_idx], alpha=0.5)
        preds = m.predict(X.iloc[test_idx])
        rmses.append(float(np.sqrt(np.mean((preds - y.iloc[test_idx]) ** 2))))
    avg_rmse = float(np.mean(rmses))
    print(f"HSI Purged K-Fold avg RMSE (q50): {avg_rmse:.5f}")

    model_q10 = train_quantile_model(X, y, alpha=0.1)
    model_q50 = train_quantile_model(X, y, alpha=0.5)
    model_q90 = train_quantile_model(X, y, alpha=0.9)

    model_q10.booster_.save_model(str(MODEL_Q10_PATH))
    model_q50.booster_.save_model(str(MODEL_Q50_PATH))
    model_q90.booster_.save_model(str(MODEL_Q90_PATH))

    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(list(X.columns), f)

    # Meta-labeling
    primary_pred = model_q50.predict(X)
    meta_labels = build_meta_labels(pd.Series(primary_pred, index=X.index), y)
    train_meta_model(X, meta_labels)

    return {
        "avg_purged_kfold_rmse_q50": avg_rmse,
        "n_samples": int(len(X)),
        "n_features": len(X.columns),
        "date_range": {"start": str(labeled_df.index.min()), "end": str(labeled_df.index.max())},
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

            model_q50 = train_quantile_model(X, y, alpha=0.5)
            model_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_q50.txt"
            model_q50.booster_.save_model(str(model_path))

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

    print("=== Training complete ===")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
