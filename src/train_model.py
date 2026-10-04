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

    ens
