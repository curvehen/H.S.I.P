"""
TRAINING SCRIPT — run this manually in Google Colab only.
Not called by GitHub Actions (Actions only runs predict.py).

Trains LightGBM quantile regressors (q10/q50/q90) with
Purged K-Fold cross-validation to avoid look-ahead leakage.
"""

import json
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

import sys
sys.path.append("..")
from config import (MODEL_Q10_PATH, MODEL_Q50_PATH, MODEL_Q90_PATH,
                     METRICS_PATH, BARRIER_HOLDING_DAYS)
from data_sources import fetch_with_fallback, save_as_last_good
from features import build_features
from labeling import triple_barrier_labels


def purged_kfold_indices(n_samples, n_splits=5, embargo=BARRIER_HOLDING_DAYS):
    """
    Simple purged K-Fold: removes samples near the test fold boundary
    to prevent label leakage from overlapping barrier windows.
    """
    kf = KFold(n_splits=n_splits, shuffle=False)
    for train_idx, test_idx in kf.split(np.arange(n_samples)):
        # purge training samples within `embargo` days of the test set
        test_start, test_end = test_idx.min(), test_idx.max()
        purge_mask = ~(
            (train_idx >= test_start - embargo) & (train_idx <= test_end + embargo)
        )
        yield train_idx[purge_mask], test_idx


def train_quantile_model(X, y, alpha):
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(X, y)
    return model


def main():
    # 1. Fetch data
    raw = fetch_with_fallback()
    save_as_last_good(raw)

    # 2. Feature engineering
    feat_df = build_features(raw)

    # 3. Labeling (target = next-period return, using barrier_return as proxy)
    labeled_df = triple_barrier_labels(feat_df)

    feature_cols = [c for c in labeled_df.columns
                    if c not in ["barrier_label", "barrier_return", "source", "is_stale"]]
    X = labeled_df[feature_cols].select_dtypes(include=[np.number])
    y = labeled_df["barrier_return"]

    # 4. Purged K-Fold validation (report only; final model trained on all data)
    rmses = []
    for train_idx, test_idx in purged_kfold_indices(len(X)):
        m = train_quantile_model(X.iloc[train_idx], y.iloc[train_idx], alpha=0.5)
        preds = m.predict(X.iloc[test_idx])
        rmse = float(np.sqrt(np.mean((preds - y.iloc[test_idx]) ** 2)))
        rmses.append(rmse)

    avg_rmse = float(np.mean(rmses))
    print(f"Purged K-Fold avg RMSE (q50): {avg_rmse:.5f}")

    # 5. Train final models on full dataset
    model_q10 = train_quantile_model(X, y, alpha=0.1)
    model_q50 = train_quantile_model(X, y, alpha=0.5)
    model_q90 = train_quantile_model(X, y, alpha=0.9

