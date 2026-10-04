"""
Walk-forward retraining controller + expanding-window validation.
Combines calendar-based trigger (forces periodic refresh) with
drift-based trigger (from evaluate_drift.py) to decide when a retrain
should happen. Also provides expanding-window validation during training
to simulate realistic sequential retraining performance.
"""

import datetime
import numpy as np
import pandas as pd

from config import LAST_TRAIN_DATE_PATH, RETRAIN_FLAG_PATH, WALK_FORWARD_MIN_DAYS_SINCE_RETRAIN


def days_since_last_train() -> int:
    if not LAST_TRAIN_DATE_PATH.exists():
        return 999999
    last_date = datetime.date.fromisoformat(LAST_TRAIN_DATE_PATH.read_text().strip())
    return (datetime.date.today() - last_date).days


def should_retrain() -> dict:
    days_elapsed = days_since_last_train()
    calendar_trigger = days_elapsed >= WALK_FORWARD_MIN_DAYS_SINCE_RETRAIN
    drift_trigger = RETRAIN_FLAG_PATH.exists()

    return {
        "days_since_last_train": days_elapsed,
        "calendar_trigger": calendar_trigger,
        "drift_trigger": drift_trigger,
        "should_retrain": calendar_trigger or drift_trigger,
    }


def mark_trained_today():
    LAST_TRAIN_DATE_PATH.write_text(datetime.date.today().isoformat())


def expanding_window_validation(X: pd.DataFrame, y: pd.Series, model_fn, n_windows: int = 5) -> dict:
    """
    Simulates realistic walk-forward deployment: train on an expanding window,
    test on the immediately following block, roll forward. More representative
    of live retraining behavior than a single static train/test split.

    model_fn: callable(X_train, y_train) -> fitted model with .predict(X)
    """
    n = len(X)
    fold_size = n // (n_windows + 1)
    rmses, directional_accs = [], []

    for i in range(1, n_windows + 1):
        train_end = fold_size * i
        test_end = min(fold_size * (i + 1), n)
        if train_end >= test_end:
            continue

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

        model = model_fn(X_train, y_train)
        preds = model.predict(X_test)

        rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
        dir_acc = float((np.sign(preds) == np.sign(y_test)).mean())

        rmses.append(rmse)
        directional_accs.append(dir_acc)

    return {
        "n_windows_evaluated": len(rmses),
        "avg_rmse": float(np.mean(rmses)) if rmses else None,
        "avg_directional_accuracy": float(np.mean(directional_accs)) if directional_accs else None,
        "rmse_per_window": rmses,
        "directional_acc_per_window": directional_accs,
    }
