"""
Runs AFTER predict.py, on each new trading day, to:
1. Fill in yesterday's actual close (now known) into the log.
2. Compute rolling directional accuracy.
3. Run Page-Hinkley drift detection on prediction errors.
4. Flag "NEEDS_RETRAIN" if performance has degraded significantly.

This does NOT retrain automatically — it only raises a flag.
Retraining itself is triggered manually in Colab by you,
based on this flag (per the agreed v1 design).
"""

import numpy as np
import pandas as pd
from river.drift import PageHinkley

import sys
sys.path.append("..")
from config import (PRED_LOG_PATH, DIRECTIONAL_ACC_MIN, ROLLING_WINDOW,
                     PAGE_HINKLEY_DELTA, PAGE_HINKLEY_THRESHOLD, HSI_TICKER)
from data_sources import fetch_with_fallback


def update_actuals(log: pd.DataFrame) -> pd.DataFrame:
    """Fill in actual_close for rows where the date has since passed."""
    raw = fetch_with_fallback()
    raw.index = pd.to_datetime(raw.index).date

    log["predict_date"] = pd.to_datetime(log["predict_date"]).dt.date
    for idx, row in log.iterrows():
        if pd.isna(row["actual_close"]) and row["predict_date"] in raw.index:
            actual = float(raw.loc[row["predict_date"], "Close"])
            log.at[idx, "actual_close"] = actual

            pred_direction = np.sign(row["pred_return_q50"])
            actual_direction = np.sign(actual - row["last_close"])
            log.at[idx, "directional_hit"] = int(pred_direction == actual_direction)

    return log


def check_drift(log: pd.DataFrame) -> dict:
    """Run Page-Hinkley on prediction errors + rolling directional accuracy check."""
    completed = log.dropna(subset=["actual_close"]).copy()
    if len(completed) < ROLLING_WINDOW:
        return {"status": "INSUFFICIENT_DATA", "needs_retrain": False}

    completed["error"] = (completed["actual_close"] - completed["pred_price_mid"]) / completed["last_close"]

    ph = PageHinkley(delta=PAGE_HINKLEY_DELTA, threshold=PAGE_HINKLEY_THRESHOLD)
    drift_detected = False
    for e in completed["error"]:
        in_drift, _ = ph.update(e)
        if in_drift:
            drift_detected = True

    rolling_acc = completed["directional_hit"].tail(ROLLING_WINDOW).mean()
    acc_breach = rolling_acc < DIRECTIONAL_ACC_MIN

    needs_retrain = drift_detected or acc_breach

    return {
        "status": "OK",
        "rolling_directional_accuracy": float(rolling_acc),
        "page_hinkley_drift_detected": drift_detected,
        "accuracy_breach": bool(acc_breach),
        "needs_retrain": bool(needs_retrain),
    }


if __name__ == "__main__":
    log = pd.read_csv(PRED_LOG_PATH)
    log = update_actuals(log)
    log.to_csv(PRED_LOG_PATH, index=False)

    report = check_drift(log)
    print(report)

    if report.get("needs_retrain"):
        flag_path = PRED_LOG_PATH.parent / "RETRAIN_NEEDED.flag"
        flag_path.write_text(
            f"Retrain flagged at {pd.Timestamp.utcnow()}. Details: {report}"
        )
        print(">>> RETRAIN_NEEDED.flag created. Go to Colab to retrain the model.")

