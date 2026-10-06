"""
Runs after predict.py on each trading day:
1. Fill in actual_close once known.
2. Compute rolling directional accuracy.
3. Page-Hinkley drift detection on prediction errors.
4. Flag RETRAIN_NEEDED if degraded — retraining stays manual (Colab).
"""

import numpy as np
import pandas as pd
from river.drift import PageHinkley

from config import (PRED_LOG_PATH, DIRECTIONAL_ACC_MIN, ROLLING_WINDOW,
                     PAGE_HINKLEY_DELTA, PAGE_HINKLEY_THRESHOLD,
                     HSI_TICKER, RETRAIN_FLAG_PATH)
from data_sources import fetch_with_fallback
from market_hours import get_market_session

def update_actuals(log: pd.DataFrame) -> pd.DataFrame:
    if get_market_session() != "post_market":
      print("Skip backfill: market not yet closed.")
      return log
    raw = fetch_with_fallback(HSI_TICKER, stooq_ticker="^hsi")
    raw.index = pd.to_datetime(raw.index).date

    log["predict_date"] = pd.to_datetime(log["predict_date"]).dt.date
    for idx, row in log.iterrows():
        if pd.isna(row["actual_close"]) and row["predict_date"] in raw.index:
            actual = float(raw.loc[row["predict_date"], "Close"])
            log.at[idx, "actual_close"] = actual
            pred_direction = np.sign(row["pred_close_return_blended"])
            actual_direction = np.sign(actual - row["last_close"])
            log.at[idx, "directional_hit"] = int(pred_direction == actual_direction)
    return log


def check_drift(log: pd.DataFrame) -> dict:
    completed = log.dropna(subset=["actual_close"]).copy()
    if len(completed) < ROLLING_WINDOW:
        return {"status": "INSUFFICIENT_DATA", "needs_retrain": False}

    completed["error"] = (completed["actual_close"] - completed["pred_close"]) / completed["last_close"]

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
        "page_hinkley_drift_detected": bool(drift_detected),
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
        RETRAIN_FLAG_PATH.write_text(f"Retrain flagged at {pd.Timestamp.utcnow()}. Details: {report}")
        print(">>> RETRAIN_NEEDED.flag created. Retrain in Colab.")
    elif RETRAIN_FLAG_PATH.exists():
        RETRAIN_FLAG_PATH.unlink()
        print(">>> Performance recovered. RETRAIN_NEEDED.flag removed.")
