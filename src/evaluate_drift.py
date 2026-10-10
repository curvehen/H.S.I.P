"""
Post-market evaluation module — run daily AFTER market close (part of the
official 4am HKT run, following drift's natural place: evaluate YESTERDAY's
prediction now that today's actual close is known).

1. Backfills `actual_close` into predictions_log.csv for rows where the
   actual outcome is now known.
2. Computes rolling directional accuracy over the trailing window.
3. Runs a Page-Hinkley test on prediction errors to flag model drift.
4. Updates RETRAIN_NEEDED.flag if directional accuracy degrades below
   threshold or drift is detected.
5. (NEW) Feeds each day's per-sub-model prediction + actual outcome into
   dynamic_ensemble_weighter.update_dynamic_weights(), so ENH#2's rolling
   Brier-score dynamic weighting actually accumulates history over time.
   Without this step, init_dynamic_weights() creates an empty tracker in
   train_model.py that NEVER gets updated, and predict.py's
   get_effective_weights() permanently falls back to static weights.

FIX LOG (this revision):
  - Corrected every column reference to predict.py's ACTUAL
    predictions_log.csv schema (confirmed by reading log_prediction()
    directly): predict_date -> date, pred_close -> pred_close_price.
    directional_hit is not a column predict.py writes — it is now computed
    HERE from sign(pred_close_price - last_close) vs
    sign(actual_close - last_close), since last_close IS available in the
    log row.
  - NEW: after backfilling actual_close, iterates newly-backfilled rows and
    calls update_dynamic_weights(date, sub_model_preds, actual_return) for
    each one, using the sub_pred_lgb/sub_pred_rf/sub_pred_ridge columns
    (corrected short-form keys matching ensemble_model.py's actual
    "lgb"/"rf"/"ridge" naming) and actual_return computed as
    (actual_close / last_close - 1), matching the return-based convention
    predict.py's sub-model predictions use. Wrapped fail-safe: a missing
    dynamic_ensemble_weighter.py or any internal error logs a warning and
    is skipped per-row, never crashing the overall evaluation run.
"""

import json
import numpy as np
import pandas as pd

from config import PRED_LOG_PATH, DIRECTIONAL_ACC_MIN, ROLLING_ACCURACY_WINDOW, \
                    PAGE_HINKLEY_DELTA, PAGE_HINKLEY_THRESHOLD, RETRAIN_FLAG_PATH, \
                    DRIFT_FLAG_PATH
from data_sources import fetch_with_fallback
from config import HSI_TICKER

try:
    from dynamic_ensemble_weighter import update_dynamic_weights
    _DYNAMIC_WEIGHTING_AVAILABLE = True
except ImportError:
    _DYNAMIC_WEIGHTING_AVAILABLE = False
    print("evaluate_drift: dynamic_ensemble_weighter not available — "
          "skipping dynamic weight updates (predict.py will keep using static weights).")


def backfill_actuals(df: pd.DataFrame) -> pd.DataFrame:
    """Fills in actual_close for any row where it's still null and the
    target trading date has already closed. Returns the updated DataFrame
    AND the set of row indices that were newly backfilled this run (needed
    downstream to feed update_dynamic_weights() exactly once per row)."""
    df = df.copy()
    newly_backfilled_idx = []

    raw = fetch_with_fallback(HSI_TICKER, stooq_ticker="^hsi")
    actual_by_date = raw["Close"].copy()
    actual_by_date.index = pd.to_datetime(actual_by_date.index).date

    for idx, row in df.iterrows():
        if pd.notna(row.get("actual_close")):
            continue
        try:
            row_date = pd.to_datetime(row["date"]).date()
        except Exception:
            continue
        if row_date in actual_by_date.index:
            df.at[idx, "actual_close"] = float(actual_by_date.loc[row_date])
            newly_backfilled_idx.append(idx)

    return df, newly_backfilled_idx


def _compute_directional_hit(row: pd.Series) -> bool:
    """predict.py does not write a pre-computed directional_hit column —
    derived here from sign(pred_close_price - last_close) vs
    sign(actual_close - last_close)."""
    if pd.isna(row.get("actual_close")) or pd.isna(row.get("pred_close_price")) or pd.isna(row.get("last_close")):
        return None
    pred_dir = np.sign(row["pred_close_price"] - row["last_close"])
    actual_dir = np.sign(row["actual_close"] - row["last_close"])
    return bool(pred_dir == actual_dir)


def compute_rolling_accuracy(df: pd.DataFrame, window: int = ROLLING_ACCURACY_WINDOW) -> float:
    evaluated = df.dropna(subset=["actual_close", "pred_close_price", "last_close"]).copy()
    if len(evaluated) == 0:
        return None
    evaluated["directional_hit"] = evaluated.apply(_compute_directional_hit, axis=1)
    recent = evaluated.tail(window)
    if len(recent) == 0:
        return None
    return float(recent["directional_hit"].mean())


def page_hinkley_test(errors: pd.Series, delta: float = PAGE_HINKLEY_DELTA,
                       threshold: float = PAGE_HINKLEY_THRESHOLD) -> bool:
    """Standard Page-Hinkley drift detector over a stream of prediction
    errors. Returns True if cumulative deviation exceeds threshold."""
    if len(errors) < 5:
        return False
    mean_error = 0.0
    cumulative = 0.0
    min_cumulative = 0.0
    drift_detected = False
    for i, e in enumerate(errors):
        mean_error += (e - mean_error) / (i + 1)
        cumulative += e - mean_error - delta
        min_cumulative = min(min_cumulative, cumulative)
        if cumulative - min_cumulative > threshold:
            drift_detected = True
    return drift_detected


def _update_dynamic_weights_for_backfilled_rows(df: pd.DataFrame, newly_backfilled_idx: list):
    """Feeds each newly-backfilled row's sub-model predictions + realized
    actual return into dynamic_ensemble_weighter.update_dynamic_weights(),
    so ENH#2's rolling Brier-score history actually accumulates over time.
    Wrapped fail-safe per-row — one bad row never blocks the rest."""
    if not _DYNAMIC_WEIGHTING_AVAILABLE or not newly_backfilled_idx:
        return

    for idx in newly_backfilled_idx:
        row = df.loc[idx]
        try:
            if pd.isna(row.get("actual_close")) or pd.isna(row.get("last_close")):
                continue
            sub_model_preds = {
                "lgb": row.get("sub_pred_lgb"),
                "rf": row.get("sub_pred_rf"),
                "ridge": row.get("sub_pred_ridge"),
            }
            if any(v is None or pd.isna(v) for v in sub_model_preds.values()):
                print(f"evaluate_drift: row {row.get('date')} missing sub-model predictions — "
                      f"skipping dynamic weight update for this row.")
                continue

            actual_return = float(row["actual_close"] / row["last_close"] - 1)
            update_dynamic_weights(
                date=row.get("date"),
                sub_model_preds=sub_model_preds,
                actual_return=actual_return,
            )
        except Exception as e:
            print(f"evaluate_drift: dynamic weight update failed for row {row.get('date')} ({e}) — skipped.")

    print(f"evaluate_drift: dynamic weight update attempted for {len(newly_backfilled_idx)} newly-backfilled row(s).")


def check_drift_and_manage_retrain_flag():
    if not PRED_LOG_PATH.exists():
        print("evaluate_drift: no prediction log found — nothing to evaluate.")
        return

    df = pd.read_csv(PRED_LOG_PATH)
    if len(df) == 0:
        print("evaluate_drift: prediction log is empty — nothing to evaluate.")
        return

    df, newly_backfilled_idx = backfill_actuals(df)
    df.to_csv(PRED_LOG_PATH, index=False)
    print(f"evaluate_drift: backfilled actual_close for {len(newly_backfilled_idx)} row(s).")

    # NEW: feed ENH#2's dynamic weighting tracker with today's newly-known outcomes
    _update_dynamic_weights_for_backfilled_rows(df, newly_backfilled_idx)

    rolling_acc = compute_rolling_accuracy(df)
    print(f"evaluate_drift: rolling directional accuracy ({ROLLING_ACCURACY_WINDOW}-day) = {rolling_acc}")

    evaluated = df.dropna(subset=["actual_close", "pred_close_price"]).copy()
    errors = (evaluated["pred_close_price"] - evaluated["actual_close"]).abs() / evaluated["actual_close"]
    drift_detected = page_hinkley_test(errors.tail(60))
    print(f"evaluate_drift: Page-Hinkley drift detected = {drift_detected}")

    retrain_needed = False
    reasons = []
    if rolling_acc is not None and rolling_acc < DIRECTIONAL_ACC_MIN:
        retrain_needed = True
        reasons.append(f"rolling_accuracy {rolling_acc:.3f} < threshold {DIRECTIONAL_ACC_MIN}")
    if drift_detected:
        retrain_needed = True
        reasons.append("Page-Hinkley drift detected on prediction errors")

    drift_status = {
        "evaluated_at": pd.Timestamp.now().isoformat(),
        "rolling_accuracy": rolling_acc,
        "drift_detected": bool(drift_detected),
        "retrain_needed": retrain_needed,
        "reasons": reasons,
    }
    with open(DRIFT_FLAG_PATH, "w") as f:
        json.dump(drift_status, f, indent=2, default=str)

    if retrain_needed:
        RETRAIN_FLAG_PATH.touch()
        print(f"evaluate_drift: RETRAIN_NEEDED.flag set — reasons: {reasons}")
    else:
        if RETRAIN_FLAG_PATH.exists():
            RETRAIN_FLAG_PATH.unlink()
        print("evaluate_drift: no retrain trigger — model performance within acceptable range.")

    return drift_status


if __name__ == "__main__":
    check_drift_and_manage_retrain_flag()
