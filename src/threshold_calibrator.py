"""
threshold_calibrator.py — ENH#1

Regime-specific trading threshold calibration via out-of-fold (OOF) F1-score
grid search.

Rationale
---------
A single global (MIN_EXPECTED_MOVE_PCT, MIN_CONFIDENCE) pair is a crude
approximation: a BULL regime with persistent momentum may support firing at
a lower confidence bar than a choppy NEUTRAL regime, where false positives
are far more costly. This module fits a SEPARATE threshold pair per market
regime (BULL / BEAR / NEUTRAL), each one chosen to maximize the F1 score of
the "fire-and-hit" binary classification problem defined below, evaluated
on genuinely out-of-fold predictions (never in-sample — see
train_model.generate_oof_predictions()).

Binary classification framing
------------------------------
For a candidate threshold pair (move_thresh, conf_thresh):
    fires      = (expected_move_pct >= move_thresh) and (confidence >= conf_thresh)
    TP = fires & actual_hit
    FP = fires & not actual_hit
    FN = not fires & actual_hit
    precision = TP / (TP + FP)   -> "when we fire, how often are we right"
    recall    = TP / (TP + FN)   -> "of all winnable days, how many did we take"
    F1        = 2 * precision * recall / (precision + recall)

We search a grid of move_thresh values (percentiles of that regime's
expected_move_pct distribution) x conf_thresh values (percentiles of that
regime's confidence distribution), and keep the pair maximizing F1 — subject
to a minimum-fired-trades floor so a degenerate threshold that only fires on
1-2 lucky rows can't "win" with an artificially perfect F1.

Fail-safe behaviour
--------------------
- A regime with fewer than REGIME_CALIBRATION_MIN_SAMPLES_PER_REGIME OOF
  rows is NOT calibrated: it falls back to the static global
  MIN_EXPECTED_MOVE_PCT / MIN_CONFIDENCE values (REGIME_CALIBRATION_FALLBACK).
- If no threshold pair in the grid clears the min-fired-trades floor, that
  regime also falls back to the static global defaults.
- If the thresholds file does not exist yet (never calibrated), or is
  corrupted / unreadable, get_regime_threshold() also returns the static
  global fallback — predict.py never crashes because of this module.
"""

import json
import numpy as np
import pandas as pd

from config import (REGIME_THRESHOLDS_PATH, REGIME_CALIBRATION_MIN_SAMPLES_PER_REGIME,
                     REGIME_CALIBRATION_F1_GRID_STEPS, REGIME_CALIBRATION_FALLBACK)

KNOWN_REGIMES = ["BULL", "BEAR", "NEUTRAL"]


def _f1_for_threshold(df: pd.DataFrame, move_thresh: float, conf_thresh: float) -> dict:
    fires = (df["expected_move_pct"] >= move_thresh) & (df["confidence"] >= conf_thresh)
    hit = df["actual_hit"].astype(bool)

    tp = int((fires & hit).sum())
    fp = int((fires & ~hit).sum())
    fn = int((~fires & hit).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"f1": f1, "precision": precision, "recall": recall, "n_fired": int(fires.sum())}


def _calibrate_single_regime(df_regime: pd.DataFrame) -> dict:
    """Grid-searches (move_thresh, conf_thresh) for a single regime's OOF
    slice, returning the F1-maximizing pair plus its diagnostic metrics."""
    n_steps = REGIME_CALIBRATION_F1_GRID_STEPS

    move_grid = np.unique(np.quantile(df_regime["expected_move_pct"],
                                       np.linspace(0.05, 0.95, n_steps)))
    conf_grid = np.unique(np.quantile(df_regime["confidence"],
                                       np.linspace(0.05, 0.95, n_steps)))

    best = {"f1": -1.0, "MIN_EXPECTED_MOVE_PCT": None, "MIN_CONFIDENCE": None,
            "precision": 0.0, "recall": 0.0, "n_fired": 0}

    min_fired_floor = max(5, int(0.02 * len(df_regime)))

    for move_thresh in move_grid:
        for conf_thresh in conf_grid:
            metrics = _f1_for_threshold(df_regime, float(move_thresh), float(conf_thresh))
            if metrics["n_fired"] < min_fired_floor:
                continue
            if metrics["f1"] > best["f1"]:
                best = {
                    "f1": metrics["f1"],
                    "MIN_EXPECTED_MOVE_PCT": float(move_thresh),
                    "MIN_CONFIDENCE": float(conf_thresh),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "n_fired": metrics["n_fired"],
                }

    if best["MIN_EXPECTED_MOVE_PCT"] is None:
        return {
            **REGIME_CALIBRATION_FALLBACK,
            "source": "fallback_no_viable_threshold",
            "f1": None, "precision": None, "recall": None, "n_fired": 0,
            "n_oof_samples": int(len(df_regime)),
        }

    return {
        "MIN_EXPECTED_MOVE_PCT": best["MIN_EXPECTED_MOVE_PCT"],
        "MIN_CONFIDENCE": best["MIN_CONFIDENCE"],
        "source": "calibrated",
        "f1": best["f1"],
        "precision": best["precision"],
        "recall": best["recall"],
        "n_fired": best["n_fired"],
        "n_oof_samples": int(len(df_regime)),
    }


def fit_and_save_regime_thresholds(oof_df: pd.DataFrame) -> dict:
    """
    Entry point called from train_model.py Step 5.5.

    Args:
        oof_df: DataFrame with columns
            ["regime", "expected_move_pct", "confidence", "actual_hit",
             "pred_return", "actual_return"]
            as produced by train_model.generate_oof_predictions().

    Returns:
        dict keyed by regime name, each value the calibration summary
        (also written to REGIME_THRESHOLDS_PATH as JSON).
    """
    required_cols = {"regime", "expected_move_pct", "confidence", "actual_hit"}
    missing = required_cols - set(oof_df.columns)
    if missing:
        raise ValueError(f"oof_df missing required columns: {missing}")

    summary = {}
    for regime in KNOWN_REGIMES:
        df_regime = oof_df[oof_df["regime"] == regime]
        if len(df_regime) < REGIME_CALIBRATION_MIN_SAMPLES_PER_REGIME:
            summary[regime] = {
                **REGIME_CALIBRATION_FALLBACK,
                "source": "fallback_insufficient_samples",
                "f1": None, "precision": None, "recall": None, "n_fired": 0,
                "n_oof_samples": int(len(df_regime)),
            }
            continue
        summary[regime] = _calibrate_single_regime(df_regime)

    with open(REGIME_THRESHOLDS_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def get_regime_threshold(regime: str) -> dict:
    """
    Entry point called from predict.py for every daily prediction.

    Returns dict with keys MIN_EXPECTED_MOVE_PCT, MIN_CONFIDENCE, source.
    Always returns a usable dict — never raises, never returns None — so
    predict.py's fail-safe wrapper is a pure defensive formality.
    """
    fallback = {**REGIME_CALIBRATION_FALLBACK, "source": "static_fallback"}

    if not REGIME_THRESHOLDS_PATH.exists():
        return fallback

    try:
        with open(REGIME_THRESHOLDS_PATH) as f:
            all_thresholds = json.load(f)
    except (json.JSONDecodeError, OSError):
        return fallback

    regime_entry = all_thresholds.get(regime)
    if not regime_entry or regime_entry.get("MIN_EXPECTED_MOVE_PCT") is None:
        return fallback

    return {
        "MIN_EXPECTED_MOVE_PCT": regime_entry["MIN_EXPECTED_MOVE_PCT"],
        "MIN_CONFIDENCE": regime_entry["MIN_CONFIDENCE"],
        "source": regime_entry.get("source", "calibrated"),
    }
