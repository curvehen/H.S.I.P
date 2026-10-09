"""
Regime-Based Threshold Calibrator (ENH#1)

Rationale: a single global (MIN_EXPECTED_MOVE_PCT, MIN_CONFIDENCE) pair is
sub-optimal because the model's directional reliability differs by market
regime (BULL/BEAR/NEUTRAL) — e.g. the model may be reliably directional in
BULL regimes even on small predicted moves, but need a much higher bar in
NEUTRAL/choppy regimes to avoid false signals.

This module fits one (expected_move_threshold, confidence_threshold) pair
PER REGIME by grid-searching for the combination that maximizes F1 score on
out-of-fold (OOF) predictions — never on in-sample fitted values, which
would make thresholds look artificially good.

F1 definition used here:
  - "label" (ground truth, positive class) = 1 if the model's raw directional
    call was correct that day (sign(pred_return) == sign(actual_return)),
    regardless of whether a signal would have been issued. This represents
    "a day where taking this trade direction would have been a win".
  - "prediction" = 1 if a signal WOULD be issued under the candidate
    thresholds (|pred_return| >= move_threshold AND confidence >=
    confidence_threshold).
  - TP = correct call AND signal issued (a win we captured)
  - FP = wrong call AND signal issued (a loss we took)
  - FN = correct call AND signal NOT issued (a win we missed by being too strict)
  - TN = wrong call AND signal NOT issued (a loss we correctly avoided)
  F1 = 2*TP / (2*TP + FP + FN)

This balances capturing winning days (recall) against avoiding false
signals (precision) — directly optimizing for trading usefulness rather
than raw accuracy.
"""

import json
import numpy as np
import pandas as pd

from config import MIN_EXPECTED_MOVE_PCT, MIN_CONFIDENCE, REGIME_THRESHOLDS_PATH

VALID_REGIMES = ["BULL", "BEAR", "NEUTRAL"]

# Grid search ranges — centered around the existing global defaults so the
# calibrated per-regime values stay in a sane, interpretable neighborhood.
MOVE_THRESHOLD_GRID = np.round(np.arange(0.002, 0.0161, 0.001), 4)       # 0.2% .. 1.6%
CONFIDENCE_THRESHOLD_GRID = np.round(np.arange(0.50, 0.91, 0.05), 2)     # 0.50 .. 0.90

MIN_SAMPLES_PER_REGIME = 40  # below this, fall back to global defaults — too
                              # few OOF samples to trust a regime-specific fit


def _f1_for_thresholds(pred_return: np.ndarray, actual_return: np.ndarray,
                        confidence: np.ndarray, move_thr: float, conf_thr: float) -> float:
    """Computes F1 score for one candidate (move_thr, conf_thr) pair over a
    regime's OOF sample, per the TP/FP/FN/TN definition above."""
    label = (np.sign(pred_return) == np.sign(actual_return)).astype(int)
    issued = ((np.abs(pred_return) >= move_thr) & (confidence >= conf_thr)).astype(int)

    tp = int(np.sum((label == 1) & (issued == 1)))
    fp = int(np.sum((label == 0) & (issued == 1)))
    fn = int(np.sum((label == 1) & (issued == 0)))

    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


def _grid_search_regime(regime_df: pd.DataFrame) -> dict:
    """Exhaustive grid search over MOVE_THRESHOLD_GRID x CONFIDENCE_THRESHOLD_GRID
    for a single regime's OOF subset. Returns the best-F1 threshold pair plus
    diagnostic counts."""
    pred_return = regime_df["pred_return"].values
    actual_return = regime_df["actual_return"].values
    confidence = regime_df["confidence"].values

    best = {"move_threshold": MIN_EXPECTED_MOVE_PCT, "confidence_threshold": MIN_CONFIDENCE,
            "f1_score": -1.0, "n_signals_issued": 0}

    for move_thr in MOVE_THRESHOLD_GRID:
        for conf_thr in CONFIDENCE_THRESHOLD_GRID:
            f1 = _f1_for_thresholds(pred_return, actual_return, confidence, move_thr, conf_thr)
            if f1 > best["f1_score"]:
                n_issued = int(np.sum((np.abs(pred_return) >= move_thr) & (confidence >= conf_thr)))
                best = {"move_threshold": float(move_thr), "confidence_threshold": float(conf_thr),
                        "f1_score": float(f1), "n_signals_issued": n_issued}

    return best


def fit_and_save_regime_thresholds(oof_df: pd.DataFrame, output_path=None) -> dict:
    """
    Main entry point called from train_model.py.

    Args:
        oof_df: DataFrame with columns ['date', 'regime', 'actual_return',
                'pred_return', 'confidence'] — must be genuinely out-of-fold
                (see train_model.generate_oof_predictions()), not in-sample.
        output_path: where to write the resulting JSON. Defaults to
                config.REGIME_THRESHOLDS_PATH.

    Returns: the full thresholds dict (also written to disk).
    """
    output_path = output_path or REGIME_THRESHOLDS_PATH
    required_cols = {"regime", "actual_return", "pred_return", "confidence"}
    missing = required_cols - set(oof_df.columns)
    if missing:
        raise ValueError(f"fit_and_save_regime_thresholds: oof_df missing columns {missing}")

    results = {}
    for regime in VALID_REGIMES:
        regime_df = oof_df[oof_df["regime"] == regime].dropna(
            subset=["actual_return", "pred_return", "confidence"])
        n = len(regime_df)

        if n < MIN_SAMPLES_PER_REGIME:
            print(f"THRESHOLD_CALIBRATOR[{regime}]: only {n} OOF samples "
                  f"(< {MIN_SAMPLES_PER_REGIME}) — falling back to global defaults.")
            results[regime] = {
                "move_threshold": MIN_EXPECTED_MOVE_PCT,
                "confidence_threshold": MIN_CONFIDENCE,
                "f1_score": None,
                "n_signals_issued": None,
                "n_oof_samples": n,
                "fallback_to_global_default": True,
            }
            continue

        best = _grid_search_regime(regime_df)
        best["n_oof_samples"] = n
        best["fallback_to_global_default"] = False
        results[regime] = best
        print(f"THRESHOLD_CALIBRATOR[{regime}]: n={n}, "
              f"move_thr={best['move_threshold']:.3f}, conf_thr={best['confidence_threshold']:.2f}, "
              f"F1={best['f1_score']:.4f}, signals_issued={best['n_signals_issued']}")

    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "min_samples_per_regime": MIN_SAMPLES_PER_REGIME,
        "global_default_move_threshold": MIN_EXPECTED_MOVE_PCT,
        "global_default_confidence_threshold": MIN_CONFIDENCE,
        "thresholds": results,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return payload


def load_regime_thresholds(path=None) -> dict:
    """Loads the calibrated thresholds JSON. Returns an all-default structure
    (fail-safe) if the file is missing or corrupted — predict.py should
    never crash because calibration hasn't run yet (e.g. a brand-new repo
    before the first train_model.py run)."""
    path = path or REGIME_THRESHOLDS_PATH
    try:
        with open(path) as f:
            payload = json.load(f)
        return payload["thresholds"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"THRESHOLD_CALIBRATOR: could not load {path} ({e}) — using global defaults for all regimes.")
        return {
            regime: {"move_threshold": MIN_EXPECTED_MOVE_PCT, "confidence_threshold": MIN_CONFIDENCE,
                      "fallback_to_global_default": True}
            for regime in VALID_REGIMES
        }


def get_calibrated_signal(regime: str, pred_return: float, confidence: float,
                           thresholds: dict = None) -> str:
    """
    Applies the regime-specific calibrated thresholds to a single live
    prediction, returning one of 'LONG', 'SHORT', '觀望' (HOLD).

    This is the function predict.py / signal_generator.py should call to
    populate result['calibrated_signal'], replacing any single global-
    threshold check.
    """
    thresholds = thresholds if thresholds is not None else load_regime_thresholds()
    regime_thr = thresholds.get(regime, {
        "move_threshold": MIN_EXPECTED_MOVE_PCT, "confidence_threshold": MIN_CONFIDENCE,
    })

    move_thr = regime_thr.get("move_threshold", MIN_EXPECTED_MOVE_PCT)
    conf_thr = regime_thr.get("confidence_threshold", MIN_CONFIDENCE)

    if abs(pred_return) < move_thr or confidence < conf_thr:
        return "觀望"
    return "LONG" if pred_return > 0 else "SHORT"


def get_regime_threshold_summary(thresholds: dict = None) -> pd.DataFrame:
    """Convenience helper for performance_report.py / dashboard_report.py to
    display calibrated thresholds per regime as a readable table."""
    thresholds = thresholds if thresholds is not None else load_regime_thresholds()
    rows = []
    for regime in VALID_REGIMES:
        t = thresholds.get(regime, {})
        rows.append({
            "regime": regime,
            "move_threshold_pct": round(t.get("move_threshold", MIN_EXPECTED_MOVE_PCT) * 100, 2),
            "confidence_threshold": t.get("confidence_threshold", MIN_CONFIDENCE),
            "f1_score": t.get("f1_score"),
            "n_oof_samples": t.get("n_oof_samples"),
            "fallback_to_global_default": t.get("fallback_to_global_default", True),
        })
    return pd.DataFrame(rows)
