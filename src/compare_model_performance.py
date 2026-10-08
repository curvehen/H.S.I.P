"""
Safety gate for automated retraining: compares the newly trained model's
walk-forward validation performance against the currently deployed model's
recorded performance (as checked out from git HEAD, before training ran).

Decision logic:
- PROCEED (safe to commit+push the new model) if:
    new_directional_accuracy >= old_directional_accuracy - ACCURACY_TOLERANCE
    AND
    new_rmse <= old_rmse * (1 + RMSE_TOLERANCE)
- ROLLBACK (discard new model, keep old one, flag for manual review) otherwise.
- If no previous metrics exist (first-ever run) or metrics are unreadable,
  PROCEED by default (nothing to compare against / fail open for bootstrap).

Reads:
  models/metrics_previous_backup.json  (old model, copied from git HEAD before training)
  models/metrics.json                   (new model, just produced by train_model.py)

Writes:
  models/retrain_decision.json          (full comparison summary)

Prints a final line "DECISION=proceed" or "DECISION=rollback" that the
calling workflow parses to branch its commit/rollback steps.
"""

import json
import sys
from pathlib import Path

from config import MODEL_DIR

METRICS_PATH = MODEL_DIR / "metrics.json"
PREVIOUS_METRICS_PATH = MODEL_DIR / "metrics_previous_backup.json"
DECISION_PATH = MODEL_DIR / "retrain_decision.json"

ACCURACY_TOLERANCE = 0.03   # allow up to 3 percentage points degradation
RMSE_TOLERANCE = 0.15       # allow up to 15% RMSE increase


def _avg_from_windows(report, key):
    """walk_forward_report is expected to contain a list of per-window dicts
    (see walk_forward.py::expanding_window_validation). Handles both a
    top-level list and a {'windows': [...]} wrapper defensively."""
    if report is None:
        return None
    windows = report.get("windows") if isinstance(report, dict) else report
    if not isinstance(windows, list):
        return None
    vals = [w.get(key) for w in windows if isinstance(w, dict) and w.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _write_decision(decision: str, **kwargs) -> dict:
    summary = {"decision": decision, **kwargs}
    with open(DECISION_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"DECISION={decision}")
    return summary


def main():
    if not METRICS_PATH.exists():
        _write_decision("rollback", reason="metrics.json missing after training — training likely failed")
        return

    with open(METRICS_PATH) as f:
        new_metrics = json.load(f)

    if not PREVIOUS_METRICS_PATH.exists():
        _write_decision("proceed", reason="no previous model to compare against (first-ever training run)")
        return

    try:
        with open(PREVIOUS_METRICS_PATH) as f:
            old_metrics = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        _write_decision("proceed", reason=f"previous metrics unreadable ({e}), failing open")
        return

    old_wf = old_metrics.get("hsi_model", {}).get("walk_forward_report")
    new_wf = new_metrics.get("hsi_model", {}).get("walk_forward_report")

    old_acc = _avg_from_windows(old_wf, "directional_accuracy")
    new_acc = _avg_from_windows(new_wf, "directional_accuracy")
    old_rmse = _avg_from_windows(old_wf, "rmse")
    new_rmse = _avg_from_windows(new_wf, "rmse")

    if old_acc is None or new_acc is None:
        _write_decision("proceed", reason="could not extract directional_accuracy from "
                                           "walk_forward_report, cannot compare — failing open")
        return

    acc_ok = new_acc >= (old_acc - ACCURACY_TOLERANCE)
    rmse_ok = True
    if old_rmse is not None and new_rmse is not None and old_rmse > 0:
        rmse_ok = new_rmse <= old_rmse * (1 + RMSE_TOLERANCE)

    proceed = acc_ok and rmse_ok

    _write_decision(
        "proceed" if proceed else "rollback",
        old_directional_accuracy=round(old_acc, 4),
        new_directional_accuracy=round(new_acc, 4),
        accuracy_tolerance=ACCURACY_TOLERANCE,
        accuracy_check_passed=acc_ok,
        old_rmse=round(old_rmse, 2) if old_rmse is not None else None,
        new_rmse=round(new_rmse, 2) if new_rmse is not None else None,
        rmse_tolerance_pct=RMSE_TOLERANCE * 100,
        rmse_check_passed=rmse_ok,
    )


if __name__ == "__main__":
    main()
    sys.exit(0)  # always exit 0 — the workflow branches on DECISION=, not exit code
