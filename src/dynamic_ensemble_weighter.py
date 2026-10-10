"""
dynamic_ensemble_weighter.py — ENH#2

Rolling Brier-score dynamic ensemble weighting for the HSI close-return
ensemble (LightGBM / Random Forest / Ridge).

Rationale
---------
HSIEnsembleModel's static weights (inverse validation RMSE, frozen at
training time) cannot adapt if one sub-model's relative skill drifts over
time — e.g. Ridge may degrade in a high-volatility regime shift while
LightGBM stays robust. This module maintains a ROLLING per-sub-model Brier
score (on the directional "up" call, not the raw regression value) and
re-weights the ensemble daily, inversely proportional to each sub-model's
recent Brier score — without ever requiring a model refit.

Why Brier score on a regression output
---------------------------------------
Each sub-model emits a predicted next-close RETURN (not a probability).
To score it against the realized binary "closed up / closed down" outcome
with Brier methodology, the predicted return is squashed through a logistic
function into a pseudo-probability of "up":
    p_up_hat = sigmoid(k * pred_return)
Brier score for that day = (p_up_hat - actual_up)^2, where actual_up is 1
if next_close_return > 0 else 0. Lower Brier = better-calibrated direction
call = higher resulting weight.

Update cadence
---------------
update_dynamic_weights() is called once per day from evaluate_drift.py,
AFTER the actual next-day close becomes known (i.e. one day after
predict.py logged the sub-model predictions) — never at prediction time,
since the actual outcome isn't known yet then.

Fail-safe behaviour
--------------------
- Before any model has accumulated DYNAMIC_WEIGHT_MIN_HISTORY days of
  history, get_effective_weights() returns the STATIC weights unchanged.
- A corrupted/missing weights file is treated as "no history yet" —
  falls back to static weights, never raises.
- init_dynamic_weights() is idempotent: calling it when a file already
  exists leaves existing history untouched (it will NOT wipe accumulated
  Brier history on every training run).
"""

import json
import math
import numpy as np

from config import (DYNAMIC_WEIGHTS_PATH, DYNAMIC_WEIGHT_ROLLING_WINDOW,
                     DYNAMIC_WEIGHT_MIN_HISTORY, DYNAMIC_WEIGHT_SIGMOID_K)


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _pseudo_prob_up(pred_return: float) -> float:
    """Squashes a predicted return into a pseudo-probability of 'up'."""
    return _sigmoid(DYNAMIC_WEIGHT_SIGMOID_K * pred_return)


def _load_state() -> dict:
    if not DYNAMIC_WEIGHTS_PATH.exists():
        return {}
    try:
        with open(DYNAMIC_WEIGHTS_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict):
    with open(DYNAMIC_WEIGHTS_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def init_dynamic_weights(models: list):
    """
    Called once from train_model.py Step 5.6. Idempotent: if the weights
    file already exists, existing per-model Brier history is preserved
    untouched — re-running training must not reset live-accumulated
    calibration history.
    """
    state = _load_state()
    changed = False
    for model_name in models:
        if model_name not in state:
            state[model_name] = {"brier_history": [], "last_weight": None}
            changed = True
    if changed or not DYNAMIC_WEIGHTS_PATH.exists():
        _save_state(state)


def update_dynamic_weights(sub_model_preds: dict, actual_return: float):
    """
    Entry point called from evaluate_drift.py once the actual next-day
    close return becomes known.

    Args:
        sub_model_preds: {"lightgbm": pred_return, "random_forest": ...,
                           "ridge": ...} — the SAME raw sub-model predictions
                           that predict.py logged the previous day.
        actual_return: realized next_close_return (float).
    """
    state = _load_state()
    actual_up = 1.0 if actual_return > 0 else 0.0

    for model_name, pred_return in sub_model_preds.items():
        if pred_return is None:
            continue
        if model_name not in state:
            state[model_name] = {"brier_history": [], "last_weight": None}

        p_up_hat = _pseudo_prob_up(float(pred_return))
        brier = (p_up_hat - actual_up) ** 2

        history = state[model_name]["brier_history"]
        history.append(brier)
        state[model_name]["brier_history"] = history[-DYNAMIC_WEIGHT_ROLLING_WINDOW:]

    # Recompute weights for every model that now has enough history.
    eligible = {m: s for m, s in state.items() if len(s["brier_history"]) >= DYNAMIC_WEIGHT_MIN_HISTORY}
    if len(eligible) == len(state) and len(state) > 0:
        inv_brier = {m: 1.0 / (np.mean(s["brier_history"]) + 1e-6) for m, s in eligible.items()}
        total = sum(inv_brier.values())
        for m in state:
            state[m]["last_weight"] = inv_brier[m] / total if total > 0 else None
    else:
        for m in state:
            state[m]["last_weight"] = None  # not all models warmed up yet

    _save_state(state)


def get_effective_weights(static_weights: dict) -> tuple:
    """
    Entry point called from predict.py on every daily prediction.

    Returns:
        (weights_dict, source_str) where source_str is "dynamic_brier" if
        every sub-model has a valid rolling weight, otherwise
        "static_fallback" (returning static_weights unchanged).
    """
    state = _load_state()

    if not state:
        return static_weights, "static_fallback"

    dynamic_weights = {}
    for model_name in static_weights:
        entry = state.get(model_name)
        if entry is None or entry.get("last_weight") is None:
            return static_weights, "static_fallback"
        dynamic_weights[model_name] = entry["last_weight"]

    total = sum(dynamic_weights.values())
    if total <= 0:
        return static_weights, "static_fallback"

    normalized = {m: w / total for m, w in dynamic_weights.items()}
    return normalized, "dynamic_brier"
