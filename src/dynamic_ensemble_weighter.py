"""
Dynamic Ensemble Weighter (ENH#2) — Rolling Brier Score Weighting

Rationale: ensemble_model.HSIEnsembleModel currently computes STATIC weights
once at training time, using inverse validation RMSE. That weighting can go
stale as market conditions drift between retraining cycles (which may be
weeks apart). This module maintains a LIVE, continuously-updated weight set
based on each sub-model's (LightGBM / RandomForest / Ridge) recent real-world
directional reliability, measured via rolling Brier score — a stricter,
probability-calibration-aware metric than raw RMSE.

Why Brier score instead of RMSE for this layer:
  - RMSE on return magnitude rewards models that are merely "less wrong in
    scale", not necessarily directionally reliable.
  - Brier score here evaluates each sub-model's implied probability of an
    "up" day (derived from its continuous return prediction via a logistic
    transform) against the realized binary up/down outcome — directly
    measuring directional calibration, which is what actually matters for
    the trading signal.
  - Lower Brier score = better calibrated = higher ensemble weight.

Lifecycle:
  1. train_model.py calls init_dynamic_weights() right after final model
     training — seeds equal weights (no live history yet).
  2. evaluate_drift.py (post-market, once actuals are known) calls
     update_dynamic_weights() daily — appends each sub-model's Brier score
     for that day, recomputes weights from a trailing rolling window.
  3. predict.py / ensemble_model.py call load_dynamic_weights() at
     inference time; if too few live samples have accumulated yet, the
     blend falls back to the ensemble's static RMSE-based weights (see
     get_effective_weights()) to avoid acting on a thin, noisy history.
"""

import json
import math
import numpy as np
import pandas as pd

from config import (MODEL_DIR, DYNAMIC_WEIGHTS_PATH, DYNAMIC_WEIGHT_ROLLING_WINDOW,
                     DYNAMIC_WEIGHT_MIN_SAMPLES, DYNAMIC_WEIGHT_PROB_SCALE)

SUBMODEL_NAMES = ["lightgbm", "random_forest", "ridge"]
EPSILON = 1e-6  # floor to avoid division by zero when a model has a perfect (zero) Brier score


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _return_to_prob_up(pred_return: float, scale: float = DYNAMIC_WEIGHT_PROB_SCALE) -> float:
    """Converts a sub-model's continuous predicted return into an implied
    probability of an 'up' day via a logistic transform centered at 0.
    `scale` controls steepness: smaller scale => a given return magnitude
    maps to a more extreme (closer to 0 or 1) probability. Default scale
    (1%) means a +1% predicted return maps to ~73% implied prob-up."""
    try:
        return 1.0 / (1.0 + math.exp(-pred_return / scale))
    except OverflowError:
        return 1.0 if pred_return > 0 else 0.0


def _brier_score(prob_up: float, actual_up: int) -> float:
    """Standard Brier score for a single binary forecast: (prob - outcome)^2.
    Range [0, 1], lower is better. 0 = perfect, 0.25 = uninformative (always
    predicting 0.5), 1 = maximally wrong."""
    return (prob_up - actual_up) ** 2


def _weights_from_brier_history(history: dict) -> dict:
    """Converts each model's rolling list of recent Brier scores into a
    normalized weight via inverse-mean-Brier, so models with lower
    (better) average recent Brier score get proportionally higher weight."""
    inv_scores = {}
    for model, scores in history.items():
        if not scores:
            inv_scores[model] = 1.0 / len(history)  # no history yet -> neutral
            continue
        mean_brier = float(np.mean(scores))
        inv_scores[model] = 1.0 / (mean_brier + EPSILON)

    total = sum(inv_scores.values())
    if total <= 0:
        n = len(history)
        return {m: 1.0 / n for m in history}
    return {m: v / total for m, v in inv_scores.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_dynamic_weights(models: list = None, output_path=None) -> dict:
    """
    Called once by train_model.py right after final model training.
    Seeds equal weights with empty rolling Brier history — the live
    pipeline has no real-world outcomes yet to weight by, so every
    sub-model starts on equal footing until update_dynamic_weights()
    accumulates enough daily samples.
    """
    models = models or SUBMODEL_NAMES
    output_path = output_path or DYNAMIC_WEIGHTS_PATH

    n = len(models)
    payload = {
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "rolling_window": DYNAMIC_WEIGHT_ROLLING_WINDOW,
        "min_samples_required": DYNAMIC_WEIGHT_MIN_SAMPLES,
        "n_samples": 0,
        "weights": {m: 1.0 / n for m in models},
        "brier_history": {m: [] for m in models},
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"DYNAMIC_ENSEMBLE: initialized equal weights {payload['weights']} -> {output_path}")
    return payload


def update_dynamic_weights(model_predictions: dict, actual_return: float,
                            output_path=None, window: int = DYNAMIC_WEIGHT_ROLLING_WINDOW) -> dict:
    """
    Called once per trading day from evaluate_drift.py, AFTER the actual
    close is known (post-market backfill step). Appends each sub-model's
    Brier score for that day and recomputes rolling weights.

    Args:
        model_predictions: {'lightgbm': pred_return, 'random_forest': pred_return,
                             'ridge': pred_return} — each sub-model's predicted
                             return for the day that just closed.
        actual_return: the realized next_close_return for that same day.
        output_path: defaults to config.DYNAMIC_WEIGHTS_PATH.
        window: trailing number of days kept per model (older entries are
                dropped so weighting tracks RECENT reliability, not all-time).

    Returns: the updated payload dict (also persisted to disk).

    Fail-safe: if the weights file doesn't exist yet (e.g. update called
    before any init_dynamic_weights() run), it self-initializes with equal
    weights for whatever models are present in model_predictions.
    """
    output_path = output_path or DYNAMIC_WEIGHTS_PATH

    try:
        with open(output_path) as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"DYNAMIC_ENSEMBLE: {output_path} missing/corrupt — self-initializing.")
        payload = init_dynamic_weights(models=list(model_predictions.keys()), output_path=output_path)

    actual_up = 1 if actual_return > 0 else 0
    history = payload.get("brier_history", {})

    for model, pred_return in model_predictions.items():
        if pred_return is None or (isinstance(pred_return, float) and np.isnan(pred_return)):
            continue  # skip models that failed to produce a prediction that day
        prob_up = _return_to_prob_up(pred_return)
        score = _brier_score(prob_up, actual_up)

        history.setdefault(model, [])
        history[model].append(score)
        history[model] = history[model][-window:]  # keep only the trailing `window` days

    payload["brier_history"] = history
    payload["weights"] = _weights_from_brier_history(history)
    payload["n_samples"] = max(len(v) for v in history.values()) if history else 0
    payload["updated_at"] = pd.Timestamp.utcnow().isoformat()
    payload["rolling_window"] = window

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"DYNAMIC_ENSEMBLE: updated weights={payload['weights']} "
          f"(n_samples={payload['n_samples']})")
    return payload


def load_dynamic_weights(path=None) -> dict:
    """Fail-safe loader for predict.py / ensemble_model.py at inference time.
    Returns an all-equal-weight fallback structure if the file is missing,
    corrupted, or empty — the live pipeline must never crash because this
    file hasn't been generated yet (e.g. brand-new repo before first
    train_model.py run)."""
    path = path or DYNAMIC_WEIGHTS_PATH
    try:
        with open(path) as f:
            payload = json.load(f)
        if not payload.get("weights"):
            raise ValueError("empty weights field")
        return payload
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"DYNAMIC_ENSEMBLE: could not load {path} ({e}) — using equal-weight fallback.")
        n = len(SUBMODEL_NAMES)
        return {
            "n_samples": 0,
            "weights": {m: 1.0 / n for m in SUBMODEL_NAMES},
            "brier_history": {m: [] for m in SUBMODEL_NAMES},
        }


def get_effective_weights(static_weights: dict, path=None,
                           min_samples: int = DYNAMIC_WEIGHT_MIN_SAMPLES) -> dict:
    """
    Decides which weight set to actually use for blending predictions:

      - If the live dynamic-weight history has accumulated at least
        `min_samples` daily observations, use the rolling-Brier dynamic
        weights (reflects current real-world reliability).
      - Otherwise, fall back to the ensemble's STATIC inverse-RMSE weights
        computed at training time (ensemble_model.HSIEnsembleModel) — too
        little live history to trust a noisy dynamic estimate yet.

    This is the single function ensemble_model.py / predict.py should call
    at inference time rather than reading either source directly.
    """
    dynamic_payload = load_dynamic_weights(path)
    n_samples = dynamic_payload.get("n_samples", 0)

    if n_samples >= min_samples:
        return dynamic_payload["weights"]

    print(f"DYNAMIC_ENSEMBLE: only {n_samples} live samples (< {min_samples}) "
          f"— falling back to static training-time weights {static_weights}.")
    return static_weights


def apply_weights(model_predictions: dict, weights: dict) -> float:
    """Computes the final weighted-average prediction from each sub-model's
    raw prediction and a weight dict (either dynamic or static). Skips any
    model missing from `model_predictions` and renormalizes over the
    remaining ones, so a single sub-model failing at inference time doesn't
    break the whole ensemble blend."""
    available = {m: p for m, p in model_predictions.items()
                 if p is not None and not (isinstance(p, float) and np.isnan(p))}
    if not available:
        raise ValueError("apply_weights: no valid sub-model predictions available.")

    relevant_weights = {m: weights.get(m, 0.0) for m in available}
    total = sum(relevant_weights.values())
    if total <= 0:
        n = len(available)
        relevant_weights = {m: 1.0 / n for m in available}
        total = 1.0

    return sum(available[m] * (relevant_weights[m] / total) for m in available)
