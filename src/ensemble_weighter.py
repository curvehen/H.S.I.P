"""
Dynamic ensemble weighting using rolling Brier Score.
Adjusts weights between global model and regime-specific model
based on recent forecasting skill. Ported from HSI_Colab.py ENH#5.
"""

import json
import numpy as np
from collections import deque
from pathlib import Path
from config import MODEL_DIR

ENSEMBLE_WEIGHTS_STATE_PATH = MODEL_DIR / "ensemble_weighter_state.json"


class DynamicEnsembleWeighter:
    """根據各模型近期 rolling Brier Score 動態調整 ensemble 權重。
    Brier Score 越低 → skill 越高 → 權重越大。"""

    def __init__(self, window: int = 20, min_weight: float = 0.1):
        self.window = window
        self.min_weight = min_weight
        self.histories = {
            "global": deque(maxlen=window),
            "regime": deque(maxlen=window),
        }

    def update(self, model_name: str, prob: float, actual: int):
        if model_name in self.histories:
            self.histories[model_name].append((float(prob), int(actual)))

    def _rolling_brier(self, history) -> float:
        if len(history) < 5:
            return 0.25
        probs = np.array([h[0] for h in history])
        actuals = np.array([h[1] for h in history])
        return float(np.mean((probs - actuals) ** 2))

    def get_weights(self) -> dict:
        scores = {}
        for name, hist in self.histories.items():
            bs = self._rolling_brier(hist)
            skill = max(0.0, 1.0 - bs / 0.25)
            scores[name] = skill

        total = sum(scores.values())
        if total < 1e-9:
            return {k: 1.0 / len(scores) for k in scores}

        weights = {k: max(v / total, self.min_weight) for k, v in scores.items()}
        total_w = sum(weights.values())
        return {k: v / total_w for k, v in weights.items()}

    def ensemble(self, g_prob: float, r_prob: float) -> float:
        w = self.get_weights()
        result = w.get("global", 0.5) * g_prob + w.get("regime", 0.5) * r_prob
        return float(np.clip(result, 0.0, 1.0))

    def save(self, path: Path = ENSEMBLE_WEIGHTS_STATE_PATH):
        state = {k: list(v) for k, v in self.histories.items()}
        with open(path, "w") as f:
            json.dump({"window": self.window, "min_weight": self.min_weight,
                       "histories": state}, f, indent=2)

    @classmethod
    def load(cls, path: Path = ENSEMBLE_WEIGHTS_STATE_PATH):
        instance = cls()
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            instance.window = data["window"]
            instance.min_weight = data["min_weight"]
            for name, hist in data["histories"].items():
                instance.histories[name] = deque(
                    [tuple(h) for h in hist], maxlen=instance.window)
        return instance
