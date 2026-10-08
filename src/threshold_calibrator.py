"""
Regime-based dynamic threshold calibrator.
Calibrates PROB_LONG / PROB_SHORT independently per HMM regime,
optimizing for F1 score on validation data.
Ported from HSI_Colab.py ENH#1.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

from config import MODEL_DIR

DEFAULT_PROB_LONG = 0.60
DEFAULT_PROB_SHORT = 0.40
THRESHOLD_CALIBRATION_PATH = MODEL_DIR / "regime_thresholds.json"


class RegimeThresholdCalibrator:
    """為每個 HMM regime 獨立 calibrate PROB_LONG / PROB_SHORT。
    用 validation set F1 score 做優化目標。"""

    def __init__(self, regimes=("BULL", "NEUTRAL", "BEAR")):
        self.regimes = regimes
        self.thresholds = {
            r: {"long": DEFAULT_PROB_LONG, "short": DEFAULT_PROB_SHORT,
                "f1": 0.0, "n": 0}
            for r in regimes
        }
        self.fitted = False

    def _score_threshold(self, probs, actuals, long_thresh, short_thresh):
        signals = np.where(probs >= long_thresh, 1,
                            np.where(probs <= short_thresh, -1, 0))
        mask = signals != 0
        if mask.sum() < 10:
            return 0.0
        y_pred = (signals[mask] == 1).astype(int)
        y_true = np.array(actuals)[mask]
        if len(np.unique(y_true)) < 2:
            return 0.0
        return f1_score(y_true, y_pred, zero_division=0)

    def calibrate(self, val_df: pd.DataFrame,
                  prob_col: str = "p_up",
                  target_col: str = "actual_direction",
                  regime_col: str = "regime"):
        if len(val_df) < 30:
            print("  [Calibrator] 樣本不足，使用 default threshold")
            return self

        for regime in self.regimes:
            rdf = val_df[val_df[regime_col] == regime]
            if len(rdf) < 20:
                continue

            probs = rdf[prob_col].values
            actuals = rdf[target_col].values
            best_long, best_short, best_score = DEFAULT_PROB_LONG, DEFAULT_PROB_SHORT, -1.0

            for lt in np.arange(0.52, 0.76, 0.02):
                for st in np.arange(0.28, 0.49, 0.02):
                    if st >= lt:
                        continue
                    score = self._score_threshold(probs, actuals, lt, st)
                    if score > best_score:
                        best_score, best_long, best_short = score, lt, st

            self.thresholds[regime] = {
                "long": round(float(best_long), 2),
                "short": round(float(best_short), 2),
                "f1": round(float(best_score), 4),
                "n": len(rdf),
            }
            print(f"  [Calibrator] {regime:8s}: long={best_long:.2f} "
                  f"short={best_short:.2f} F1={best_score:.4f} n={len(rdf)}")

        self.fitted = True
        return self

    def get_signal(self, prob: float, regime: str) -> int:
        t = self.thresholds.get(regime, self.thresholds.get("NEUTRAL"))
        if prob >= t["long"]:
            return 1
        elif prob <= t["short"]:
            return -1
        return 0

    def save(self, path: Path = THRESHOLD_CALIBRATION_PATH):
        with open(path, "w") as f:
            json.dump({"thresholds": self.thresholds, "fitted": self.fitted}, f, indent=2)

    @classmethod
    def load(cls, path: Path = THRESHOLD_CALIBRATION_PATH):
        instance = cls()
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            instance.thresholds = data["thresholds"]
            instance.fitted = data["fitted"]
        return instance
