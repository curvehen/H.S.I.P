"""
Fractional Kelly Criterion position sizing.
Computes position size fraction based on prediction probability,
regime, and rolling accuracy. Ported from HSI_Colab.py ENH#3.
"""

import numpy as np


class PositionSizer:
    """根據預測概率動態計算倉位比例。方法: Fractional Kelly Criterion。"""

    def __init__(self, method: str = "fractional_kelly",
                 max_position: float = 1.0,
                 kelly_fraction: float = 0.25,
                 min_edge: float = 0.02):
        self.method = method
        self.max_position = max_position
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge

    def compute_size(self, prob: float, signal: int,
                      regime: str = "NEUTRAL",
                      roll_acc: float = 0.5) -> float:
        if signal == 0:
            return 0.0

        win_prob = prob if signal == 1 else (1.0 - prob)
        edge = win_prob - 0.5

        if edge < self.min_edge:
            return 0.0

        if self.method == "fractional_kelly":
            size = edge * self.kelly_fraction
        elif self.method == "prob_linear":
            size = (edge * 2) * self.max_position
        else:
            size = self.max_position

        regime_mult = {"BULL": 1.10, "NEUTRAL": 1.00, "BEAR": 0.70}.get(regime, 1.0)
        acc_mult = max(0.3, min(1.5, roll_acc / 0.5))

        size = size * regime_mult * acc_mult
        return float(np.clip(size, 0.0, self.max_position))
