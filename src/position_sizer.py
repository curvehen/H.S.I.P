"""
Position Sizer — Fractional Kelly criterion position sizing for HSI /
constituent / watchlist trading signals (ENH#3).
"""

import numpy as np

from config import (
    KELLY_FRACTION,
    MAX_POSITION_PCT,
    MIN_POSITION_PCT_FLOOR,
    KELLY_MIN_EDGE,
    REGIME_VOL_DAMPENING,
)


def _directional_win_probability(p_up: float, calibrated_signal: str) -> float:
    """Converts the model's P(up) into a directional win probability
    matching the calibrated signal direction. LONG wins if price goes up;
    SHORT wins if price goes down."""
    if calibrated_signal == "LONG":
        return p_up
    elif calibrated_signal == "SHORT":
        return 1 - p_up
    return 0.5


def _raw_kelly_fraction(p_win: float, risk_reward_ratio: float) -> float:
    """Standard Kelly formula: f* = p - (1-p)/b, where b is the
    risk_reward_ratio (reward/risk). Returns 0 if the edge is negative."""
    if risk_reward_ratio is None or risk_reward_ratio <= 0:
        return 0.0
    b = risk_reward_ratio
    p = p_win
    q = 1 - p
    raw = p - (q / b)
    return max(raw, 0.0)


def _regime_vol_dampener(regime: str) -> float:
    """NEUTRAL regime gets an extra size trim (lower conviction environment).
    BULL/BEAR regimes trade at full calculated size."""
    if not REGIME_VOL_DAMPENING:
        return 1.0
    if regime == "NEUTRAL":
        return 0.6
    return 1.0


class PositionSizer:
    def __init__(self, kelly_fraction: float = KELLY_FRACTION,
                 max_position_pct: float = MAX_POSITION_PCT,
                 min_position_pct_floor: float = MIN_POSITION_PCT_FLOOR,
                 kelly_min_edge: float = KELLY_MIN_EDGE):
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_position_pct_floor = min_position_pct_floor
        self.kelly_min_edge = kelly_min_edge

    def compute(self, p_up: float, calibrated_signal: str, signal_confidence: float,
                risk_reward_ratio: float, regime: str = "NEUTRAL",
                worth_trading: bool = True) -> dict:
        if not worth_trading or calibrated_signal == "觀望":
            return {
                "position_size_pct": 0.0,
                "direction": calibrated_signal,
                "p_win": None,
                "raw_kelly_fraction": 0.0,
                "fractional_kelly": 0.0,
                "confidence_scaled_kelly": 0.0,
                "regime_dampener": None,
                "capped": False,
                "reason": "worth_trading=False or signal=觀望",
            }

        p_win = _directional_win_probability(p_up, calibrated_signal)
        raw_kelly = _raw_kelly_fraction(p_win, risk_reward_ratio)

        if raw_kelly <= self.kelly_min_edge:
            return {
                "position_size_pct": 0.0,
                "direction": calibrated_signal,
                "p_win": p_win,
                "raw_kelly_fraction": raw_kelly,
                "fractional_kelly": 0.0,
                "confidence_scaled_kelly": 0.0,
                "regime_dampener": None,
                "capped": False,
                "reason": f"raw_kelly {raw_kelly:.4f} <= min_edge {self.kelly_min_edge}",
            }

        fractional_kelly = raw_kelly * self.kelly_fraction
        confidence_scaled = fractional_kelly * signal_confidence
        dampener = _regime_vol_dampener(regime)
        dampened_size = confidence_scaled * dampener

        capped = dampened_size > self.max_position_pct
        final_size = min(dampened_size, self.max_position_pct)

        if 0 < final_size < self.min_position_pct_floor:
            final_size = self.min_position_pct_floor

        return {
            "position_size_pct": round(final_size, 4),
            "direction": calibrated_signal,
            "p_win": round(p_win, 4),
            "raw_kelly_fraction": round(raw_kelly, 4),
            "fractional_kelly": round(fractional_kelly, 4),
            "confidence_scaled_kelly": round(confidence_scaled, 4),
            "regime_dampener": dampener,
            "capped": capped,
            "reason": "computed",
        }


_default_sizer = PositionSizer()


def compute_position_size(p_up: float, calibrated_signal: str, signal_confidence: float,
                           risk_reward_ratio: float, regime: str = "NEUTRAL",
                           worth_trading: bool = True) -> dict:
    """Module-level convenience wrapper around a shared default PositionSizer
    instance — used by predict.py and run_daily_backtest.py."""
    return _default_sizer.compute(
        p_up=p_up,
        calibrated_signal=calibrated_signal,
        signal_confidence=signal_confidence,
        risk_reward_ratio=risk_reward_ratio,
        regime=regime,
        worth_trading=worth_trading,
    )
