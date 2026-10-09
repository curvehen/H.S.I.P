"""
Position Sizer — Fractional Kelly Criterion

Rationale: a binary "trade / don't trade" verdict (worth_trading) ignores
the fact that signal QUALITY varies continuously — a signal with p_up=0.82
and a 3:1 risk-reward ratio deserves a materially larger position than one
that barely clears the worth_trading bar at p_up=0.58 with a 1.2:1 ratio.
The Kelly criterion converts (win probability, payoff ratio) into a
mathematically principled position size that maximizes long-run geometric
growth — but FULL Kelly is known to be too aggressive/volatile for live
trading (highly sensitive to estimation error in p_win), so this module
always applies a configurable fractional multiplier (default: half-Kelly).

Inputs consumed (already computed elsewhere in the pipeline):
  - p_up                 : probability_model.py's P(next close return > 0)
  - calibrated_signal    : 'LONG' / 'SHORT' / '觀望' from threshold_calibrator.py
  - signal_confidence     : confidence.py / meta_labeling.py's meta-model probability
  - risk_reward_ratio     : signal_generator.py's reward/risk ratio
  - regime               : regime.py's BULL/BEAR/NEUTRAL tag (optional vol dampening)

Output: a position_size_pct in [0, MAX_POSITION_PCT], plus a full diagnostic
breakdown so email_report.py / dashboard_report.py can show WHY a given size
was chosen (never a black-box number).

Fail-safe: any malformed/missing input, degenerate risk-reward ratio, or
negative Kelly edge returns 0% position (觀望) rather than raising or
guessing — sizing errors are far more costly than a missed trade.
"""

import numpy as np

from config import (KELLY_FRACTION, MAX_POSITION_PCT, MIN_POSITION_PCT_FLOOR,
                     KELLY_MIN_EDGE, REGIME_VOL_DAMPENING)


# ---------------------------------------------------------------------------
# Core Kelly math
# ---------------------------------------------------------------------------

def _raw_kelly_fraction(p_win: float, win_loss_ratio: float) -> float:
    """
    Classic Kelly formula: f* = (b*p - q) / b
      where b = win_loss_ratio (payoff per \$1 risked if correct),
            p = probability of winning,
            q = 1 - p = probability of losing.

    Returns the UNCAPPED, UNSCALED Kelly fraction. Can be negative (no edge
    — the bet has negative expected value at this probability/payoff combo)
    or occasionally > 1 (full bankroll) with extreme inputs; both are
    clipped by the caller, never here, so this function stays a pure,
    auditable implementation of the textbook formula.
    """
    if win_loss_ratio is None or win_loss_ratio <= 0:
        return 0.0
    q = 1.0 - p_win
    return (win_loss_ratio * p_win - q) / win_loss_ratio


def _directional_win_probability(p_up: float, direction: str) -> float:
    """Converts the model's P(up) into P(win) for the SPECIFIC direction
    being traded. A SHORT signal wins when the market goes down, so its win
    probability is (1 - p_up), not p_up itself — a common sizing bug if
    p_up is used directly regardless of trade direction."""
    if direction == "LONG":
        return p_up
    elif direction == "SHORT":
        return 1.0 - p_up
    else:
        return 0.0  # 觀望 (HOLD) has no directional bet to size


def _regime_vol_dampener(regime: str) -> float:
    """Optional secondary safety multiplier (independent of Kelly's own
    math) applied on top of the fractional-Kelly size. NEUTRAL regimes
    historically show choppier, less trending price action in this
    pipeline's backtests, so positions are proportionally trimmed even when
    Kelly math alone would size them normally — a conservative buffer
    against regime misclassification or whipsaw conditions. Returns 1.0
    (no dampening) if REGIME_VOL_DAMPENING is disabled in config."""
    if not REGIME_VOL_DAMPENING:
        return 1.0
    return {"BULL": 1.0, "BEAR": 1.0, "NEUTRAL": 0.6}.get(regime, 0.8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PositionSizer:
    """
    Stateless sizing engine — holds no mutable state between calls, so a
    single module-level instance can be safely reused across predict.py's
    HSI call and every per-stock call within the same run.
    """

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
        """
        Computes the final recommended position size and a full diagnostic
        breakdown.

        Args:
            p_up: probability_model.py's raw P(up) for next close, in [0, 1].
            calibrated_signal: 'LONG' / 'SHORT' / '觀望' from
                threshold_calibrator.get_calibrated_signal().
            signal_confidence: meta-labeling confidence probability, in [0, 1].
            risk_reward_ratio: signal_generator.py's reward/risk ratio (b in
                the Kelly formula). Must be > 0 to size a position.
            regime: 'BULL' / 'BEAR' / 'NEUTRAL' from regime.py — applies an
                optional secondary volatility dampener (see
                _regime_vol_dampener). Defaults to the most conservative
                tag if unavailable.
            worth_trading: signal_generator.py's existing binary verdict.
                If False, this function short-circuits to 0% regardless of
                the Kelly math — the binary gate is still an upstream veto.

        Returns dict with:
            position_size_pct       : final recommended size, in [0, max_position_pct]
            direction                : 'LONG' / 'SHORT' / 'NONE'
            p_win                    : directional win probability used
            raw_kelly_fraction       : uncapped textbook Kelly output
            fractional_kelly         : raw_kelly_fraction * kelly_fraction
            confidence_scaled_kelly  : fractional_kelly * signal_confidence
            regime_dampener          : multiplier applied for regime
            capped                   : True if the cap/floor logic altered the raw value
            reason                   : human-readable explanation of the result
        """
        direction = calibrated_signal if calibrated_signal in ("LONG", "SHORT") else "NONE"

        if not worth_trading or direction == "NONE":
            return self._zero_result(direction, p_up, reason="觀望訊號或 worth_trading=False，不開倉。")

        if p_up is None or not (0.0 <= p_up <= 1.0):
            return self._zero_result(direction, p_up, reason="p_up 數值無效，為安全起見不開倉。")

        if risk_reward_ratio is None or risk_reward_ratio <= 0:
            return self._zero_result(direction, p_up, reason="風險回報比無效或非正數，Kelly 無法計算，不開倉。")

        p_win = _directional_win_probability(p_up, direction)
        raw_kelly = _raw_kelly_fraction(p_win, risk_reward_ratio)

        if raw_kelly <= self.kelly_min_edge:
            return self._zero_result(direction, p_up, p_win=p_win, raw_kelly=raw_kelly,
                                      reason=f"Kelly 邊際 ({raw_kelly:.4f}) 低於最小門檻 "
                                             f"({self.kelly_min_edge})，無足夠優勢，不開倉。")

        fractional_kelly = raw_kelly * self.kelly_fraction

        confidence = signal_confidence if signal_confidence is not None else 0.5
        confidence = float(np.clip(confidence, 0.0, 1.0))
        confidence_scaled_kelly = fractional_kelly * confidence

        regime_dampener = _regime_vol_dampener(regime)
        dampened_size = confidence_scaled_kelly * regime_dampener

        capped = False
        final_size = dampened_size
        if final_size > self.max_position_pct:
            final_size = self.max_position_pct
            capped = True
        if 0 < final_size < self.min_position_pct_floor:
            # A genuinely positive but tiny edge is floored up to a minimum
            # tradeable size rather than left as a negligible, cost-eroded
            # position (transaction cost / slippage would dominate a near-zero size).
            final_size = self.min_position_pct_floor
            capped = True

        reason = (f"方向={direction}, p_win={p_win:.3f}, R:R={risk_reward_ratio:.2f}, "
                  f"原始Kelly={raw_kelly:.4f}, 分數Kelly(x{self.kelly_fraction})={fractional_kelly:.4f}, "
                  f"信心加權={confidence_scaled_kelly:.4f}, regime調整(x{regime_dampener})="
                  f"{dampened_size:.4f} -> 最終={final_size:.4f}"
                  + ("（已套用上限/下限）" if capped else ""))

        return {
            "position_size_pct": round(float(final_size), 4),
            "direction": direction,
            "p_win": round(float(p_win), 4),
            "raw_kelly_fraction": round(float(raw_kelly), 4),
            "fractional_kelly": round(float(fractional_kelly), 4),
            "confidence_scaled_kelly": round(float(confidence_scaled_kelly), 4),
            "regime_dampener": regime_dampener,
            "capped": capped,
            "reason": reason,
        }

    def _zero_result(self, direction: str, p_up, p_win: float = None,
                      raw_kelly: float = None, reason: str = "") -> dict:
        """Shared zero-position return path — keeps every early-exit branch
        structurally identical to the success path so callers (predict.py,
        email_report.py) never need to special-case a missing key."""
        return {
            "position_size_pct": 0.0,
            "direction": direction if direction in ("LONG", "SHORT") else "NONE",
            "p_win": round(float(p_win), 4) if p_win is not None else None,
            "raw_kelly_fraction": round(float(raw_kelly), 4) if raw_kelly is not None else None,
            "fractional_kelly": 0.0,
            "confidence_scaled_kelly": 0.0,
            "regime_dampener": None,
            "capped": False,
            "reason": reason,
        }


# ---------------------------------------------------------------------------
# Module-level convenience singleton + function (matches the calling style
# of threshold_calibrator.get_calibrated_signal() — a single importable
# function, no need for callers to instantiate the class themselves)
# ---------------------------------------------------------------------------

_default_sizer = PositionSizer()


def compute_position_size(p_up: float, calibrated_signal: str, signal_confidence: float,
                           risk_reward_ratio: float, regime: str = "NEUTRAL",
                           worth_trading: bool = True) -> dict:
    """Convenience wrapper around the default PositionSizer instance. This
    is the function predict.py / signal_generator.py should import and call
    directly to populate result['position_sizing']."""
    return _default_sizer.compute(p_up=p_up, calibrated_signal=calibrated_signal,
                                   signal_confidence=signal_confidence,
                                   risk_reward_ratio=risk_reward_ratio,
                                   regime=regime, worth_trading=worth_trading)
