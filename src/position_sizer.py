# ---------------------------------------------------------------------------
# Cost + position-sizing helpers (shared logic with backtest.py)
#
# FIX LOG (this revision):
#   - PositionSizer(kelly_fraction=0.25, max_position=1.0, min_edge=0.02)
#     used kwarg names that NEVER existed on PositionSizer.__init__()
#     (actual names: kelly_fraction, max_position_pct, min_position_pct_floor,
#     kelly_min_edge) — this alone would raise TypeError on import of this
#     module's first call.
#   - sizer.compute_size(prob=..., signal=..., regime=..., roll_acc=...) was
#     calling a method that does not exist at all. The real method is
#     .compute(p_up, calibrated_signal, signal_confidence, risk_reward_ratio,
#     regime, worth_trading) -> dict with key "position_size_pct" (not a
#     bare float).
#   - p_up is now derived from the confidence column + direction (confidence
#     alone is a meta-label correctness probability, not P(up) — treating it
#     as "prob of up" when signal is SHORT would have been directionally
#     backwards; now: p_up_proxy = confidence if signal>0 else 1-confidence).
#   - calibrated_signal derived from the already-computed `signal` column
#     (+1 -> LONG, -1 -> SHORT, 0 -> 觀望).
#   - risk_reward_ratio: for HSI (pred_high/pred_low columns present),
#     computed direction-aware against last_close as the entry price
#     (no gap-estimated entry price exists in this historical backtest
#     context). For per-stock backtests, pred_high/pred_low are NOT
#     currently produced by daily_backtest_insample_stock() (it only loads
#     the close model) — risk_reward_ratio falls back to a flagged neutral
#     1.0 default for those rows. This is a known simplification, not a
#     real computed ratio; extending daily_backtest_insample_stock() to
#     also load stock_high_model_path()/stock_low_model_path() would remove
#     this gap but is out of scope for this fix.
#   - worth_trading is passed as True unconditionally: this backtest's
#     purpose is to size EVERY day's existing signal via Kelly math (a day
#     with insufficient edge naturally sizes to 0% through compute()'s own
#     raw_kelly <= kelly_min_edge branch), not to re-apply the regime
#     threshold veto a second time on top of the signal already produced.
# ---------------------------------------------------------------------------

from position_sizer import compute_position_size


def apply_cost_simulation(df: pd.DataFrame, confidence_col: str = "confidence",
                           regime_col: str = None, use_position_sizing: bool = True) -> pd.DataFrame:
    """Adds gross/net strategy return columns using Kelly position sizing
    and transaction cost + slippage simulation."""
    df = df.copy()
    df["signal"] = np.sign(df["pred_close_return"])

    has_high_low = "pred_high" in df.columns and "pred_low" in df.columns

    if use_position_sizing:
        sizes = []
        for _, row in df.iterrows():
            confidence = row[confidence_col] if confidence_col in df.columns and pd.notna(row[confidence_col]) else 0.55
            regime = row[regime_col] if regime_col and regime_col in df.columns else "NEUTRAL"
            signal = int(row["signal"])

            if signal > 0:
                calibrated_signal = "LONG"
                p_up_proxy = confidence
            elif signal < 0:
                calibrated_signal = "SHORT"
                p_up_proxy = 1 - confidence
            else:
                calibrated_signal = "觀望"
                p_up_proxy = 0.5

            if has_high_low:
                entry = row["last_close"]
                if signal > 0:
                    reward = row["pred_high"] - entry
                    risk = entry - row["pred_low"]
                else:
                    reward = entry - row["pred_low"]
                    risk = row["pred_high"] - entry
                risk_reward_ratio = (reward / risk) if risk and risk > 0 else 1.0
            else:
                # Per-stock backtest has no high/low predictions available —
                # flagged neutral fallback, not a genuinely computed ratio.
                risk_reward_ratio = 1.0

            result = compute_position_size(
                p_up=p_up_proxy,
                calibrated_signal=calibrated_signal,
                signal_confidence=confidence,
                risk_reward_ratio=risk_reward_ratio,
                regime=regime,
                worth_trading=True,
            )
            sizes.append(result["position_size_pct"])
        df["position_size"] = sizes
    else:
        df["position_size"] = df["signal"].abs()

    df["gross_ret"] = df["signal"] * df["actual_close_return"]

    df["sig_change"] = df["signal"].diff().abs().fillna(abs(df["signal"].iloc[0]) if len(df) else 0)
    slippage_pct = (SLIPPAGE_POINTS / df["last_close"]).fillna(0)
    df["cost"] = df["sig_change"] * (TRANSACTION_COST + slippage_pct) * df["position_size"].clip(lower=0.01)

    df["strategy_ret"] = df["position_size"] * df["signal"] * df["actual_close_return"] - df["cost"]
    df["bh_ret"] = df["actual_close_return"]

    return df
