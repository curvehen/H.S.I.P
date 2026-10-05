"""
Market regime detection: BULL / BEAR / NEUTRAL, based on price position
relative to moving averages. Uses MA20/MA60 already computed in features.py.
"""

import pandas as pd


def detect_regime(feat_df: pd.DataFrame) -> str:
    """Determines regime from the latest row of a feature dataframe."""
    latest = feat_df.iloc[-1]
    close = latest["Close"]
    ma20 = latest.get("MA20")
    ma60 = latest.get("MA60")

    if ma20 is None or ma60 is None or pd.isna(ma20) or pd.isna(ma60):
        return "NEUTRAL"

    if close > ma60 and ma20 > ma60:
        return "BULL"
    elif close < ma60 and ma20 < ma60:
        return "BEAR"
    else:
        return "NEUTRAL"
