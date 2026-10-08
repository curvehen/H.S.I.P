"""
Next-day return labeling for HSI / stock price prediction.

Computes forward-looking labels for day t using day t+1's OHLC data,
relative to day t's close (classic "next close return" framing used
throughout this pipeline for entry-at-last-close backtesting).

ENH#2 (ported from HSI_Colab.py): also provides an OPEN-BASED variant,
where next_high/next_low are expressed relative to next day's OPEN price
instead of today's close. This matters because the realistic entry point
for a next-day trade is tomorrow's open, not today's close — there is
almost always a gap between the two. Using close-based labels can
overstate or understate the true achievable profit range.

Both variants are provided; existing pipeline code continues to use the
original close-based function and LABEL_COLUMNS unchanged, so nothing
downstream breaks. The open-based variant is opt-in via
build_nextday_labels(..., use_open_base=True) or the standalone
build_nextday_labels_open_based().
"""

import pandas as pd

LABEL_COLUMNS = ["next_close", "next_high", "next_low",
                  "next_close_return", "next_high_return", "next_low_return"]

# Additional columns produced only when use_open_base=True
OPEN_BASED_LABEL_COLUMNS = ["next_open", "next_high_return_open", "next_low_return_open"]


def build_nextday_labels(df: pd.DataFrame, use_open_base: bool = False) -> pd.DataFrame:
    """
    Computes next-day return labels.

    close-based (default, unchanged from original behaviour):
        next_close_return = (next_close - close) / close
        next_high_return  = (next_high  - close) / close
        next_low_return   = (next_low   - close) / close

    open-based (ENH#2, opt-in via use_open_base=True):
        additionally computes next_high_return_open / next_low_return_open,
        expressed relative to next day's OPEN instead of today's close.
        This better reflects the realistic profit range achievable when
        entering at tomorrow's open rather than today's close.

    The last row(s) without a following day will have NaN labels and
    should be dropped by the caller before training (as before).
    """
    df = df.copy()

    df["next_close"] = df["Close"].shift(-1)
    df["next_high"] = df["High"].shift(-1)
    df["next_low"] = df["Low"].shift(-1)

    df["next_close_return"] = (df["next_close"] - df["Close"]) / df["Close"]
    df["next_high_return"] = (df["next_high"] - df["Close"]) / df["Close"]
    df["next_low_return"] = (df["next_low"] - df["Close"]) / df["Close"]

    if use_open_base and "Open" in df.columns:
        df["next_open"] = df["Open"].shift(-1)
        df["next_high_return_open"] = (df["next_high"] - df["next_open"]) / df["next_open"]
        df["next_low_return_open"] = (df["next_low"] - df["next_open"]) / df["next_open"]

    return df.dropna(subset=LABEL_COLUMNS)


def build_nextday_labels_open_based(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper: returns labels with BOTH close-based (for
    backward compatibility with existing close_return-driven code paths
    like predict.py / signal_generator.py) AND open-based high/low
    targets (ENH#2), for training high/low regressors that better match
    a realistic next-open entry.

    Use this when training NEW high/low models intended to be paired
    with an entry-at-next-open execution assumption. Existing models
    trained on the original close-based next_high_return/next_low_return
    remain valid and unaffected — this is purely an additive, opt-in
    label set.
    """
    required_cols = {"Open", "High", "Low", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"build_nextday_labels_open_based() requires columns {required_cols}, "
                          f"missing: {missing}")

    labeled = build_nextday_labels(df, use_open_base=True)
    required_open_cols = LABEL_COLUMNS + OPEN_BASED_LABEL_COLUMNS
    return labeled.dropna(subset=required_open_cols)
