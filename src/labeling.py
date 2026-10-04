"""
Next-day labeling: for each row at day t, labels are derived from day t+1's
actual High/Low/Close relative to day t's Close. This directly matches the
requirement of predicting "tomorrow's entry/high/low/close".
"""

import pandas as pd

LABEL_COLUMNS = ["next_close_return", "next_high_return", "next_low_return"]


def build_nextday_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["next_close_return"] = df["Close"].shift(-1) / df["Close"] - 1
    df["next_high_return"] = df["High"].shift(-1) / df["Close"] - 1
    df["next_low_return"] = df["Low"].shift(-1) / df["Close"] - 1
    return df.dropna(subset=LABEL_COLUMNS)
