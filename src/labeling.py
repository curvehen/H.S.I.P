"""Triple-barrier labeling (López de Prado style)."""

import numpy as np
import pandas as pd
from config import BARRIER_HOLDING_DAYS, PROFIT_TAKE_ATR_MULT, STOP_LOSS_ATR_MULT


def triple_barrier_labels(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].values
    atr = df["ATR14"].values
    n = len(df)

    labels = np.full(n, np.nan)
    barrier_return = np.full(n, np.nan)

    for i in range(n - BARRIER_HOLDING_DAYS):
        entry_price = close[i]
        upper = entry_price + PROFIT_TAKE_ATR_MULT * atr[i]
        lower = entry_price - STOP_LOSS_ATR_MULT * atr[i]

        window = close[i + 1: i + 1 + BARRIER_HOLDING_DAYS]
        hit_upper = np.where(window >= upper)[0]
        hit_lower = np.where(window <= lower)[0]

        if len(hit_upper) > 0 and (len(hit_lower) == 0 or hit_upper[0] <= hit_lower[0]):
            labels[i] = 1
            barrier_return[i] = (upper - entry_price) / entry_price
        elif len(hit_lower) > 0:
            labels[i] = -1
            barrier_return[i] = (lower - entry_price) / entry_price
        else:
            labels[i] = 0
            barrier_return[i] = (window[-1] - entry_price) / entry_price

    df["barrier_label"] = labels
    df["barrier_return"] = barrier_return
    return df.dropna(subset=["barrier_label"])
