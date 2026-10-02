"""
Feature engineering: technical indicators + candlestick patterns.
Requires TA-Lib (see installation notes in planning doc).
"""

import pandas as pd
import numpy as np
import talib


CANDLE_PATTERNS = {
    "CDLDOJI": talib.CDLDOJI,
    "CDLENGULFING": talib.CDLENGULFING,
    "CDLHAMMER": talib.CDLHAMMER,
    "CDLMORNINGSTAR": talib.CDLMORNINGSTAR,
    "CDLEVENINGSTAR": talib.CDLEVENINGSTAR,
    "CDL3WHITESOLDIERS": talib.CDL3WHITESOLDIERS,
    "CDLDARKCLOUDCOVER": talib.CDLDARKCLOUDCOVER,
    "CDLHARAMI": talib.CDLHARAMI,
}


def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    for name, func in CANDLE_PATTERNS.items():
        df[name] = func(o, h, l, c)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    df["MA5"] = talib.SMA(close, timeperiod=5)
    df["MA20"] = talib.SMA(close, timeperiod=20)
    df["MA60"] = talib.SMA(close, timeperiod=60)
    df["RSI14"] = talib.RSI(close, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(close)
    df["MACD"] = macd
    df["MACD_SIGNAL"] = macd_signal
    df["MACD_HIST"] = macd_hist
    df["ATR14"] = talib.ATR(high, low, close, timeperiod=14)
    upper, mid, lower = talib.BBANDS(close, timeperiod=20)
    df["BB_PCT"] = (close - lower) / (upper - lower)
    df["OBV"] = talib.OBV(close, volume)
    slowk, slowd = talib.STOCH(high, low, close)
    df["STOCH_K"] = slowk
    df["STOCH_D"] = slowd

    # lag features
    for lag in [1, 2, 3, 5]:
        df[f"return_lag{lag}"] = close.pct_change(lag)

    return df


def add_cross_market_features(df: pd.DataFrame, us_futures: pd.Series = None,
                               vix: pd.Series = None) -> pd.DataFrame:
    """Overnight US futures & VIX as leading indicators for HSI open."""
    if us_futures is not None:
        df["us_futures_return"] = us_futures.pct_change().reindex(df.index).ffill()
    if vix is not None:
        df["vix_level"] = vix.reindex(df.index).ffill()
    return df


def build_features(df: pd.DataFrame, us_futures=None, vix=None) -> pd.DataFrame:
    df = add_candlestick_patterns(df)
    df = add_technical_indicators(df)
    df = add_cross_market_features(df, us_futures, vix)
    df = df.dropna()
    return df

