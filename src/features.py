"""Feature engineering: candlestick patterns + technical indicators + cross-market + CCASS/news."""

import talib
import pandas as pd
import numpy as np

CANDLE_PATTERNS = {
    "CDLDOJI": talib.CDLDOJI,
    "CDLENGULFING": talib.CDLENGULFING,
    "CDLHAMMER": talib.CDLHAMMER,
    "CDLMORNINGSTAR": talib.CDLMORNINGSTAR,
    "CDLEVENINGSTAR": talib.CDLEVENINGSTAR,
    "CDL3WHITESOLDIERS": talib.CDL3WHITESOLDIERS,
    "CDLDARKCLOUDCOVER": talib.CDLDARKCLOUDCOVER,
    "CDLHARAMI": talib.CDLHARAMI,
    "CDLSHOOTINGSTAR": talib.CDLSHOOTINGSTAR,
    "CDLPIERCING": talib.CDLPIERCING,
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
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd, macd_signal, macd_hist
    df["ATR14"] = talib.ATR(high, low, close, timeperiod=14)
    upper, mid, lower = talib.BBANDS(close, timeperiod=20)
    df["BB_PCT"] = (close - lower) / (upper - lower)
    df["OBV"] = talib.OBV(close, volume)
    slowk, slowd = talib.STOCH(high, low, close)
    df["STOCH_K"], df["STOCH_D"] = slowk, slowd

    for lag in [1, 2, 3, 5]:
        df[f"return_lag{lag}"] = close.pct_change(lag)

    return df


def add_cross_market_features(df, us_futures=None, vix=None) -> pd.DataFrame:
    if us_futures is not None:
        df["us_futures_return"] = us_futures["Close"].pct_change().reindex(df.index).ffill()
    if vix is not None:
        df["vix_level"] = vix["Close"].reindex(df.index).ffill()
    return df


def add_ccass_news_features(df: pd.DataFrame, ccass_change: float = 0.0,
                             market_sentiment: float = 0.0,
                             stock_sentiment: float = 0.0) -> pd.DataFrame:
    """Broadcasts scalar daily CCASS/news signals onto the dataframe (same value each row's latest date)."""
    df["ccass_top10_change"] = ccass_change
    df["market_sentiment"] = market_sentiment
    df["stock_sentiment"] = stock_sentiment
    return df


def build_features(df, us_futures=None, vix=None, ccass_change=0.0,
                    market_sentiment=0.0, stock_sentiment=0.0) -> pd.DataFrame:
    df = add_candlestick_patterns(df)
    df = add_technical_indicators(df)
    df = add_cross_market_features(df, us_futures, vix)
    df = add_ccass_news_features(df, ccass_change, market_sentiment, stock_sentiment)
    df = df.dropna()
    return df
