"""
Feature engineering: candlestick patterns, technical indicators, cross-market
factors, CCASS, news sentiment, and macro (Stock Connect / GARCH / ADR).
"""

import talib
import numpy as np
import pandas as pd

from macro_features import build_macro_features

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
    df["MACD"] = macd
    df["MACD_SIGNAL"] = macd_signal
    df["MACD_HIST"] = macd_hist

    df["ATR14"] = talib.ATR(high, low, close, timeperiod=14)

    upper, mid, lower = talib.BBANDS(close, timeperiod=20)
    band_width = (upper - lower).replace(0, np.nan)
    df["BB_PCT"] = (close - lower) / band_width

    df["OBV"] = talib.OBV(close, volume)

    slowk, slowd = talib.STOCH(high, low, close)
    df["STOCH_K"] = slowk
    df["STOCH_D"] = slowd

    for lag in [1, 2, 3, 5]:
        df[f"return_lag{lag}"] = close.pct_change(lag)

    df["daily_range_pct"] = (high - low) / close
    df["close_to_open_pct"] = (close - df["Open"]) / df["Open"]

    return df


def add_cross_market_features(df: pd.DataFrame, us_futures: pd.DataFrame = None,
                               vix: pd.DataFrame = None) -> pd.DataFrame:
    if us_futures is not None and not us_futures.empty:
        df["us_futures_return"] = us_futures["Close"].pct_change().reindex(df.index).ffill()
    else:
        df["us_futures_return"] = 0.0

    if vix is not None and not vix.empty:
        df["vix_level"] = vix["Close"].reindex(df.index).ffill()
        df["vix_change"] = vix["Close"].pct_change().reindex(df.index).ffill()
    else:
        df["vix_level"] = 0.0
        df["vix_change"] = 0.0

    return df


def add_ccass_news_features(df: pd.DataFrame, ccass_change: float = 0.0,
                             market_sentiment: float = 0.0,
                             stock_sentiment: float = 0.0) -> pd.DataFrame:
    df["ccass_top10_change"] = ccass_change
    df["market_sentiment"] = market_sentiment
    df["stock_sentiment"] = stock_sentiment
    return df


def add_macro_features(df: pd.DataFrame, hk_ticker: str = None) -> pd.DataFrame:
    macro = build_macro_features(df["Close"], hk_ticker)
    df["southbound_net_flow_latest"] = macro["southbound_net_flow_latest"]
    df["southbound_net_flow_5d_avg"] = macro["southbound_net_flow_5d_avg"]
    df["southbound_flow_accelerating"] = macro["southbound_flow_accelerating"]
    df["garch_volatility"] = macro["garch_volatility"].reindex(df.index).ffill().bfill().fillna(0.0)
    df["adr_implied_return"] = macro["adr_implied_return"]
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["RSI_x_VIX"] = df["RSI14"] * df["vix_level"]
    df["MACD_x_USFutures"] = df["MACD_HIST"] * df["us_futures_return"]
    return df


def build_features(df: pd.DataFrame, us_futures: pd.DataFrame = None,
                    vix: pd.DataFrame = None, ccass_change: float = 0.0,
                    market_sentiment: float = 0.0, stock_sentiment: float = 0.0,
                    ticker: str = None, include_macro: bool = True) -> pd.DataFrame:
    df = df.copy()
    df = add_candlestick_patterns(df)
    df = add_cross_market_features(df, us_futures, vix)
    df = add_technical_indicators(df)
    df = add_ccass_news_features(df, ccass_change, market_sentiment, stock_sentiment)
    df = add_interaction_features(df)

    if include_macro:
        df = add_macro_features(df, hk_ticker=ticker)

    df = df.dropna()
    return df


def get_numeric_feature_columns(df: pd.DataFrame, exclude: list = None) -> list:
    exclude = set(exclude or []) | {"source", "is_stale"}
    numeric_df = df.select_dtypes(include=[np.number])
    return [c for c in numeric_df.columns if c not in exclude]
