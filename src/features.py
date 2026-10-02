"""Feature engineering: candlestick + technicals + cross-market + CCASS/news + macro + intraday."""

import talib
import pandas as pd
import numpy as np

from macro_features import build_macro_features
from intraday_data import get_intraday_features

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

    # ---- Interaction features ----
    df["RSI_x_VIX"] = df["RSI14"] * df.get("vix_level", 0)
    df["MACD_x_USFutures"] = df["MACD_HIST"] * df.get("us_futures_return", 0)

    return df


def add_cross_market_features(df, us_futures=None, vix=None) -> pd.DataFrame:
    if us_futures is not None:
        df["us_futures_return"] = us_futures["Close"].pct_change().reindex(df.index).ffill()
    if vix is not None:
        df["vix_level"] = vix["Close"].reindex(df.index).ffill()
    return df


def add_ccass_news_features(df, ccass_change=0.0, market_sentiment=0.0, stock_sentiment=0.0):
    df["ccass_top10_change"] = ccass_change
    df["market_sentiment"] = market_sentiment
    df["stock_sentiment"] = stock_sentiment
    return df


def add_macro_features(df: pd.DataFrame, hk_ticker: str = None) -> pd.DataFrame:
    """Adds Stock Connect flow + GARCH volatility + ADR-implied proxy."""
    macro = build_macro_features(df["Close"], hk_ticker)

    df["southbound_net_flow_latest"] = macro["southbound_net_flow_latest"]
    df["southbound_net_flow_5d_avg"] = macro["southbound_net_flow_5d_avg"]
    df["southbound_flow_accelerating"] = macro["southbound_flow_accelerating"]
    df["garch_volatility"] = macro["garch_volatility"].reindex(df.index).ffill().bfill()
    df["adr_implied_return"] = macro["adr_implied_return"]

    return df


def add_intraday_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merges aggregated intraday microstructure features (previous day's session)
    onto the daily dataframe. Uses previous day's intraday pattern as a feature
    for predicting the NEXT day (avoids look-ahead: today's intraday data is
    obviously not available before today's prediction is made).
    """
    intraday_feats = get_intraday_features(ticker)
    if intraday_feats.empty:
        for col in ["first_30min_return", "last_30min_return",
                    "intraday_realized_vol", "vwap_deviation", "intraday_range_pct"]:
            df[col] = 0.0
        return df

    intraday_feats.index = pd.to_datetime(intraday_feats.index)
    df = df.join(intraday_feats, how="left")
    intraday_cols = ["first_30min_return", "last_30min_return",
                      "intraday_realized_vol", "vwap_deviation", "intraday_range_pct"]
    df[intraday_cols] = df[intraday_cols].ffill().fillna(0.0)
    return df


def build_features(df, us_futures=None, vix=None, ccass_change=0.0,
                    market_sentiment=0.0, stock_sentiment=0.0,
                    ticker=None, include_macro=True, include_intraday=True) -> pd.DataFrame:
    df = add_candlestick_patterns(df)
    df = add_cross_market_features(df, us_futures, vix)
    df = add_technical_indicators(df)
    df = add_ccass_news_features(df, ccass_change, market_sentiment, stock_sentiment)

    if include_macro:
        df = add_macro_features(df, hk_ticker=ticker)

    if include_intraday and ticker is not None:
        df = add_intraday_features(df, ticker)

    df = df.dropna()
    return df

