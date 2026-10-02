"""
Intraday data fetcher + aggregator.
Free intraday data (yfinance 15m/60m interval) is limited to ~60 days history,
so this module aggregates intraday patterns into DAILY features
(e.g. first-30-min return, VWAP deviation, intraday realized volatility)
rather than training directly on raw intraday bars — this keeps the model's
daily prediction granularity while enriching it with intraday microstructure signals.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from config import INTRADAY_INTERVAL, INTRADAY_LOOKBACK_DAYS, INTRADAY_DIR


def fetch_intraday(ticker: str) -> pd.DataFrame:
    """Fetches recent intraday bars. Returns empty DataFrame on failure (fail-safe)."""
    try:
        df = yf.download(
            ticker,
            period=f"{INTRADAY_LOOKBACK_DAYS}d",
            interval=INTRADAY_INTERVAL,
            progress=False,
        )
        if df.empty:
            raise ValueError("Empty intraday data returned")
        df.index = pd.to_datetime(df.index)
        cache_path = INTRADAY_DIR / f"{ticker.replace('^','').replace('.','_')}_intraday.csv"
        df.to_csv(cache_path)
        return df
    except Exception as e:
        print(f"Intraday fetch failed for {ticker}: {e}")
        cache_path = INTRADAY_DIR / f"{ticker.replace('^','').replace('.','_')}_intraday.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return pd.DataFrame()


def aggregate_intraday_to_daily_features(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts intraday bars into per-day summary features:
    - first_30min_return: momentum right after open
    - last_30min_return: momentum into close
    - intraday_realized_vol: sum of squared intraday returns (realized volatility)
    - vwap_deviation: close vs VWAP (positive = closed above volume-weighted average)
    - intraday_range_pct: (high-low)/open, captures intraday choppiness
    """
    if intraday_df.empty:
        return pd.DataFrame()

    intraday_df["date"] = intraday_df.index.date
    daily_features = []

    for date, group in intraday_df.groupby("date"):
        group = group.sort_index()
        if len(group) < 4:
            continue

        open_price = group["Open"].iloc[0]
        close_price = group["Close"].iloc[-1]
        high_price = group["High"].max()
        low_price = group["Low"].min()

        n_bars_30min = max(1, len(group) // 13)  # approx first/last 30min given bar count
        first_30min_return = (group["Close"].iloc[n_bars_30min - 1] - open_price) / open_price
        last_30min_return = (close_price - group["Close"].iloc[-n_bars_30min]) / group["Close"].iloc[-n_bars_30min]

        intraday_returns = group["Close"].pct_change().dropna()
        intraday_realized_vol = float(np.sqrt((intraday_returns ** 2).sum()))

        vwap = (group["Close"] * group["Volume"]).sum() / group["Volume"].sum() if group["Volume"].sum() > 0 else close_price
        vwap_deviation = (close_price - vwap) / vwap

        intraday_range_pct = (high_price - low_price) / open_price

        daily_features.append({
            "date": pd.Timestamp(date),
            "first_30min_return": first_30min_return,
            "last_30min_return": last_30min_return,
            "intraday_realized_vol": intraday_realized_vol,
            "vwap_deviation": vwap_deviation,
            "intraday_range_pct": intraday_range_pct,
        })

    result = pd.DataFrame(daily_features).set_index("date")
    return result


def get_intraday_features(ticker: str) -> pd.DataFrame:
    """Main entry point: fetch + aggregate, fail-safe to empty df."""
    raw = fetch_intraday(ticker)
    return aggregate_intraday_to_daily_features(raw)
