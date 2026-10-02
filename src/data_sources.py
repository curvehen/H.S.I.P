"""
Multi-source data fetcher with automatic fallback.
Ensures pipeline doesn't break when one data source is stale/down.
Each returned row is tagged with its source and staleness flag.
"""

import pandas as pd
import yfinance as yf
import time
import sys
sys.path.append("..")
from config import HSI_TICKER, DATA_DIR


def fetch_yfinance(ticker: str, start: str = "2010-01-01") -> pd.DataFrame:
    """Primary source: Yahoo Finance via yfinance."""
    df = yf.download(ticker, start=start, progress=False)
    if df.empty:
        raise ValueError("yfinance returned empty dataframe")
    df["source"] = "yfinance"
    return df


def fetch_stooq(ticker: str = "^hsi") -> pd.DataFrame:
    """Fallback source: Stooq CSV endpoint (free, no key needed)."""
    url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
    df = pd.read_csv(url, index_col="Date", parse_dates=True)
    if df.empty:
        raise ValueError("stooq returned empty dataframe")
    df["source"] = "stooq"
    return df


def fetch_with_fallback(ticker: str = HSI_TICKER, stooq_ticker: str = "^hsi",
                         retries: int = 2) -> pd.DataFrame:
    """
    Try primary source first, fall back to secondary on failure.
    Returns dataframe tagged with source name + is_stale flag.
    """
    sources = [
        lambda: fetch_yfinance(ticker),
        lambda: fetch_stooq(stooq_ticker),
    ]

    last_error = None
    for source_fn in sources:
        for attempt in range(retries):
            try:
                df = source_fn()
                df["is_stale"] = False
                return df
            except Exception as e:
                last_error = e
                time.sleep(2)
                continue

    # All sources failed -> load last known good cache, mark as stale
    cache_path = DATA_DIR / "hsi_last_good.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df["source"] = "cache_fallback"
        df["is_stale"] = True
        print(f"WARNING: all live sources failed ({last_error}). Using stale cache.")
        return df

    raise RuntimeError(f"All data sources failed and no cache available: {last_error}")


def save_as_last_good(df: pd.DataFrame):
    """Persist the most recent successful fetch as fallback cache."""
    df.to_csv(DATA_DIR / "hsi_last_good.csv")


if __name__ == "__main__":
    data = fetch_with_fallback()
    save_as_last_good(data)
    data.to_csv(DATA_DIR / "hsi_raw.csv")
    print(f"Fetched {len(data)} rows from source={data['source'].iloc[-1]}")

