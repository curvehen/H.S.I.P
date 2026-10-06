"""
Multi-source data fetcher with automatic fallback.
Primary: yfinance. Fallback: Stooq. Last resort: cached last-known-good file.
Every returned dataframe is tagged with 'source' and 'is_stale' columns.
"""

import time
import pandas as pd
import yfinance as yf
from config import DATA_DIR


def _clean_ticker_for_filename(ticker: str) -> str:
    return ticker.replace("^", "").replace("=", "_").replace(".", "_")


def fetch_yfinance(ticker: str, start: str = "2010-01-01") -> pd.DataFrame:
    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"yfinance returned empty data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df["source"] = "yfinance"
    return df


def fetch_stooq(stooq_ticker: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={stooq_ticker}&i=d"
    df = pd.read_csv(url, index_col="Date", parse_dates=True)
    if df.empty:
        raise ValueError(f"stooq returned empty data for {stooq_ticker}")
    df["source"] = "stooq"
    return df

def to_stooq_hk_code(ticker: str) -> str:
    """0700.HK -> 0700.hk (Stooq uses 4-digit HK codes, not 5)."""
    return ticker.replace(".HK", "").zfill(4) + ".hk"

def fetch_with_fallback(ticker: str, stooq_ticker: str = None, retries: int = 2) -> pd.DataFrame:
    cache_path = DATA_DIR / f"{_clean_ticker_for_filename(ticker)}_last_good.csv"

    attempts = [lambda: fetch_yfinance(ticker)]
    if stooq_ticker:
        attempts.append(lambda: fetch_stooq(stooq_ticker))

    last_error = None
    for attempt_fn in attempts:
        for _ in range(retries):
            try:
                df = attempt_fn()
                df["is_stale"] = False
                df.to_csv(cache_path)
                return df
            except Exception as e:
                last_error = e
                time.sleep(1.0)

    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df["source"] = "cache_fallback"
        df["is_stale"] = True
        print(f"WARNING: live sources failed for {ticker} ({last_error}). Using cached data.")
        return df

    raise RuntimeError(f"All data sources failed for {ticker} and no cache exists: {last_error}")
