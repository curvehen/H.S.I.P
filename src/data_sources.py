"""Multi-source data fetcher with automatic fallback + staleness tagging."""

import time
import pandas as pd
import yfinance as yf
from config import DATA_DIR


def fetch_yfinance(ticker: str, start: str = "2010-01-01") -> pd.DataFrame:
    df = yf.download(ticker, start=start, progress=False)
    if df.empty:
        raise ValueError(f"yfinance empty for {ticker}")
    df["source"] = "yfinance"
    return df


def fetch_stooq(ticker: str) -> pd.DataFrame:
    """Fallback: Stooq. Ticker format e.g. '^hsi', 'tcehy.us'."""
    url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
    df = pd.read_csv(url, index_col="Date", parse_dates=True)
    if df.empty:
        raise ValueError(f"stooq empty for {ticker}")
    df["source"] = "stooq"
    return df


def fetch_with_fallback(ticker: str, stooq_ticker: str = None, retries: int = 2) -> pd.DataFrame:
    """Try yfinance -> stooq -> last known cache (marked stale)."""
    cache_path = DATA_DIR / f"{ticker.replace('^','').replace('.','_')}_last_good.csv"
    sources = [lambda: fetch_yfinance(ticker)]
    if stooq_ticker:
        sources.append(lambda: fetch_stooq(stooq_ticker))

    last_error = None
    for source_fn in sources:
        for _ in range(retries):
            try:
                df = source_fn()
                df["is_stale"] = False
                df.to_csv(cache_path)
                return df
            except Exception as e:
                last_error = e
                time.sleep(1.5)

    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df["source"] = "cache_fallback"
        df["is_stale"] = True
        print(f"WARNING: all sources failed for {ticker} ({last_error}); using stale cache.")
        return df

    raise RuntimeError(f"No data available for {ticker}: {last_error}")
