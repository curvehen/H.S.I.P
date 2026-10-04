"""
News sentiment scoring using FinBERT.
Fetches headlines via RSS, scores each, aggregates to daily market-wide
and per-stock sentiment indices. Fails safe to neutral (0.0) on any error.
"""

import feedparser
import pandas as pd
from datetime import datetime
from config import NEWS_RSS_FEEDS, NEWS_DIR

_sentiment_pipeline = None


def _get_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        _sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return _sentiment_pipeline


def fetch_headlines() -> pd.DataFrame:
    records = []
    for feed_url in NEWS_RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                records.append({
                    "title": entry.get("title", ""),
                    "published": entry.get("published", ""),
                    "source": feed_url,
                })
        except Exception as e:
            print(f"RSS fetch failed for {feed_url}: {e}")
    return pd.DataFrame(records)


def score_sentiment(headlines: pd.DataFrame) -> pd.DataFrame:
    if headlines.empty:
        return headlines
    try:
        clf = _get_pipeline()
        scores = clf(list(headlines["title"]), truncation=True)
        headlines["sentiment_label"] = [s["label"] for s in scores]
        headlines["sentiment_score"] = [
            s["score"] if s["label"] == "positive" else -s["score"] if s["label"] == "negative" else 0.0
            for s in scores
        ]
    except Exception as e:
        print(f"Sentiment scoring failed: {e}")
        headlines["sentiment_label"] = "neutral"
        headlines["sentiment_score"] = 0.0
    return headlines


def get_daily_market_sentiment() -> float:
    try:
        headlines = fetch_headlines()
        scored = score_sentiment(headlines)
        if scored.empty:
            return 0.0
        today_str = datetime.today().strftime("%Y-%m-%d")
        scored.to_csv(NEWS_DIR / f"headlines_{today_str}.csv", index=False)
        return float(scored["sentiment_score"].mean())
    except Exception as e:
        print(f"get_daily_market_sentiment failed: {e}")
        return 0.0


def get_stock_sentiment(stock_name_keywords: list) -> float:
    try:
        today_str = datetime.today().strftime("%Y-%m-%d")
        path = NEWS_DIR / f"headlines_{today_str}.csv"
        if not path.exists():
            return 0.0
        df = pd.read_csv(path)
        mask = df["title"].str.contains("|".join(stock_name_keywords), case=False, na=False)
        matched = df[mask]
        if matched.empty:
            return 0.0
        return float(matched["sentiment_score"].mean())
    except Exception:
        return 0.0
