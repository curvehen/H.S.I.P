"""
News sentiment scoring using FinBERT.
Fetches headlines via RSS, scores each with FinBERT, aggregates to a
daily market-wide sentiment index and optional per-stock index (keyword match).
"""

import feedparser
import pandas as pd
from datetime import datetime
from transformers import pipeline
from config import NEWS_RSS_FEEDS, NEWS_DIR

_sentiment_pipeline = None


def _get_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis", model="ProsusAI/finbert"
        )
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
    clf = _get_pipeline()
    scores = clf(list(headlines["title"]))
    headlines["sentiment_label"] = [s["label"] for s in scores]
    headlines["sentiment_score"] = [
        s["score"] if s["label"] == "positive" else -s["score"] if s["label"] == "negative" else 0.0
        for s in scores
    ]
    return headlines


def get_daily_market_sentiment() -> float:
    """
    Returns average sentiment score for today's headlines (-1 to +1).
    Returns 0.0 (neutral) on any failure — sentiment is a supplementary
    signal and must never break the core pipeline.
    """
    try:
        headlines = fetch_headlines()
        scored = score_sentiment(headlines)
        if scored.empty:
            return 0.0
        today_str = datetime.today().strftime("%Y-%m-%d")
        scored.to_csv(NEWS_DIR / f"headlines_{today_str}.csv", index=False)
        return float(scored["sentiment_score"].mean())
    except Exception as e:
        print(f"Sentiment scoring failed: {e}")
        return 0.0


def get_stock_sentiment(stock_name_keywords: list) -> float:
    """Filters cached headlines by keyword match for a specific stock."""
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
