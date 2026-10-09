"""
HSI constituent universe with sector info, for bottom-up aggregation + email display.

NEW: also defines a separate, standalone WATCHLIST — additional tickers to
monitor and predict (e.g. 1211.HK, 0968.HK) that are NOT HSI constituents.
Watchlist stocks must NEVER be mixed into get_universe()'s weight dict: that
dict feeds the HSI bottom-up index aggregation, and an un-weighted or
arbitrarily-weighted watchlist ticker injected there would silently distort
the aggregated index-level prediction. Watchlist tickers are tracked and
reported entirely separately (own prediction rows, own email section).
"""

HSI_CONSTITUENTS_INFO = {
    "0700.HK": {"weight": 0.081, "sector": "科技",   "name": "騰訊控股"},
    "9988.HK": {"weight": 0.062, "sector": "科技",   "name": "阿里巴巴"},
    "0941.HK": {"weight": 0.058, "sector": "電訊",   "name": "中國移動"},
    "1299.HK": {"weight": 0.055, "sector": "保險",   "name": "友邦保險"},
    "0388.HK": {"weight": 0.052, "sector": "金融",   "name": "香港交易所"},
    "3690.HK": {"weight": 0.045, "sector": "科技",   "name": "美團"},
    "0005.HK": {"weight": 0.044, "sector": "金融",   "name": "滙豐控股"},
    "1810.HK": {"weight": 0.030, "sector": "科技",   "name": "小米集團"},
    "2318.HK": {"weight": 0.028, "sector": "保險",   "name": "中國平安"},
    "0016.HK": {"weight": 0.025, "sector": "地產",   "name": "新鴻基地產"},
    "0027.HK": {"weight": 0.020, "sector": "博彩",   "name": "銀河娛樂"},
    "0883.HK": {"weight": 0.020, "sector": "能源",   "name": "中國海洋石油"},
    "1398.HK": {"weight": 0.019, "sector": "金融",   "name": "工商銀行"},
    "3988.HK": {"weight": 0.018, "sector": "金融",   "name": "中國銀行"},
    "0002.HK": {"weight": 0.017, "sector": "公用",   "name": "中電控股"},
    "0001.HK": {"weight": 0.016, "sector": "綜合企業", "name": "長和"},
}

# ---------------------------------------------------------------------------
# NEW: standalone watchlist — tracked/predicted independently of the HSI
# bottom-up aggregation. No "weight" field is needed since these tickers
# never participate in any index-level weighted sum; sector/name are kept
# for consistent email/dashboard display alongside the HSI constituent table.
# ---------------------------------------------------------------------------
WATCHLIST_INFO = {
    "1211.HK": {"sector": "汽車", "name": "比亞迪股份"},
    "0968.HK": {"sector": "新能源", "name": "信義光能"},
}


def get_universe() -> dict:
    """Returns {ticker: normalized_weight} for HSI bottom-up aggregation.
    Watchlist tickers are intentionally excluded — see module docstring."""
    total = sum(v["weight"] for v in HSI_CONSTITUENTS_INFO.values())
    return {k: v["weight"] / total for k, v in HSI_CONSTITUENTS_INFO.items()}


def get_stock_info(ticker: str) -> dict:
    """Looks up sector/name for an HSI constituent. Falls back to watchlist
    info if the ticker isn't a constituent, so callers that don't
    distinguish the two lists (e.g. a generic 'stock info' lookup in
    email_report.py) still get a sensible result."""
    if ticker in HSI_CONSTITUENTS_INFO:
        return HSI_CONSTITUENTS_INFO[ticker]
    if ticker in WATCHLIST_INFO:
        return WATCHLIST_INFO[ticker]
    return {"sector": "未知", "name": ticker}


def get_watchlist() -> list:
    """Returns the list of standalone watchlist tickers to predict and
    report on, separately from the HSI constituent universe. predict.py
    should iterate this list with the SAME per-stock prediction function
    used for HSI constituents, but must write results to a separate
    section/log so they are never folded into get_universe()'s weighted sum."""
    return list(WATCHLIST_INFO.keys())


def get_watchlist_info(ticker: str) -> dict:
    """Looks up sector/name for a watchlist ticker specifically. Returns a
    safe fallback dict (never raises) if the ticker is not on the watchlist."""
    return WATCHLIST_INFO.get(ticker, {"sector": "未知", "name": ticker})


def is_watchlist_ticker(ticker: str) -> bool:
    """Convenience check used by predict.py/email_report.py to route a
    ticker's results to the watchlist section instead of the HSI
    constituent table."""
    return ticker in WATCHLIST_INFO
