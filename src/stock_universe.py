"""HSI constituent universe with sector info, for bottom-up aggregation +
email display, plus a standalone watchlist for extra stocks to monitor
(e.g. 1211.HK, 0968.HK) that are NOT HSI constituents.

IMPORTANT: get_universe() only ever returns HSI_CONSTITUENTS_INFO weights.
Watchlist tickers are intentionally excluded from get_universe() so that
predict_hsi_bottom_up() in predict.py is never distorted by non-constituent
stocks. Watchlist tickers are only surfaced via get_all_tracked_tickers()
for standalone per-stock prediction + display purposes.
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

# 觀察名單：唔屬於 HSI 成份股，只作獨立監察/交易訊號之用，
# 絕對不會進入 get_universe()，確保 HSI bottom-up 聚合權重不受影響。
WATCHLIST_INFO = {
    "1211.HK": {"sector": "汽車", "name": "比亞迪股份"},
    "0968.HK": {"sector": "新能源", "name": "信義光能"},
}


def get_universe() -> dict:
    """Returns {ticker: normalized_weight} for HSI bottom-up aggregation only.
    Watchlist tickers are never included here."""
    total = sum(v["weight"] for v in HSI_CONSTITUENTS_INFO.values())
    return {k: v["weight"] / total for k, v in HSI_CONSTITUENTS_INFO.items()}


def get_all_tracked_tickers() -> list:
    """Returns every ticker to run a standalone prediction for:
    HSI constituents + watchlist combined. Used by predict_all_stocks()
    for the full per-stock prediction table (email/report display),
    independent from HSI index-level aggregation."""
    return list(HSI_CONSTITUENTS_INFO.keys()) + list(WATCHLIST_INFO.keys())


def is_hsi_constituent(ticker: str) -> bool:
    """True if ticker counts toward HSI bottom-up weight aggregation."""
    return ticker in HSI_CONSTITUENTS_INFO


def get_stock_info(ticker: str) -> dict:
    """Looks up sector/name from either HSI constituents or the watchlist."""
    if ticker in HSI_CONSTITUENTS_INFO:
        return HSI_CONSTITUENTS_INFO[ticker]
    if ticker in WATCHLIST_INFO:
        return WATCHLIST_INFO[ticker]
    return {"sector": "未知", "name": ticker}
