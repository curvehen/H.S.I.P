"""HSI constituent universe with sector info, for bottom-up aggregation + email display."""

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


def get_universe() -> dict:
    """Returns {ticker: normalized_weight} for bottom-up aggregation."""
    total = sum(v["weight"] for v in HSI_CONSTITUENTS_INFO.values())
    return {k: v["weight"] / total for k, v in HSI_CONSTITUENTS_INFO.items()}


def get_stock_info(ticker: str) -> dict:
    return HSI_CONSTITUENTS_INFO.get(ticker, {"sector": "未知", "name": ticker})
