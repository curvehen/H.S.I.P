from datetime import datetime, time
from zoneinfo import ZoneInfo

HKT = ZoneInfo("Asia/Hong_Kong")

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


def get_market_session(now: datetime = None) -> str:
    """
    Returns 'pre_market', 'intraday', or 'post_market' based on HK time.
    Weekends are treated as 'post_market' (safe default: no partial-day risk).
    Note: public holidays are NOT detected here -- on a holiday this will
    report 'intraday'/'pre_market' incorrectly, but since no new row exists
    for that date in the data, no harm is done (nothing to strip).
    """
    now = now or datetime.now(HKT)
    now = now.astimezone(HKT) if now.tzinfo else now.replace(tzinfo=HKT)

    if now.weekday() >= 5:  # Sat/Sun
        return "post_market"

    t = now.time()
    if t < MARKET_OPEN:
        return "pre_market"
    elif t < MARKET_CLOSE:
        return "intraday"
    else:
        return "post_market"

# market_hours.py (新增)
def get_latest_usable_row(df, now=None):
    """session-aware: 盤前/盤中時，剔除「今日未收市」嘅未完成一行。"""
    session = get_market_session(now)
    now = now or datetime.now(HKT)
    today = now.date()
    last_date = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]

    if session in ("pre_market", "intraday") and last_date == today:
        df = df.iloc[:-1]          # 用返「前收」(上個完整交易日)
    return df, session
