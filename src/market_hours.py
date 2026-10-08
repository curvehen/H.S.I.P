from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import pandas_market_calendars as mcal

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


def get_latest_usable_row(df, now=None):
    """session-aware: 盤前/盤中時，剔除「今日未收市」嘅未完成一行。"""
    session = get_market_session(now)
    now = now or datetime.now(HKT)
    today = now.date()
    last_date = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]

    if session in ("pre_market", "intraday") and last_date == today:
        df = df.iloc[:-1]          # 用返「前收」(上個完整交易日)
    return df, session


_hkex_calendar = mcal.get_calendar("HKEX")


def get_next_trading_day(from_date):
    """
    動態計算下一個 HKEX 交易日，自動跳過週末及公眾假期。
    毋須手動維護年度假期清單，pandas_market_calendars 內建香港交易所日曆。
    """
    if isinstance(from_date, str):
        from_date = datetime.strptime(from_date, "%Y-%m-%d").date()

    schedule = _hkex_calendar.schedule(
        start_date=from_date + timedelta(days=1),
        end_date=from_date + timedelta(days=14)
    )
    return schedule.index[0].date()

def get_prediction_context(now: datetime = None) -> dict:
    """
    核心函數：根據當前時間決定預測情境。
    Ported from HSI_Colab.py get_prediction_context(), 整合你現有嘅
    get_market_session() + get_next_trading_day()。

    四種情境：
    1. 今日收市後 → 預測明日（下一交易日）
    2. 明日開市前 → 預測今日（只用昨收數據）
    3. 今日開市中 → 預測今日（用當前日內數據，附可用度分級）
    4. 週末/假期後 → 預測下一交易日
    """
    now = now or datetime.now(HKT)
    today = now.date()
    session = get_market_session(now)

    result = {
        "now_hkt": now,
        "today_str": today.strftime("%Y-%m-%d"),
    }

    if session == "post_market" and now.weekday() < 5:
        # 情境1：平日收市後
        target = get_next_trading_day(today)
        result.update({
            "scenario": 1,
            "scenario_desc": "平日收市後 → 預測明日",
            "predict_date_str": str(target),
            "base_date_str": str(today),
            "use_intraday": False,
            "data_note": f"使用今日（{today}）完整收市數據，預測 {target}",
        })
    elif session == "post_market" and now.weekday() >= 5:
        # 情境4：週末
        target = get_next_trading_day(today)
        result.update({
            "scenario": 4,
            "scenario_desc": "週末 → 預測下一交易日",
            "predict_date_str": str(target),
            "base_date_str": str(today),
            "use_intraday": False,
            "data_note": f"使用最後已知收市數據，預測 {target}",
        })
    elif session == "pre_market":
        # 情境2：開市前
        result.update({
            "scenario": 2,
            "scenario_desc": "開市前 → 預測今日（昨收數據）",
            "predict_date_str": str(today),
            "base_date_str": str(today),
            "use_intraday": False,
            "data_note": f"使用昨日收市數據，預測今日 {today} 收市方向",
        })
    elif session == "intraday":
        # 情境3：開市中
        t = now.time()
        if t.hour < 10:
            intraday_pct, qual = 0.1, "早盤（數據有限）"
        elif t.hour < 12:
            intraday_pct, qual = 0.4, "上午盤（數據中等）"
        elif t.hour < 14:
            intraday_pct, qual = 0.6, "午後（數據較好）"
        else:
            intraday_pct, qual = 0.85, "尾市（數據充足）"

        result.update({
            "scenario": 3,
            "scenario_desc": f"開市中 → 預測今日（{qual}）",
            "predict_date_str": str(today),
            "base_date_str": str(today),
            "use_intraday": True,
            "intraday_avail": qual,
            "intraday_pct": intraday_pct,
            "data_note": f"使用日內實時數據（{qual}），預測今日收市方向",
        })

    return result
