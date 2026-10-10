"""
Email Report — builds and sends the daily HSI prediction HTML email.
Reads the result JSON written by predict.py (LATEST_RESULT_PATH for official
runs, a separate preliminary-only snapshot for preliminary runs — see
__main__ and predict.py's log_prediction()).

FIX LOG (this revision):
  - Every field access below has been realigned to predict.py's ACTUAL
    result dict keys (confirmed by reading predict.py's source directly).
    The previous version referenced an entirely different/older schema
    (target_trading_date, data_as_of_date, data_source, _run_mode, p_down,
    signal_strength_label, signal_confidence, position_size_pct, pred_low/
    pred_high/pred_close, pred_low_pct/pred_high_pct, pred_range_points/
    pred_range_pct, bottom_up_coverage_weight, hit_rate, worth_trading_reason,
    calibrated_signal, _stock_predictions, llm_model, run_timestamp,
    _is_cached_snapshot) — NONE of these keys exist under those names in the
    current predict.py's result dict, and several were accessed via
    [...] indexing, which would raise KeyError and crash the report build.
  - calibrated_signal and worth_trading_reason are NOT computed by
    predict.py at all. Rather than re-deriving this logic independently
    (risking drift from the signal text sent in the same day's report),
    this module now imports and reuses signal_generator.py's
    `_determine_calibrated_signal()` and `_build_worth_trading_reason()`
    directly, so the email and the trading signal text always agree.
  - HSI-level hit_rate is not attached by predict_hsi(); now fetched
    directly via hit_rate_tracker.get_hit_rate(HSI_TICKER), same source
    signal_generator.py uses.
  - pred_low/pred_high/pred_close are now read from predict.py's actual
    *_price keys; percentage figures (pred_low_pct, pred_high_pct,
    pred_range_pct) and point range are derived here from last_close,
    since predict.py does not store them pre-computed.
  - bottom_up_coverage_weight is now computed as the sum of each stock's
    `weight` field across result["stock_predictions"] (constituents that
    successfully produced a prediction), since predict.py does not emit
    this directly.
  - Stock/watchlist sections now read predict.py's ACTUAL two separate
    keys — result["stock_predictions"] (constituents, carries a "weight"
    key) and result["watchlist_predictions"] (no "weight" key) — instead
    of a single combined "_stock_predictions" list filtered by a
    nonexistent "is_constituent" flag.
  - Per-stock table rows now derive absolute price levels and percentage
    return display from predict_single_stock()'s actual return-based
    fields (pred_close_return, pred_high_return, pred_low_return) relative
    to last_close, since predict.py does not emit absolute per-stock
    pred_low/pred_high/pred_close or a "signal" label directly.
  - run_mode key aligned to predict.py's actual "run_mode" (no underscore),
    matching the fix already applied in signal_generator.py.
  - NEW: added the mandatory preliminary-run warning banner
    ("初步方向,非交易信號") per the dual run-time requirement — this was
    completely absent previously. Rendered prominently at the top AND
    bottom of the email body, and reflected in the subject line.
  - llm_model display now falls back to config.DASHSCOPE_MODEL_NAME (the
    actual model used for the call) since predict.py does not attach a
    per-run "llm_model" field.
  - run_timestamp is not produced by predict.py; replaced with a
    "報告產生時間" (report-generation time) computed locally in HKT at
    send time — explicitly labeled as generation time, not prediction
    computation time, to avoid implying a precision predict.py doesn't
    provide.
  - _is_cached_snapshot is not yet implemented anywhere in the pipeline
    (weekend/holiday snapshot reuse is still a pending design item per
    earlier discussion) — the cached-snapshot note is kept but will simply
    never render until that field is actually produced upstream.
"""

import json
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


from config import LATEST_RESULT_PATH, PRED_DIR, HSI_TICKER, DASHSCOPE_MODEL_NAME
from hit_rate_tracker import get_hit_rate
from market_hours import HKT
# Reused rather than re-derived, to keep the email and the signal text in
# agreement on direction/verdict/reason (see FIX LOG).

from signal_generator import _build_worth_trading_reason  # calibrated_signal no longer re-derived here

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
EMAIL_TO = os.environ.get("EMAIL_TO")


def _fmt_pct(x, decimals=2):
    if x is None:
        return "N/A"
    return f"{x:+.{decimals}f}%"


def _fmt_price(x, decimals=1):
    if x is None:
        return "N/A"
    return f"{x:,.{decimals}f}"


def _calibrated_signal_color(signal: str) -> str:
    return {"LONG": "#0a7d2c", "SHORT": "#c0392b", "觀望": "#8a8a8a"}.get(signal, "#333333")


def _safe_hit_rate(ticker: str):
    try:
        return get_hit_rate(ticker)
    except Exception:
        return "N/A"


def build_preliminary_banner() -> str:
    """Mandatory, unambiguous warning block for 7pm HKT preliminary runs.
    Rendered at the very top of the email body."""
    return """
    <div style="background:#fff3cd;border:2px solid #c0392b;border-radius:6px;
                padding:14px 18px;margin-bottom:16px;">
      <p style="margin:0;color:#c0392b;font-weight:bold;font-size:15px;">
        ⚠️ 初步方向,非交易信號 ⚠️
      </p>
      <p style="margin:6px 0 0 0;color:#555;font-size:13px;">
        此為香港時間晚上7時之盤前初步參考,並非正式交易信號。資料未經收盤後完整驗證,
        亦不會記錄於正式歷史訊號記錄。正式訊號請以凌晨4時官方版本為準。
      </p>
    </div>
    """


def build_hsi_section(result: dict) -> str:
    raw_direction = "LONG" if result.get("pred_close_return_blended", 0) > 0 else "SHORT"
    calibrated = _determine_calibrated_signal(result, raw_direction)
    calibrated_color = _calibrated_signal_color(calibrated)
    reason = _build_worth_trading_reason(result)

    verdict_color = "#0a7d2c" if result.get("worth_trading") else "#8a8a8a"
    verdict_text = "值博 ✅" if result.get("worth_trading") else "不值博 ⏸️"

    last_close = result.get("last_close")
    pred_low = result.get("pred_low_price")
    pred_high = result.get("pred_high_price")
    pred_close = result.get("pred_close_price")
    pred_low_pct = (pred_low / last_close - 1) * 100 if last_close and pred_low is not None else None
    pred_high_pct = (pred_high / last_close - 1) * 100 if last_close and pred_high is not None else None
    pred_close_pct = result.get("pred_close_return_blended")
    pred_close_pct = pred_close_pct * 100 if pred_close_pct is not None else None
    pred_range_points = (pred_high - pred_low) if (pred_high is not None and pred_low is not None) else None
    pred_range_pct = result.get("expected_move_pct")
    pred_range_pct = pred_range_pct * 100 if pred_range_pct is not None else None

    stock_predictions = result.get("stock_predictions") or []
    coverage_weight = sum(s.get("weight", 0) for s in stock_predictions)

    hit_rate = _safe_hit_rate(HSI_TICKER)
    position_pct = result.get("position_size_pct")
    position_row = (f'<tr><td>建議倉位 (Fractional Kelly)</td><td>{position_pct:.2%}</td></tr>'
                     if position_pct is not None else "")

    html = f"""
    <h2 style="margin-bottom:4px;">恆生指數 (HSI) 次日預測 — {result.get('date', 'N/A')}</h2>
    <p style="color:#666;font-size:13px;margin-top:0;">
        執行模式: {result.get('run_mode', 'N/A')} |
        Ensemble權重來源: {result.get('ensemble_weight_source', 'N/A')}
    </p>
    <table border="0" cellpadding="6" cellspacing="0" style="border-collapse:collapse;width:100%;max-width:560px;">
      <tr style="background:#f5f5f5;"><td><b>市場 Regime</b></td><td><b>{result.get('regime', 'N/A')}</b></td></tr>
      <tr><td>校準後訊號 (Regime-Calibrated)</td>
          <td style="color:{calibrated_color};font-weight:bold;">{calibrated}</td></tr>
      <tr style="background:#f5f5f5;"><td>P(上升)</td><td>{result.get('p_up')}</td></tr>
      <tr><td>訊號強度</td><td>{result.get('signal_strength', 'N/A')}</td></tr>
      <tr style="background:#f5f5f5;"><td>模型信心分數</td><td>{result.get('confidence')}</td></tr>
      {position_row}
      <tr><td>現價 (Last Close)</td><td>{_fmt_price(last_close)}</td></tr>
      <tr style="background:#f5f5f5;"><td>預測低位</td>
          <td>{_fmt_price(pred_low)} ({_fmt_pct(pred_low_pct)})</td></tr>
      <tr><td>預測高位</td>
          <td>{_fmt_price(pred_high)} ({_fmt_pct(pred_high_pct)})</td></tr>
      <tr style="background:#f5f5f5;"><td>預測收市</td>
          <td>{_fmt_price(pred_close)} ({_fmt_pct(pred_close_pct)})</td></tr>
      <tr><td>預測波幅</td>
          <td>{_fmt_price(pred_range_points)} 點 ({_fmt_pct(pred_range_pct)})</td></tr>
      <tr style="background:#f5f5f5;"><td>Bottom-up 覆蓋權重</td><td>{coverage_weight:.2f}</td></tr>
      <tr><td>歷史命中率 (HSI)</td><td>{hit_rate}</td></tr>
      <tr style="background:#f5f5f5;">
          <td style="color:{verdict_color};font-weight:bold;">是否值博</td>
          <td style="color:{verdict_color};font-weight:bold;">{verdict_text}</td></tr>
      <tr><td>判斷原因</td><td style="font-size:12px;color:#555;">{reason}</td></tr>
    </table>
    """
    return html


def _stock_row_html(stock: dict) -> str:
    """Builds one <tr> for a single stock/watchlist prediction row.
    predict_single_stock() only emits return-based fields
    (pred_close_return, pred_high_return, pred_low_return) — absolute price
    levels are derived here relative to last_close, since predict.py does
    not compute or store absolute per-stock pred_low/pred_high/pred_close."""
    last_close = stock.get("last_close")
    pred_close_return = stock.get("pred_close_return")
    pred_high_return = stock.get("pred_high_return")
    pred_low_return = stock.get("pred_low_return")

    pred_close_price = (last_close * (1 + pred_close_return)
                         if last_close is not None and pred_close_return is not None else None)
    pred_high_price = (last_close * (1 + pred_high_return)
                        if last_close is not None and pred_high_return is not None else None)
    pred_low_price = (last_close * (1 + pred_low_return)
                       if last_close is not None and pred_low_return is not None else None)

    direction = "N/A"
    direction_color = "#333"
    p_up = stock.get("p_up")
    if p_up is not None:
        if p_up >= 0.55:
            direction, direction_color = "看升", "#0a7d2c"
        elif p_up <= 0.45:
            direction, direction_color = "看跌", "#c0392b"
        else:
            direction, direction_color = "中性", "#8a8a8a"

    return f"""
    <tr>
      <td>{stock.get('ticker', 'N/A')}</td>
      <td>{stock.get('name', 'N/A')}</td>
      <td>{stock.get('sector', 'N/A')}</td>
      <td style="color:{direction_color};">{direction}</td>
      <td>{stock.get('p_up')}</td>
      <td>{stock.get('signal_strength', 'N/A')}</td>
      <td>{_fmt_price(last_close)}</td>
      <td>{_fmt_price(pred_close_price)} ({_fmt_pct(pred_close_return * 100 if pred_close_return is not None else None)})</td>
      <td>{_fmt_price(pred_high_price)}</td>
      <td>{_fmt_price(pred_low_price)}</td>
      <td>{stock.get('hit_rate', 'N/A')}</td>
    </tr>
    """


def _stock_table_html(rows: list, title: str, note: str = "") -> str:
    if not rows:
        return f"<h3>{title}</h3><p style='color:#999;'>無可用數據。</p>"
    body_rows = "".join(_stock_row_html(s) for s in rows)
    note_html = f"<p style='color:#666;font-size:12px;'>{note}</p>" if note else ""
    return f"""
    <h3 style="margin-bottom:4px;">{title}</h3>
    {note_html}
    <table border="0" cellpadding="6" cellspacing="0"
           style="border-collapse:collapse;width:100%;font-size:13px;">
      <tr style="background:#eee;font-weight:bold;">
        <td>代號</td><td>名稱</td><td>行業</td><td>方向</td><td>P(升)</td>
        <td>訊號強度</td><td>現價</td><td>預測收市</td><td>預測高位</td>
        <td>預測低位</td><td>命中率</td>
      </tr>
      {body_rows}
    </table>
    """


def build_stock_section(result: dict) -> str:
    """HSI constituents only — predict.py's result["stock_predictions"]
    (each row carries a "weight" key, confirming bottom-up inclusion)."""
    stock_predictions = result.get("stock_predictions") or []
    return _stock_table_html(stock_predictions, "HSI 成份股預測")


def build_watchlist_section(result: dict) -> str:
    """Independent watchlist tickers (1211.HK, 0968.HK) — predict.py's
    result["watchlist_predictions"]. These carry NO "weight" key and are
    never folded into predict_bottom_up()'s weighted return (ENH#4
    isolation requirement)."""
    watchlist_predictions = result.get("watchlist_predictions") or []
    return _stock_table_html(
        watchlist_predictions,
        "獨立觀察名單 (Watchlist — 不計入 HSI Bottom-up 加權)",
        note="此名單股票獨立追蹤,其預測結果不會影響恆生指數 Bottom-up 聚合計算。",
    )


def build_llm_section(result: dict) -> str:
    """Renders the HSI-level market commentary plus any per-stock LLM
    summaries predict.py attached to the result dict. Gracefully omits
    itself (returns empty string) if no commentary is available — e.g. the
    API key wasn't configured, or the call failed/timed out — per the
    fail-safe contract (None is NOT an error state to alarm the reader
    about, it's an expected degraded mode)."""
    commentary = result.get("llm_commentary")
    stock_summaries = result.get("stock_llm_summaries") or {}

    if not commentary and not stock_summaries:
        return ""

    html = '<h3 style="margin-bottom:4px;">AI 市場評論 (DashScope)</h3>'
    if commentary:
        html += f'<p style="background:#f0f4ff;padding:10px 14px;border-radius:6px;">{commentary}</p>'
    else:
        html += '<p style="color:#999;font-size:12px;">本次大盤評論暫不可用。</p>'

    if stock_summaries:
        rows = "".join(
            f"<tr><td>{ticker}</td><td>{summary if summary else '暫不可用'}</td></tr>"
            for ticker, summary in stock_summaries.items()
        )
        html += f"""
        <table border="0" cellpadding="6" cellspacing="0"
               style="border-collapse:collapse;width:100%;font-size:13px;margin-top:8px;">
          <tr style="background:#eee;font-weight:bold;"><td>代號</td><td>個股AI摘要</td></tr>
          {rows}
        </table>
        """
    html += f'<p style="color:#999;font-size:11px;margin-top:4px;">模型: {DASHSCOPE_MODEL_NAME}</p>'
    return html


def build_email_html(result: dict) -> str:
    is_preliminary = result.get("run_mode") != "official"
    report_time = datetime.now(HKT).strftime("%Y-%m-%d %H:%M:%S HKT")

    sections = []
    if is_preliminary:
        sections.append(build_preliminary_banner())
    sections.append(build_hsi_section(result))
    sections.append(build_stock_section(result))
    sections.append(build_watchlist_section(result))
    sections.append(build_llm_section(result))
    if is_preliminary:
        sections.append(build_preliminary_banner())

    body = "".join(sections)
    return f"""
    <html>
    <body style="font-family:Arial,Helvetica,sans-serif;color:#222;">
      {body}
      <p style="color:#999;font-size:11px;margin-top:20px;">報告產生時間: {report_time}</p>
    </body>
    </html>
    """


def build_email_subject(result: dict) -> str:
    date = result.get("date", "N/A")
    if result.get("run_mode") != "official":
        return f"[初步/非交易信號] HSI 盤前預覽 — {date}"
    return f"HSI 每日預測報告 — {date}"


def send_email(result: dict):
    if not (SMTP_USER and SMTP_PASSWORD and EMAIL_TO):
        print("email_report: SMTP credentials or recipient not configured — skipping send.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = build_email_subject(result)
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(build_email_html(result), "html", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, [EMAIL_TO], msg.as_string())
        print("email_report: email sent successfully.")
        return True
    except Exception as e:
        print(f"email_report: failed to send email ({e})")
        return False


def _load_result() -> dict:
    """Reads whichever result file matches the current RUN_MODE — mirrors
    signal_generator.py's run() logic exactly, so both scripts always agree
    on which file is the source of truth for a given invocation."""
    run_mode = os.environ.get("RUN_MODE", "official")
    if run_mode == "official":
        result_path = LATEST_RESULT_PATH
    else:
        result_path = PRED_DIR / "latest_preliminary_result.json"

    if not result_path.exists():
        raise FileNotFoundError(
            f"email_report: expected result file not found: {result_path}. "
            f"Ensure predict.py has run with RUN_MODE='{run_mode}' first."
        )
    with open(result_path) as f:
        return json.load(f)


if __name__ == "__main__":
    result = _load_result()
    send_email(result)
