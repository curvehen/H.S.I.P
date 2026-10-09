"""
Email Report Generator — builds and sends the daily HTML market report.

Reads the single frozen result for this run from LATEST_RESULT_PATH (written
by predict.py), rather than independently calling predict_today(). This
guarantees the email always matches exactly what predict.py and
signal_generator.py already computed/logged for this run — no risk of
re-fetching live data a second time and getting a slightly different number.
"""

import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import LATEST_RESULT_PATH

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


def build_hsi_section(result: dict) -> str:
    verdict_color = "#0a7d2c" if result["worth_trading"] else "#8a8a8a"
    verdict_text = "值博 ✅" if result["worth_trading"] else "不值博 ⏸️"
    stale_badge = ('<span style="color:#c0392b;font-weight:bold;"> ⚠️ 數據可能非即時</span>'
                   if result.get("is_stale") else "")
    calibrated = result.get("calibrated_signal", "N/A")
    calibrated_color = _calibrated_signal_color(calibrated)
    position_pct = result.get("position_size_pct")
    position_row = (f'<tr><td>建議倉位 (Fractional Kelly)</td><td>{position_pct}%</td></tr>'
                     if position_pct is not None else "")

    html = f"""
    <h2 style="margin-bottom:4px;">恒生指數 (HSI) 次日預測 — {result.get('target_trading_date', 'N/A')}</h2>
    <p style="color:#666;font-size:13px;margin-top:0;">
        數據截止日: {result.get('data_as_of_date', 'N/A')} |
        數據來源: {result.get('data_source', 'N/A')}{stale_badge} |
        執行模式: {result.get('_run_mode', 'N/A')}
    </p>
    <table border="0" cellpadding="6" cellspacing="0" style="border-collapse:collapse;width:100%;max-width:560px;">
      <tr style="background:#f5f5f5;"><td><b>市場 Regime</b></td><td><b>{result.get('regime', 'N/A')}</b></td></tr>
      <tr><td>校準後訊號 (Regime-Calibrated)</td>
          <td style="color:{calibrated_color};font-weight:bold;">{calibrated}</td></tr>
      <tr style="background:#f5f5f5;"><td>P(上升) / P(下跌)</td>
          <td>{result.get('p_up')} / {result.get('p_down')}</td></tr>
      <tr><td>訊號強度</td><td>{result.get('signal_strength_label', 'N/A')}</td></tr>
      <tr style="background:#f5f5f5;"><td>模型信心分數</td><td>{result.get('signal_confidence')}</td></tr>
      {position_row}
      <tr><td>現價 (Last Close)</td><td>{_fmt_price(result.get('last_close'))}</td></tr>
      <tr style="background:#f5f5f5;"><td>預測低位</td>
          <td>{_fmt_price(result.get('pred_low'))} ({_fmt_pct(result.get('pred_low_pct'))})</td></tr>
      <tr><td>預測高位</td>
          <td>{_fmt_price(result.get('pred_high'))} ({_fmt_pct(result.get('pred_high_pct'))})</td></tr>
      <tr style="background:#f5f5f5;"><td>預測收市</td>
          <td>{_fmt_price(result.get('pred_close'))} ({_fmt_pct(result.get('pred_close_return_blended', 0) * 100)})</td></tr>
      <tr><td>預測波幅</td>
          <td>{_fmt_price(result.get('pred_range_points'))} 點 ({_fmt_pct(result.get('pred_range_pct'))})</td></tr>
      <tr style="background:#f5f5f5;"><td>Bottom-up 覆蓋權重</td><td>{result.get('bottom_up_coverage_weight', 0):.2f}</td></tr>
      <tr><td>歷史命中率 (HSI)</td><td>{result.get('hit_rate', 'N/A')}</td></tr>
      <tr style="background:#f5f5f5;"><td><b>交易建議</b></td>
          <td style="color:{verdict_color};font-weight:bold;">{verdict_text}</td></tr>
      <tr><td colspan="2" style="font-size:12px;color:#666;">{result.get('worth_trading_reason', '')}</td></tr>
    </table>
    """
    return html


def build_llm_commentary_section(result: dict) -> str:
    commentary = result.get("llm_commentary")
    if not commentary:
        return ""
    model_name = result.get("llm_model", "LLM")
    return f"""
    <h3 style="margin-bottom:4px;">AI 市場分析 ({model_name})</h3>
    <div style="background:#f9f9f9;border-left:4px solid #0a7d2c;padding:10px 14px;
                font-size:13px;line-height:1.6;color:#333;white-space:pre-wrap;">
        {commentary}
    </div>
    """


def _stock_table_rows(stocks: list) -> str:
    if not stocks:
        return '<tr><td colspan="8" style="text-align:center;color:#999;">無可用預測</td></tr>'
    rows = []
    for s in stocks:
        signal_color = "#0a7d2c" if s["signal"] == "LONG" else "#c0392b"
        rr = None
        if s.get("pred_low") and s.get("last_close") and s["pred_close"] != s["last_close"]:
            risk = abs(s["last_close"] - s["pred_low"])
            reward = abs(s["pred_close"] - s["last_close"])
            rr = round(reward / risk, 2) if risk > 0 else None
        rows.append(f"""
        <tr>
          <td>{s['ticker']}</td>
          <td>{s['name']}</td>
          <td>{s['sector']}</td>
          <td style="color:{signal_color};font-weight:bold;">{s['signal']}</td>
          <td>{s['p_up']}</td>
          <td>{_fmt_price(s['last_close'])}</td>
          <td>{_fmt_price(s['pred_low'])} / {_fmt_price(s['pred_high'])}</td>
          <td>{_fmt_price(s['pred_close'])} ({_fmt_pct(s['pred_return_pct'])})</td>
          <td>{rr if rr is not None else 'N/A'}</td>
          <td>{s.get('hit_rate', 'N/A')}</td>
        </tr>
        """)
    return "".join(rows)


def build_stock_section(result: dict) -> str:
    all_stocks = result.get("_stock_predictions") or []
    constituents = [s for s in all_stocks if s.get("is_constituent")]
    watchlist = [s for s in all_stocks if not s.get("is_constituent")]

    header = """
    <tr style="background:#333;color:#fff;">
      <th>代號</th><th>名稱</th><th>行業</th><th>方向</th><th>P(升)</th>
      <th>現價</th><th>預測低/高</th><th>預測收市</th><th>R:R</th><th>命中率</th>
    </tr>
    """

    html = f"""
    <h3 style="margin-bottom:4px;">HSI 成份股預測 ({len(constituents)} 支)</h3>
    <table border="0" cellpadding="5" cellspacing="0" style="border-collapse:collapse;width:100%;font-size:13px;">
      {header}
      {_stock_table_rows(constituents)}
    </table>
    """

    if watchlist:
        html += f"""
        <h3 style="margin-bottom:4px;margin-top:20px;">觀察名單 ({len(watchlist)} 支，不計入 HSI 聚合)</h3>
        <table border="0" cellpadding="5" cellspacing="0" style="border-collapse:collapse;width:100%;font-size:13px;">
          {header}
          {_stock_table_rows(watchlist)}
        </table>
        """

    return html


def build_email_html(result: dict) -> str:
    cached_note = ""
    if result.get("_is_cached_snapshot"):
        cached_note = ('<p style="font-size:12px;color:#999;">'
                        '本次結果讀取自今日已凍結之 snapshot，與早前 official run 完全一致。</p>')

    return f"""
    <html>
    <body style="font-family:Arial,Helvetica,sans-serif;color:#222;max-width:800px;margin:0 auto;">
      {build_hsi_section(result)}
      {cached_note}
      {build_llm_commentary_section(result)}
      {build_stock_section(result)}
      <p style="font-size:11px;color:#999;margin-top:24px;">
        本報告由自動化模型生成，僅供參考，不構成投資建議。
        執行時間: {result.get('run_timestamp', 'N/A')}
      </p>
    </body>
    </html>
    """


def build_email_subject(result: dict) -> str:
    calibrated = result.get("calibrated_signal", "N/A")
    date = result.get("target_trading_date", "N/A")
    verdict = "值博" if result.get("worth_trading") else "觀望"
    return f"[HSI 每日預測] {date} | {calibrated} | {verdict}"


def send_email(result: dict):
    if not all([SMTP_USER, SMTP_PASSWORD, EMAIL_TO]):
        print("SMTP credentials or recipient missing — skipping email send.")
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
            server.sendmail(SMTP_USER, EMAIL_TO.split(","), msg.as_string())
        print("Email sent successfully.")
        return True
    except Exception as e:
        print(f"Email send failed: {e}")
        return False


if __name__ == "__main__":
    with open(LATEST_RESULT_PATH) as f:
        result = json.load(f)
    send_email(result)
