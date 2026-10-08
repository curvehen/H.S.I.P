"""
Standalone HTML dashboard generator — separate from email_report.py.
Produces a visual dashboard (predictions/dashboard.html) with HSI charts
(price trend + predicted range, P(up)/P(down) gauge, regime badge) and
the per-stock prediction table. Also sends a SEPARATE email containing
this dashboard as an attachment, independent from the existing daily
text-style email in email_report.py (which remains untouched).
"""

import os
import io
import base64
import smtplib
import datetime
import matplotlib
matplotlib.use("Agg")  # headless rendering for CI environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from config import PRED_DIR, HSI_TICKER
from data_sources import fetch_with_fallback
from predict import predict_today

DASHBOARD_PATH = PRED_DIR / "dashboard.html"
CHART_HISTORY_DAYS = 30


# ---------------------------------------------------------------------------
# Chart generation (returns base64-encoded PNG strings for embedding in HTML)
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_price_trend_chart(result: dict) -> str:
    """近30日 HSI 收市價走勢 + 今日預測高/低/收區間（陰影）。"""
    raw = fetch_with_fallback(HSI_TICKER, stooq_ticker="^hsi")
    recent = raw.tail(CHART_HISTORY_DAYS).copy()

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(recent.index, recent["Close"], color="#1a73e8", linewidth=1.8, label="歷史收市價")

    last_date = recent.index[-1]
    next_date = last_date + pd.Timedelta(days=1)

    pred_high = result["pred_high"]
    pred_low = result["pred_low"]
    pred_close = result["pred_close"]
    last_close = result["last_close"]

    ax.plot([last_date, next_date], [last_close, pred_close],
            color="#ff6d00", linestyle="--", linewidth=1.8, marker="o", markersize=5,
            label="預測收市")
    ax.fill_between([last_date, next_date], [last_close, pred_low], [last_close, pred_high],
                     color="#ff6d00", alpha=0.15, label="預測高低區間")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_title(f"HSI 近{CHART_HISTORY_DAYS}日走勢 + {result['target_trading_date']} 預測區間", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate(rotation=30)

    return _fig_to_base64(fig)


def generate_probability_gauge(result: dict) -> str:
    """P(升)/P(跌) 橫向長條對比圖。"""
    p_up = result["p_up"] * 100
    p_down = result["p_down"] * 100

    fig, ax = plt.subplots(figsize=(6, 1.6))
    ax.barh(["機率"], [p_up], color="#2e7d32", label=f"P(升) {p_up:.1f}%")
    ax.barh(["機率"], [p_down], left=[p_up], color="#c62828", label=f"P(跌) {p_down:.1f}%")

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_title("P(升) vs P(跌)", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize=9, frameon=False)

    return _fig_to_base64(fig)


def generate_signal_strength_chart(result: dict) -> str:
    """校準訊號 + 建議倉位嘅視覺化儀表。"""
    position_pct = result.get("position_size_pct", 0)
    signal = result.get("calibrated_signal", "觀望")
    color = {"LONG": "#2e7d32", "SHORT": "#c62828", "觀望": "#757575"}.get(signal, "#757575")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie([position_pct, max(100 - position_pct, 0)],
           colors=[color, "#e0e0e0"], startangle=90, counterclock=False,
           wedgeprops={"width": 0.35})
    ax.text(0, 0.1, signal, ha="center", va="center", fontsize=16, fontweight="bold", color=color)
    ax.text(0, -0.15, f"倉位 {position_pct:.1f}%", ha="center", va="center", fontsize=11, color="#555")
    ax.set_title("校準訊號 & 建議倉位", fontsize=11)

    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML dashboard assembly
# ---------------------------------------------------------------------------

def build_dashboard_html(result: dict) -> str:
    price_chart_b64 = generate_price_trend_chart(result)
    prob_chart_b64 = generate_probability_gauge(result)
    signal_chart_b64 = generate_signal_strength_chart(result)

    regime_color = {"BULL": "#2e7d32", "BEAR": "#c62828", "NEUTRAL": "#757575"}.get(result["regime"], "#757575")

    stock_rows = ""
    for s in result["_stock_predictions"]:
        signal_color = "#2e7d32" if s["signal"] == "LONG" else "#c62828"
        rsi_display = f"{s['rsi']:.1f}" if s['rsi'] is not None else 'N/A'
        stock_rows += f"""
        <tr>
            <td>{s['name']} ({s['ticker']})</td>
            <td>{s['sector']}</td>
            <td style="color:{signal_color}; font-weight:bold;">{s['signal']}</td>
            <td>{s['p_up']*100:.1f}%</td>
            <td>{s['strength_label']}</td>
            <td>{s['last_close']:.2f}</td>
            <td>{s['pred_high']:.2f}</td>
            <td>{s['pred_low']:.2f}</td>
            <td>{s['pred_close']:.2f}</td>
            <td>{s['pred_return_pct']:+.2f}%</td>
            <td>{rsi_display}</td>
            <td>{s['hit_rate']}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
    <meta charset="UTF-8">
    <title>HSI 預測視覺化報告 — {result['target_trading_date']}</title>
    <style>
        body {{ font-family: "Microsoft JhengHei", Arial, sans-serif; background:#f5f6f8; color:#333; margin:0; padding:20px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{ background:#1a1a2e; color:white; padding:20px 24px; border-radius:10px; margin-bottom:20px; }}
        .header h1 {{ margin:0; font-size:22px; }}
        .header p {{ margin:6px 0 0; font-size:13px; color:#c5c5e0; }}
        .cards {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px; }}
        .card {{ background:white; border-radius:10px; padding:16px 20px; box-shadow:0 1px 4px rgba(0,0,0,0.08); flex:1; min-width:160px; }}
        .card .label {{ font-size:12px; color:#888; margin-bottom:4px; }}
        .card .value {{ font-size:20px; font-weight:bold; }}
        .chart-row {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px; }}
        .chart-box {{ background:white; border-radius:10px; padding:16px; box-shadow:0 1px 4px rgba(0,0,0,0.08); flex:1; min-width:300px; text-align:center; }}
        .chart-box img {{ max-width:100%; }}
        table {{ width:100%; border-collapse:collapse; background:white; border-radius:10px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,0.08); }}
        th {{ background:#1a1a2e; color:white; padding:10px 8px; font-size:12px; }}
        td {{ padding:8px; border-bottom:1px solid #eee; font-size:13px; text-align:center; }}
        tr:hover {{ background:#f9f9fb; }}
        .section-title {{ font-size:16px; font-weight:bold; margin:24px 0 12px; }}
        .footer {{ color:#999; font-size:11px; text-align:center; margin-top:24px; }}
    </style>
    </head>
    <body>
    <div class="container">

        <div class="header">
            <h1>HSI 次日預測視覺化報告</h1>
            <p>數據截數日: {result['data_as_of_date']} ｜ 預測交易日: {result['target_trading_date']} ｜
               生成時間: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
        </div>

        <div class="cards">
            <div class="card"><div class="label">Regime</div>
                <div class="value" style="color:{regime_color};">{result['regime']}</div></div>
            <div class="card"><div class="label">P(升)</div>
                <div class="value" style="color:#2e7d32;">{result['p_up']*100:.1f}%</div></div>
            <div class="card"><div class="label">值博率</div>
                <div class="value">{result.get('signal_strength_label','-')}</div></div>
            <div class="card"><div class="label">歷史命中率</div>
                <div class="value">{result['hit_rate']}</div></div>
            <div class="card"><div class="label">建議倉位</div>
                <div class="value">{result.get('position_size_pct', 0):.1f}%</div></div>
        </div>

        <div class="chart-row">
            <div class="chart-box">
                <img src="data:image/png;base64,{price_chart_b64}">
            </div>
        </div>

        <div class="chart-row">
            <div class="chart-box">
                <img src="data:image/png;base64,{prob_chart_b64}">
            </div>
            <div class="chart-box">
                <img src="data:image/png;base64,{signal_chart_b64}">
            </div>
        </div>

        <div class="section-title">個股預測一覽</div>
        <table>
            <tr>
                <th>股票</th><th>板塊</th><th>信號</th><th>P(升)</th><th>強度</th>
                <th>昨收</th><th>預測高位</th><th>預測低位</th><th>預測收市</th>
                <th>1日回報</th><th>RSI</th><th>命中率</th>
            </tr>
            {stock_rows}
        </table>

        <div class="footer">此報告僅供參考，不構成投資建議。命中率基於歷史回測，不代表未來表現。</div>
    </div>
    </body>
    </html>
    """
    return html


# ---------------------------------------------------------------------------
# Second, independent email — sends the dashboard as an attachment
# ---------------------------------------------------------------------------

def send_dashboard_email(sender_email, app_password, recipient_email, subject, html_path):
    recipients = [r.strip() for r in recipient_email.split(",") if r.strip()]

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject

    body = MIMEText(
        "附件為今日 HSI 次日預測視覺化報告（HTML 檔案），請下載並用瀏覽器開啟查看圖表。\n\n"
        "此為獨立報告，與每日文字版預測郵件分開發送。",
        "plain"
    )
    msg.attach(body)

    with open(html_path, "rb") as f:
        attachment = MIMEApplication(f.read(), _subtype="html")
        attachment.add_header("Content-Disposition", "attachment", filename="dashboard.html")
        msg.attach(attachment)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipients, msg.as_string())
    print(f"Dashboard email sent to {', '.join(recipients)}")


if __name__ == "__main__":
    SENDER_EMAIL = os.environ.get("EMAIL_SENDER")
    APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD")
    RECIPIENT_EMAIL = os.environ.get("EMAIL_RECIPIENT")

    if not all([SENDER_EMAIL, APP_PASSWORD, RECIPIENT_EMAIL]):
        raise SystemExit("Missing email credentials: EMAIL_SENDER, EMAIL_APP_PASSWORD, EMAIL_RECIPIENT")

    result = predict_today()

    html = build_dashboard_html(result)
    with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard saved to {DASHBOARD_PATH}")

    subject = f"HSI視覺化報告 {result['target_trading_date']} | {result['regime']} | P升{result['p_up']*100:.0f}%"
    send_dashboard_email(SENDER_EMAIL, APP_PASSWORD, RECIPIENT_EMAIL, subject, DASHBOARD_PATH)
