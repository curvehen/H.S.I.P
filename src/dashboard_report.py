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
    """Last 30 days HSI close price trend + today's predicted high/low/close range (shaded)."""
    raw = fetch_with_fallback(HSI_TICKER, stooq_ticker="^hsi")
    recent = raw.tail(CHART_HISTORY_DAYS).copy()

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(recent.index, recent["Close"], color="#1a73e8", linewidth=1.8, label="Historical Close")

    last_date = recent.index[-1]
    next_date = last_date + pd.Timedelta(days=1)

    pred_high = result["pred_high"]
    pred_low = result["pred_low"]
    pred_close = result["pred_close"]
    last_close = result["last_close"]

    ax.plot([last_date, next_date], [last_close, pred_close],
            color="#ff6d00", linestyle="--", linewidth=1.8, marker="o", markersize=5,
            label="Predicted Close")
    ax.fill_between([last_date, next_date], [last_close, pred_low], [last_close, pred_high],
                     color="#ff6d00", alpha=0.15, label="Predicted High/Low Range")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_title(f"HSI Last {CHART_HISTORY_DAYS} Days + {result['target_trading_date']} Forecast Range", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate(rotation=30)

    return _fig_to_base64(fig)


def generate_probability_gauge(result: dict) -> str:
    """Horizontal bar comparing P(up) vs P(down)."""
    p_up = result["p_up"] * 100
    p_down = result["p_down"] * 100

    fig, ax = plt.subplots(figsize=(6, 1.6))
    ax.barh(["Probability"], [p_up], color="#2e7d32", label=f"P(Up) {p_up:.1f}%")
    ax.barh(["Probability"], [p_down], left=[p_up], color="#c62828", label=f"P(Down) {p_down:.1f}%")

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_title("P(Up) vs P(Down)", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize=9, frameon=False)

    return _fig_to_base64(fig)


def generate_signal_strength_chart(result: dict) -> str:
    """Donut chart showing calibrated signal direction + recommended position size."""
    position_pct = result.get("position_size_pct", 0)
    signal = result.get("calibrated_signal", "WAIT")
    signal_display = {"LONG": "LONG", "SHORT": "SHORT", "觀望": "WAIT"}.get(signal, signal)
    color = {"LONG": "#2e7d32", "SHORT": "#c62828", "觀望": "#757575", "WAIT": "#757575"}.get(signal, "#757575")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie([position_pct, max(100 - position_pct, 0)],
           colors=[color, "#e0e0e0"], startangle=90, counterclock=False,
           wedgeprops={"width": 0.35})
    ax.text(0, 0.1, signal_display, ha="center", va="center", fontsize=16, fontweight="bold", color=color)
    ax.text(0, -0.15, f"Position {position_pct:.1f}%", ha="center", va="center", fontsize=11, color="#555")
    ax.set_title("Calibrated Signal & Position Size", fontsize=11)

    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML dashboard assembly
# ---------------------------------------------------------------------------

def build_dashboard_html(result: dict) -> str:
    price_chart_b64 = generate_price_trend_chart(result)
    prob_chart_b64 = generate_probability_gauge(result)
    signal_chart_b64 = generate_signal_strength_chart(result)

    regime_color = {"BULL": "#2e7d32", "BEAR": "#c62828", "NEUTRAL": "#757575"}.get(result["regime"], "#757575")
    signal_label = result.get("calibrated_signal", "觀望")
    signal_display = {"LONG": "LONG", "SHORT": "SHORT", "觀望": "WAIT"}.get(signal_label, signal_label)

    stock_rows = ""
    for s in result["_stock_predictions"]:
        signal_color = "#2e7d32" if s["signal"] == "LONG" else "#c62828"
        rsi_display = f"{s['rsi']:.1f}" if s['rsi'] is not None else 'N/A'
        strength_display = {"弱": "Weak", "中": "Medium", "強": "Strong"}.get(s['strength_label'], s['strength_label'])
        stock_rows += f"""
        <tr>
            <td>{s['name']} ({s['ticker']})</td>
            <td>{s['sector']}</td>
            <td style="color:{signal_color}; font-weight:bold;">{s['signal']}</td>
            <td>{s['p_up']*100:.1f}%</td>
            <td>{strength_display}</td>
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
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>HSI Prediction Dashboard — {result['target_trading_date']}</title>
    <style>
        body {{ font-family: Arial, Helvetica, sans-serif; background:#f5f6f8; color:#333; margin:0; padding:20px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{ background:#1a1a2e; color:white; padding:20px 24px; border-radius:10px; margin-bottom:20px; }}
        .header h1 {{ margin:0; font-size:22px; }}
        .header p {{ margin:6px 0 0; font-size:13px; color:#c5c5e0; }}
        .cards {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px; }}
        .card {{ background:white; border-radius:10px; padding:16px 20px; box-shadow:0 1px 4px rgba(0,0,0,0.08); flex:1; min-width:160px; }}
        .card .label {{ font-size:12px; color:#888; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.5px; }}
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
            <h1>HSI Next-Day Prediction Dashboard</h1>
            <p>Data as of: {result['data_as_of_date']} | Target trading date: {result['target_trading_date']} |
               Generated: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
        </div>

        <div class="cards">
            <div class="card"><div class="label">Regime</div>
                <div class="value" style="color:{regime_color};">{result['regime']}</div></div>
            <div class="card"><div class="label">P(Up)</div>
                <div class="value" style="color:#2e7d32;">{result['p_up']*100:.1f}%</div></div>
            <div class="card"><div class="label">Signal Strength</div>
                <div class="value">{result.get('signal_strength_label','-')}</div></div>
            <div class="card"><div class="label">Historical Hit Rate</div>
                <div class="value">{result['hit_rate']}</div></div>
            <div class="card"><div class="label">Recommended Position</div>
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

        <div class="section-title">Per-Stock Predictions</div>
        <table>
            <tr>
                <th>Stock</th><th>Sector</th><th>Signal</th><th>P(Up)</th><th>Strength</th>
                <th>Last Close</th><th>Pred. High</th><th>Pred. Low</th><th>Pred. Close</th>
                <th>1D Return</th><th>RSI</th><th>Hit Rate</th>
            </tr>
            {stock_rows}
        </table>

        <div class="footer">This report is for reference only and does not constitute investment advice. Hit rates are based on historical backtesting and do not guarantee future performance.</div>
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
        "Attached is today's HSI next-day prediction visual dashboard (HTML file). "
        "Please download and open it in a web browser to view the charts.\n\n"
        "This is a separate report, sent independently from the daily text-based prediction email.",
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

    subject = f"HSI Prediction Dashboard {result['target_trading_date']} | {result['regime']} | P(Up) {result['p_up']*100:.0f}%"
    send_dashboard_email(SENDER_EMAIL, APP_PASSWORD, RECIPIENT_EMAIL, subject, DASHBOARD_PATH)
