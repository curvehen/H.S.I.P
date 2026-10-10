"""
Standalone HTML dashboard generator — separate from email_report.py.
Produces a visual dashboard (predictions/dashboard.html) with HSI charts
(price trend + predicted range, P(up)/P(down) gauge, regime badge) and
the per-stock + watchlist prediction tables. Also sends a SEPARATE email
containing this dashboard as an attachment, independent from the existing
daily text-style email in email_report.py (which remains untouched).

FIX LOG (this revision):
  - REMOVED `from predict import predict_today` (function never existed —
    guaranteed ImportError on every run) and the resulting re-execution of
    the ENTIRE prediction pipeline a second time inside this script. That
    pattern risked producing a SECOND, slightly different prediction
    (different data-fetch timestamp than the official predict.py run
    earlier in the same GitHub Actions job) and wasted a full pipeline
    run's worth of compute/API calls for no reason. This dashboard now
    reads the SAME LATEST_RESULT_PATH JSON that predict.py already wrote
    earlier in the same job, exactly like email_report.py does — guaranteeing
    both reports describe the identical prediction run.
  - Every field access realigned to predict.py's ACTUAL result dict keys
    (confirmed by reading predict.py's source directly): target_trading_date
    -> date, pred_high/pred_low/pred_close -> pred_high_price/
    pred_low_price/pred_close_price, p_down (didn't exist) -> derived as
    1 - p_up, signal_strength_label -> signal_strength, hit_rate (HSI-level,
    didn't exist) -> fetched via hit_rate_tracker.get_hit_rate(HSI_TICKER),
    data_as_of_date (didn't exist) -> replaced with a clearly-labeled
    report-generation timestamp.
  - _stock_predictions (single merged list with nonexistent per-row keys
    signal/rsi/strength_label/pred_high/pred_low/pred_close/pred_return_pct)
    replaced with predict.py's actual two separate keys:
    result["stock_predictions"] (HSI constituents) and
    result["watchlist_predictions"] (1211.HK, 0968.HK — independent,
    never folded into bottom-up weighting). Absolute per-stock price
    levels and direction are now DERIVED here from the real return-based
    fields (pred_close_return, pred_high_return, pred_low_return, p_up),
    since predict_single_stock() does not emit absolute prices, an RSI
    value, or a pre-labeled signal string directly.
  - position_size_pct is a FRACTION in [0, max_position_pct] per
    position_sizer.py's actual contract (e.g. 0.08 = 8%), not a 0-100
    number as the previous chart code assumed. All display/chart code now
    multiplies by 100 explicitly before rendering as a percentage.
  - Added a run_mode guard in __main__: this dashboard is intended for
    OFFICIAL runs only (per daily_predict.yml, which already gates this
    step to the official job) — but the script now also self-checks
    RUN_MODE and exits cleanly with a message if invoked under
    'preliminary', rather than silently building a dashboard for a run
    that was never meant to produce one.
  - Added a watchlist table section, previously completely absent, to
    satisfy the ENH#4 requirement that watchlist tickers get their own
    visible, clearly-separated reporting section.
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

from config import PRED_DIR, HSI_TICKER, LATEST_RESULT_PATH, DASHSCOPE_MODEL_NAME
from data_sources import fetch_with_fallback
from hit_rate_tracker import get_hit_rate
from market_hours import HKT

DASHBOARD_PATH = PRED_DIR / "dashboard.html"
CHART_HISTORY_DAYS = 30


# ---------------------------------------------------------------------------
# Result loading — reads the SAME JSON predict.py wrote earlier in this
# same job, instead of re-running the pipeline a second time.
# ---------------------------------------------------------------------------

def _load_result() -> dict:
    if not LATEST_RESULT_PATH.exists():
        raise FileNotFoundError(
            f"dashboard_report: {LATEST_RESULT_PATH} not found. "
            f"Ensure predict.py has run with RUN_MODE='official' earlier in this job."
        )
    import json
    with open(LATEST_RESULT_PATH) as f:
        return json.load(f)


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

    pred_high = result["pred_high_price"]
    pred_low = result["pred_low_price"]
    pred_close = result["pred_close_price"]
    last_close = result["last_close"]

    ax.plot([last_date, next_date], [last_close, pred_close],
            color="#ff6d00", linestyle="--", linewidth=1.8, marker="o", markersize=5,
            label="Predicted Close")
    ax.fill_between([last_date, next_date], [last_close, pred_low], [last_close, pred_high],
                     color="#ff6d00", alpha=0.15, label="Predicted High/Low Range")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_title(f"HSI Last {CHART_HISTORY_DAYS} Days + {result.get('date', 'N/A')} Forecast Range", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate(rotation=30)

    return _fig_to_base64(fig)


def generate_probability_gauge(result: dict) -> str:
    """Horizontal bar comparing P(up) vs P(down). predict.py only emits
    p_up; p_down is derived here as the complement."""
    p_up = (result.get("p_up") or 0.5) * 100
    p_down = 100 - p_up

    fig, ax = plt.subplots(figsize=(6, 1.6))
    ax.barh(["Probability"], [p_up], color="#2e7d32", label=f"P(Up) {p_up:.1f}%")
    ax.barh(["Probability"], [p_down], left=[p_up], color="#c62828", label=f"P(Down) {p_down:.1f}%")

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_title("P(Up) vs P(Down)", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize=9, frameon=False)

    return _fig_to_base64(fig)


def generate_signal_strength_chart(result: dict) -> str:
    """Donut chart showing calibrated signal direction + recommended position size.
    position_size_pct is a FRACTION (e.g. 0.08) per position_sizer.py's
    actual contract — multiplied by 100 here before any percentage display."""
    position_pct = (result.get("position_size_pct") or 0.0) * 100
    signal = result.get("calibrated_signal", "觀望")
    signal_display = {"LONG": "LONG", "SHORT": "SHORT", "觀望": "WAIT"}.get(signal, signal)
    color = {"LONG": "#2e7d32", "SHORT": "#c62828", "觀望": "#757575"}.get(signal, "#757575")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie([position_pct, max(100 - position_pct, 0)],
           colors=[color, "#e0e0e0"], startangle=90, counterclock=False,
           wedgeprops={"width": 0.35})
    ax.text(0, 0.1, signal_display, ha="center", va="center", fontsize=16, fontweight="bold", color=color)
    ax.text(0, -0.15, f"Position {position_pct:.1f}%", ha="center", va="center", fontsize=11, color="#555")
    ax.set_title("Calibrated Signal & Position Size", fontsize=11)

    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Per-stock table row builder (shared by constituents + watchlist tables).
# predict_single_stock() emits return-based fields only — absolute prices
# and a direction label are derived here relative to last_close.
# ---------------------------------------------------------------------------

def _stock_row_html(s: dict) -> str:
    last_close = s.get("last_close")
    pred_close_return = s.get("pred_close_return")
    pred_high_return = s.get("pred_high_return")
    pred_low_return = s.get("pred_low_return")

    pred_close_price = (last_close * (1 + pred_close_return)
                         if last_close is not None and pred_close_return is not None else None)
    pred_high_price = (last_close * (1 + pred_high_return)
                        if last_close is not None and pred_high_return is not None else None)
    pred_low_price = (last_close * (1 + pred_low_return)
                       if last_close is not None and pred_low_return is not None else None)

    p_up = s.get("p_up")
    if p_up is not None and p_up >= 0.55:
        signal, signal_color = "LONG", "#2e7d32"
    elif p_up is not None and p_up <= 0.45:
        signal, signal_color = "SHORT", "#c62828"
    else:
        signal, signal_color = "NEUTRAL", "#757575"

    strength_display = {"weak": "Weak", "medium": "Medium", "strong": "Strong"}.get(
        s.get("signal_strength"), s.get("signal_strength", "N/A"))

    def _fmt(x):
        return f"{x:.2f}" if x is not None else "N/A"

    pred_return_pct_display = f"{pred_close_return*100:+.2f}%" if pred_close_return is not None else "N/A"

    return f"""
    <tr>
        <td>{s.get('name', 'N/A')} ({s.get('ticker', 'N/A')})</td>
        <td>{s.get('sector', 'N/A')}</td>
        <td style="color:{signal_color}; font-weight:bold;">{signal}</td>
        <td>{p_up*100:.1f}%</td>
        <td>{strength_display}</td>
        <td>{_fmt(last_close)}</td>
        <td>{_fmt(pred_high_price)}</td>
        <td>{_fmt(pred_low_price)}</td>
        <td>{_fmt(pred_close_price)}</td>
        <td>{pred_return_pct_display}</td>
        <td>{s.get('hit_rate', 'N/A')}</td>
    </tr>
    """


def _stock_table_rows(rows: list) -> str:
    if not rows:
        return '<tr><td colspan="10" style="color:#999;">No data available.</td></tr>'
    return "".join(_stock_row_html(s) for s in rows)


# ---------------------------------------------------------------------------
# HTML dashboard assembly
# ---------------------------------------------------------------------------

def build_dashboard_html(result: dict) -> str:
    price_chart_b64 = generate_price_trend_chart(result)
    prob_chart_b64 = generate_probability_gauge(result)
    signal_chart_b64 = generate_signal_strength_chart(result)

    regime_color = {"BULL": "#2e7d32", "BEAR": "#c62828", "NEUTRAL": "#757575"}.get(
        result.get("regime"), "#757575")

    hsi_hit_rate = result.get("hit_rate")
    if hsi_hit_rate is None:
        try:
            hsi_hit_rate = get_hit_rate(HSI_TICKER)
        except Exception:
            hsi_hit_rate = "N/A"

    constituent_rows = _stock_table_rows(result.get("stock_predictions") or [])
    watchlist_rows = _stock_table_rows(result.get("watchlist_predictions") or [])

    report_time = datetime.datetime.now(HKT).strftime("%Y-%m-%d %H:%M:%S HKT")
    position_pct_display = (result.get("position_size_pct") or 0.0) * 100

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>HSI Prediction Dashboard — {result.get('date', 'N/A')}</title>
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
        .note {{ color:#666; font-size:12px; margin:-6px 0 10px; }}
        .footer {{ color:#999; font-size:11px; text-align:center; margin-top:24px; }}
    </style>
    </head>
    <body>
    <div class="container">

        <div class="header">
            <h1>HSI Next-Day Prediction Dashboard</h1>
            <p>Target trading date: {result.get('date', 'N/A')} | Run mode: {result.get('run_mode', 'N/A')} |
               Report generated: {report_time}</p>
        </div>

        <div class="cards">
            <div class="card"><div class="label">Regime</div>
                <div class="value" style="color:{regime_color};">{result.get('regime', 'N/A')}</div></div>
            <div class="card"><div class="label">P(Up)</div>
                <div class="value" style="color:#2e7d32;">{(result.get('p_up') or 0)*100:.1f}%</div></div>
            <div class="card"><div class="label">Signal Strength</div>
                <div class="value">{result.get('signal_strength', 'N/A')}</div></div>
            <div class="card"><div class="label">Historical Hit Rate (HSI)</div>
                <div class="value">{hsi_hit_rate}</div></div>
            <div class="card"><div class="label">Recommended Position</div>
                <div class="value">{position_pct_display:.1f}%</div></div>
        </div>

        <div class="chart-row">
            <div class="chart-box">
                <img src="data:image/png;base64,{price_chart_b64}" alt="Price trend chart">
            </div>
            <div class="chart-box">
                <img src="data:image/png;base64,{prob_chart_b64}" alt="Probability gauge">
            </div>
            <div class="chart-box">
                <img src="data:image/png;base64,{signal_chart_b64}" alt="Signal strength donut">
            </div>
        </div>

        <div class="section-title">HSI Forecast Detail</div>
        <table>
            <tr>
                <th>Last Close</th><th>Est. Entry Price</th><th>Pred. High</th>
                <th>Pred. Low</th><th>Pred. Close</th><th>Expected Move</th>
                <th>Confidence</th><th>Worth Trading</th><th>Risk/Reward</th>
            </tr>
            <tr>
                <td>{result.get('last_close', 0):.1f}</td>
                <td>{(result.get('estimated_entry_price') or result.get('last_close', 0)):.1f}
                    {'' if result.get('gap_model_status') == 'FITTED' else f"({result.get('gap_model_status', 'N/A')})"}</td>
                <td>{result.get('pred_high_price', 0):.1f}</td>
                <td>{result.get('pred_low_price', 0):.1f}</td>
                <td>{result.get('pred_close_price', 0):.1f}</td>
                <td>{(result.get('expected_move_pct') or 0)*100:.2f}%</td>
                <td>{result.get('confidence', 0):.2f}</td>
                <td>{'Yes ✅' if result.get('worth_trading') else 'No ⏸️'}</td>
                <td>{result.get('risk_reward_ratio', 'N/A')}</td>
            </tr>
        </table>

        <div class="section-title">HSI Constituent Stock Predictions</div>
        <table>
            <tr>
                <th>Stock</th><th>Sector</th><th>Signal</th><th>P(Up)</th><th>Strength</th>
                <th>Last Close</th><th>Pred. High</th><th>Pred. Low</th><th>Pred. Close</th>
                <th>Pred. Return</th><th>Hit Rate</th>
            </tr>
            {constituent_rows}
        </table>

        <div class="section-title">Watchlist (Independent — Excluded from HSI Bottom-Up Weighting)</div>
        <p class="note">These tickers (1211.HK, 0968.HK) are tracked independently and never contribute to
           predict_bottom_up()'s weighted HSI return.</p>
        <table>
            <tr>
                <th>Stock</th><th>Sector</th><th>Signal</th><th>P(Up)</th><th>Strength</th>
                <th>Last Close</th><th>Pred. High</th><th>Pred. Low</th><th>Pred. Close</th>
                <th>Pred. Return</th><th>Hit Rate</th>
            </tr>
            {watchlist_rows}
        </table>

        {_build_llm_block(result)}

        <div class="footer">
            Generated by dashboard_report.py | LLM model: {DASHSCOPE_MODEL_NAME if result.get('llm_commentary') else 'N/A'}
        </div>

    </div>
    </body>
    </html>
    """
    return html


def _build_llm_block(result: dict) -> str:
    commentary = result.get("llm_commentary")
    if not commentary:
        return ""
    return f"""
    <div class="section-title">AI Market Commentary</div>
    <div class="card" style="text-align:left;">{commentary}</div>
    """


# ---------------------------------------------------------------------------
# Email delivery (dashboard attached as standalone HTML file)
# ---------------------------------------------------------------------------

def send_dashboard_email(html_content: str, result: dict):
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_APP_PASSWORD")
    recipient = os.environ.get("EMAIL_RECIPIENT")
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))

    if not (sender and password and recipient):
        print("dashboard_report: EMAIL_SENDER/EMAIL_APP_PASSWORD/EMAIL_RECIPIENT not configured — skipping send.")
        return False

    msg = MIMEMultipart()
    msg["Subject"] = f"HSI Dashboard Report — {result.get('date', 'N/A')}"
    msg["From"] = sender
    msg["To"] = recipient

    msg.attach(MIMEText("Dashboard attached. Open the HTML file in a browser to view.", "plain"))
    attachment = MIMEApplication(html_content.encode("utf-8"), _subtype="html")
    attachment.add_header("Content-Disposition", "attachment", filename=f"hsi_dashboard_{result.get('date', 'na')}.html")
    msg.attach(attachment)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, [recipient], msg.as_string())
        print("dashboard_report: dashboard email sent successfully.")
        return True
    except Exception as e:
        print(f"dashboard_report: failed to send dashboard email ({e})")
        return False


def main():
    run_mode = os.environ.get("RUN_MODE", "official")
    if run_mode != "official":
        print(f"dashboard_report: run_mode='{run_mode}' — dashboard is official-run only, skipping.")
        return

    dashboard_enabled = os.environ.get("DASHBOARD_ENABLED", "true").lower() in ("1", "true", "yes")
    if not dashboard_enabled:
        print("dashboard_report: DASHBOARD_ENABLED is false — skipping.")
        return

    result = _load_result()
    html_content = build_dashboard_html(result)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"dashboard_report: dashboard saved to {DASHBOARD_PATH}")

    send_dashboard_email(html_content, result)


if __name__ == "__main__":
    main()

            
