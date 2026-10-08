"""
Email report sender — formatted to match the requested layout:
P(up)/P(down), signal strength, regime, HSI high/low/close/range,
verdict, and full per-stock prediction table.
"""

import os
import json
import smtplib
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import PRED_LOG_PATH
from predict import predict_today
from signal_generator import generate_signal


def send_email(sender_email, app_password, recipient_email, subject, html_body):
    msg = MIMEMultipart("alternative")
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
    print(f"Email sent to {recipient_email}")


def build_html_report(result: dict) -> str:
    regime_color = {"BULL": "#2e7d32", "BEAR": "#c62828", "NEUTRAL": "#757575"}.get(result["regime"], "#757575")
    verdict_text = result["worth_trading_reason"]

    # 修正：rr_display 移出迴圈，喺迴圈之前計好一次，避免 NameError（當 _stock_predictions 係空list）
    signal = generate_signal(result)
    rr_display = f"1 : {signal['risk_reward_ratio']}" if signal["risk_reward_ratio"] else "N/A"

    stock_rows = ""
    for s in result["_stock_predictions"]:
        signal_color = "#2e7d32" if s["signal"] == "LONG" else "#c62828"
        rsi_display = f"{s['rsi']:.1f}" if s['rsi'] is not None else 'N/A'

        stock_rows += f"""
        <tr>
            <td style="padding:6px; border:1px solid #ddd;">{s['name']} ({s['ticker']})</td>
            <td style="padding:6px; border:1px solid #ddd;">{s['sector']}</td>
            <td style="padding:6px; border:1px solid #ddd; color:{signal_color}; font-weight:bold;">{s['signal']}</td>
            <td style="padding:6px; border:1px solid #ddd;">{s['p_up']*100:.1f}%</td>
            <td style="padding:6px; border:1px solid #ddd;">{s['strength_label']}</td>
            <td style="padding:6px; border:1px solid #ddd;">{s['last_close']:.2f}</td>
            <td style="padding:6px; border:1px solid #ddd;">{s['pred_high']:.2f}</td>
            <td style="padding:6px; border:1px solid #ddd;">{s['pred_low']:.2f}</td>
            <td style="padding:6px; border:1px solid #ddd;">{s['pred_close']:.2f}</td>
            <td style="padding:6px; border:1px solid #ddd;">{s['pred_return_pct']:+.2f}%</td>
            <td style="padding:6px; border:1px solid #ddd;">{rsi_display}</td>
            <td style="padding:6px; border:1px solid #ddd;">{s['hit_rate']}</td>
        </tr>
        """

    html = f"""
    <html><body style="font-family: Arial, sans-serif; color:#333;">
    <h2>HSI 次日預測報告</h2>
    <p style="font-size:14px; color:#555;">
        <b>數據截數日:</b> {result['data_as_of_date']} &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>預測交易日:</b> {result['target_trading_date']}
    </p>

    <table style="border-collapse: collapse; width:100%; max-width:600px; margin-bottom:20px;">
        <tr>
            <td style="padding:8px; border:1px solid #ddd;"><b>P(升)</b></td>
            <td style="padding:8px; border:1px solid #ddd; color:#2e7d32; font-weight:bold;">{result['p_up']*100:.1f}%</td>
            <td style="padding:8px; border:1px solid #ddd;"><b>P(跌)</b></td>
            <td style="padding:8px; border:1px solid #ddd; color:#c62828; font-weight:bold;">{result['p_down']*100:.1f}%</td>
        </tr>

        <tr>
            <td style="padding:8px; border:1px solid #ddd;"><b>值博率 (Risk:Reward)</b></td>
            <td style="padding:8px; border:1px solid #ddd; font-weight:bold;" colspan="3">{rr_display}</td>
        </tr>
        <tr>
            <td style="padding:8px; border:1px solid #ddd;"><b>歷史命中率 (HSI)</b></td>
            <td style="padding:8px; border:1px solid #ddd; font-weight:bold;" colspan="3">{result['hit_rate']}</td>
        </tr>

        <!-- 新加：校準訊號 (Regime-based threshold) -->
        <tr>
            <td style="padding:8px; border:1px solid #ddd;"><b>校準訊號 (Regime校準)</b></td>
            <td style="padding:8px; border:1px solid #ddd; font-weight:bold; color:{
                '#2e7d32' if result['calibrated_signal']=='LONG' else
                '#c62828' if result['calibrated_signal']=='SHORT' else '#757575'
            };" colspan="3">
                {result['calibrated_signal']}
                （門檻: Long≥{result['calibrated_signal_thresholds']['long']:.2f} /
                Short≤{result['calibrated_signal_thresholds']['short']:.2f}）
            </td>
        </tr>

        <!-- 新加：建議倉位 (Fractional Kelly) -->
        <tr>
            <td style="padding:8px; border:1px solid #ddd;"><b>建議倉位 (Kelly)</b></td>
            <td style="padding:8px; border:1px solid #ddd; font-weight:bold;" colspan="3">
                {result['position_size_pct']:.1f}%
            </td>
        </tr>

        <tr>
            <td style="padding:8px; border:1px solid #ddd;"><b>信號強度</b></td>
            <td style="padding:8px; border:1px solid #ddd;" colspan="3">
                {result['signal_strength_label']} ({result['signal_strength_margin']}%)
            </td>
        </tr>
        <tr>
            <td style="padding:8px; border:1px solid #ddd;"><b>Regime</b></td>
            <td style="padding:8px; border:1px solid #ddd; color:{regime_color}; font-weight:bold;" colspan="3">
                {result['regime']}
            </td>
        </tr>
        <tr>
            <td style="padding:8px; border:1px solid #ddd;"><b>預測高位</b></td>
            <td style="padding:8px; border:1px solid #ddd;">{result['pred_high']:,.0f}</td>
            <td style="padding:8px; border:1px solid #ddd;" colspan="2">{result['pred_high_pct']:+.2f}%</td>
        </tr>
        <tr>
            <td style="padding:8px; border:1px solid #ddd;"><b>今日收市</b></td>
            <td style="padding:8px; border:1px solid #ddd;">{result['last_close']:,.0f}</td>
            <td style="padding:8px; border:1px solid #ddd;" colspan="2">基準</td>
        </tr>
        <tr>
            <td style="padding:8px; border:1px solid #ddd;"><b>預測低位</b></td>
            <td style="padding:8px; border:1px solid #ddd;">{result['pred_low']:,.0f}</td>
            <td style="padding:8px; border:1px solid #ddd;" colspan="2">{result['pred_low_pct']:+.2f}%</td>
        </tr>
        <tr>
            <td style="padding:8px; border:1px solid #ddd;"><b>預測範圍</b></td>
            <td style="padding:8px; border:1px solid #ddd;">{result['pred_range_points']:,.0f} 點</td>
            <td style="padding:8px; border:1px solid #ddd;" colspan="2">{result['pred_range_pct']:+.2f}%</td>
        </tr>
    </table>

    <p style="padding:10px; background:#fff3e0; border-left:4px solid #ff9800;">
        <b>{verdict_text}</b><br>
        留意高位 {result['pred_high']:,.0f} / 低位 {result['pred_low']:,.0f}
    </p>

    <h3>個股預測</h3>
    <table style="border-collapse: collapse; width:100%; font-size:13px;">
        <tr style="background:#f0f0f0;">
            <th style="padding:6px; border:1px solid #ddd;">股票</th>
            <th style="padding:6px; border:1px solid #ddd;">板塊</th>
            <th style="padding:6px; border:1px solid #ddd;">信號</th>
            <th style="padding:6px; border:1px solid #ddd;">P(升)</th>
            <th style="padding:6px; border:1px solid #ddd;">強度</th>
            <th style="padding:6px; border:1px solid #ddd;">昨收</th>
            <th style="padding:6px; border:1px solid #ddd;">預測高位</th>
            <th style="padding:6px; border:1px solid #ddd;">預測低位</th>
            <th style="padding:6px; border:1px solid #ddd;">預測收市</th>
            <th style="padding:6px; border:1px solid #ddd;">1日回報</th>
            <th style="padding:6px; border:1px solid #ddd;">RSI</th>
            <th style="padding:6px; border:1px solid #ddd;">命中率</th>
        </tr>
        {stock_rows}
    </table>

    <p style="color:#888; font-size:12px; margin-top:20px;">
        此報告僅供參考，不構成投資建議。命中率基於歷史回測，不代表未來表現。
    </p>
    </body></html>
    """
    return html


if __name__ == "__main__":
    SENDER_EMAIL = os.environ.get("EMAIL_SENDER")
    APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD")
    RECIPIENT_EMAIL = os.environ.get("EMAIL_RECIPIENT")

    if not all([SENDER_EMAIL, APP_PASSWORD, RECIPIENT_EMAIL]):
        raise SystemExit("Missing email credentials: EMAIL_SENDER, EMAIL_APP_PASSWORD, EMAIL_RECIPIENT")

    result = predict_today()
    html = build_html_report(result)

    subject = (f"HSI預測 {result['target_trading_date']} "
           f"(數據:{result['data_as_of_date']}) | {result['regime']} | "
           f"{result['calibrated_signal']} | P升{result['p_up']*100:.0f}%")

    send_email(SENDER_EMAIL, APP_PASSWORD, RECIPIENT_EMAIL, subject, html)
