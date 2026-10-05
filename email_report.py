"""
Email report sender using Gmail SMTP + App Password.
Setup required: enable 2FA on your Gmail account, then generate an
"App Password" at https://myaccount.google.com/apppasswords — do NOT
use your regular Gmail password here.
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from pathlib import Path


def send_email_report(sender_email: str, app_password: str, recipient_email: str,
                       subject: str, html_body: str, attachment_paths: list = None):
    msg = MIMEMultipart("mixed")
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject

    msg.attach(MIMEText(html_body, "html"))

    if attachment_paths:
        for path in attachment_paths:
            path = Path(path)
            if not path.exists():
                print(f"Attachment not found, skipping: {path}")
                continue
            with open(path, "rb") as f:
                part = MIMEApplication(f.read(), Name=path.name)
            part["Content-Disposition"] = f'attachment; filename="{path.name}"'
            msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        print(f"Email sent successfully to {recipient_email}")
        return True
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False


def build_daily_signal_email_html(signal: dict) -> str:
    """Builds HTML email body for the daily next-day trading signal."""
    verdict_color = "#2e7d32" if signal["worth_trading"] else "#c62828"
    verdict_text = "值博" if signal["worth_trading"] else "不值博"
    direction_color = "#2e7d32" if signal["direction"] == "LONG" else "#c62828"

    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #333;">
        <h2>HSI 次日交易訊號 — {signal['signal_date']}</h2>
        <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
            <tr><td style="padding:8px; border:1px solid #ddd;"><b>方向</b></td>
                <td style="padding:8px; border:1px solid #ddd; color:{direction_color}; font-weight:bold;">{signal['direction']}</td></tr>
            <tr><td style="padding:8px; border:1px solid #ddd;"><b>信心分數</b></td>
                <td style="padding:8px; border:1px solid #ddd;">{signal['confidence_score']}</td></tr>
            <tr><td style="padding:8px; border:1px solid #ddd;"><b>值博與否</b></td>
                <td style="padding:8px; border:1px solid #ddd; color:{verdict_color}; font-weight:bold;">{verdict_text}</td></tr>
            <tr><td style="padding:8px; border:1px solid #ddd;" colspan="2">{signal['reason']}</td></tr>
            <tr><td style="padding:8px; border:1px solid #ddd;"><b>現價</b></td>
                <td style="padding:8px; border:1px solid #ddd;">{signal['last_close']}</td></tr>
            <tr><td style="padding:8px; border:1px solid #ddd;"><b>建議入場位</b></td>
                <td style="padding:8px; border:1px solid #ddd;">{signal['entry_price']}</td></tr>
            <tr><td style="padding:8px; border:1px solid #ddd;"><b>預測低位</b></td>
                <td style="padding:8px; border:1px solid #ddd;">{signal['pred_low']}</td></tr>
            <tr><td style="padding:8px; border:1px solid #ddd;"><b>預測高位</b></td>
                <td style="padding:8px; border:1px solid #ddd;">{signal['pred_high']}</td></tr>
            <tr><td style="padding:8px; border:1px solid #ddd;"><b>預測收市位</b></td>
                <td style="padding:8px; border:1px solid #ddd;">{signal['pred_close']}</td></tr>
            <tr><td style="padding:8px; border:1px solid #ddd;"><b>Risk:Reward</b></td>
                <td style="padding:8px; border:1px solid #ddd;">1:{signal['risk_reward_ratio']}</td></tr>
            <tr><td style="padding:8px; border:1px solid #ddd;"><b>數據來源</b></td>
                <td style="padding:8px; border:1px solid #ddd;">{signal['data_source']} (stale={signal['is_stale']})</td></tr>
        </table>
        <p style="color:#888; font-size:12px; margin-top:20px;">
            此訊號僅供參考，不構成投資建議。預測存在不確定性，請自行判斷風險。
        </p>
    </body>
    </html>
    """
    return html


def build_backtest_summary_email_html(summary: dict) -> str:
    """Builds HTML email body for the backtest summary report."""
    hsi = summary.get("hsi_summary", {})
    stocks = summary.get("stock_summaries", [])

    stock_rows = ""
    for s in stocks:
        if s.get("status") == "NO_DATA":
            continue
        stock_rows += f"""
        <tr>
            <td style="padding:6px; border:1px solid #ddd;">{s['label']}</td>
            <td style="padding:6px; border:1px solid #ddd;">{s['n_days']}</td>
            <td style="padding:6px; border:1px solid #ddd;">{s['directional_accuracy']}</td>
            <td style="padding:6px; border:1px solid #ddd;">{s['mean_abs_error_pct']}%</td>
        </tr>
        """

    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #333;">
        <h2>HSI 回測報告 ({summary.get('mode', 'N/A')} mode)</h2>
        <h3>HSI 大盤表現</h3>
        <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
            <tr><td style="padding:8px; border:1px solid #ddd;"><b>評估天數</b></td>
                <td style="padding:8px; border:1px solid #ddd;">{hsi.get('n_days')}</td></tr>
            <tr><td style="padding:8px; border:1px solid #ddd;"><b>日期範圍</b></td>
                <td style="padding:8px; border:1px solid #ddd;">{hsi.get('date_range')}</td></tr>
            <tr><td style="padding:8px; border:1px solid #ddd;"><b>方向準確度</b></td>
                <td style="padding:8px; border:1px solid #ddd;">{hsi.get('directional_accuracy')}</td></tr>
            <tr><td style="padding:8px; border:1px solid #ddd;"><b>平均誤差%</b></td>
                <td style="padding:8px; border:1px solid #ddd;">{hsi.get('mean_abs_error_pct')}%</td></tr>
            <tr><td style="padding:8px; border:1px solid #ddd;"><b>RMSE</b></td>
                <td style="padding:8px; border:1px solid #ddd;">{hsi.get('rmse')}</td></tr>
        </table>

        <h3>個股表現 (Bottom-up)</h3>
        <table style="border-collapse: collapse; width: 100%; max-width: 600px;">
            <tr style="background:#f0f0f0;">
                <th style="padding:6px; border:1px solid #ddd;">股票</th>
                <th style="padding:6px; border:1px solid #ddd;">天數</th>
                <th style="padding:6px; border:1px solid #ddd;">方向準確度</th>
                <th style="padding:6px; border:1px solid #ddd;">平均誤差%</th>
            </tr>
            {stock_rows}
        </table>

        <p style="color:#c62828; font-size:13px; margin-top:20px;">
            ⚠️ 若mode為'insample'，模型訓練時已包含此期間數據，結果偏樂觀，僅供系統驗證參考，
            不代表真實預測能力。如需真實評估請使用'walkforward'模式。
        </p>
    </body>
    </html>
    """
    return html


if __name__ == "__main__":
    import json
    import pandas as pd
    import os
    from config import PRED_DIR, SIGNAL_LOG_PATH

    # Credentials should be set as environment variables (GitHub Secrets / Colab Secrets)
    SENDER_EMAIL = os.environ.get("EMAIL_SENDER")
    APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD")
    RECIPIENT_EMAIL = os.environ.get("EMAIL_RECIPIENT")

    if not all([SENDER_EMAIL, APP_PASSWORD, RECIPIENT_EMAIL]):
        raise SystemExit("Missing email credentials in environment variables: "
                          "EMAIL_SENDER, EMAIL_APP_PASSWORD, EMAIL_RECIPIENT")

    # Send daily signal email (reads latest row from signals_log.csv)
    if SIGNAL_LOG_PATH.exists():
        log = pd.read_csv(SIGNAL_LOG_PATH)
        latest_signal = log.iloc[-1].to_dict()
        html = build_daily_signal_email_html(latest_signal)
        send_email_report(
            SENDER_EMAIL, APP_PASSWORD, RECIPIENT_EMAIL,
            subject=f"HSI次日訊號 {latest_signal['signal_date']} - {latest_signal['direction']}",
            html_body=html,
            attachment_paths=[SIGNAL_LOG_PATH],
        )
