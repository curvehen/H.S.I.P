"""
Signal Generator — formats predict.py output into a clean, human-readable
next-day trading signal: entry / high / low / close / worth-trading verdict.
"""

import json
import pandas as pd
from config import PRED_LOG_PATH, SIGNAL_LOG_PATH


def generate_signal(prediction: dict) -> dict:
    direction = "LONG" if prediction["pred_close_return_blended"] > 0 else "SHORT"

    signal = {
        "signal_date": prediction["predict_date"],
        "direction": direction,
        "confidence_score": round(prediction["signal_confidence"], 4),
        "worth_trading": prediction["worth_trading"],
        "reason": prediction["worth_trading_reason"],
        "last_close": round(prediction["last_close"], 1),
        "entry_price": round(prediction["entry_price"], 1),
        "pred_low": round(prediction["pred_low"], 1),
        "pred_high": round(prediction["pred_high"], 1),
        "pred_close": round(prediction["pred_close"], 1),
        "data_source": prediction.get("data_source"),
        "is_stale": prediction.get("is_stale"),
    }

    risk = abs(signal["entry_price"] - signal["pred_low"])
    reward = abs(signal["pred_close"] - signal["entry_price"])
    signal["risk_reward_ratio"] = round(reward / risk, 2) if risk > 0 else None

    return signal


def format_signal_text(signal: dict) -> str:
    verdict_cn = "值博" if signal["worth_trading"] else "不值博"
    lines = [
        f"=== HSI 次日交易訊號 — {signal['signal_date']} ===",
        f"方向: {signal['direction']}  (信心分數: {signal['confidence_score']})",
        f"值博與否: {verdict_cn}  ({signal['reason']})",
        f"現價 (Last Close): {signal['last_close']}",
        f"建議入場位: {signal['entry_price']}",
        f"預測低位: {signal['pred_low']}",
        f"預測高位: {signal['pred_high']}",
        f"預測收市位: {signal['pred_close']}",
        f"Risk:Reward = 1:{signal['risk_reward_ratio']}",
        f"數據來源: {signal['data_source']} (stale={signal['is_stale']})",
    ]
    return "\n".join(lines)


def save_signal(signal: dict):
    row = pd.DataFrame([signal])
    if SIGNAL_LOG_PATH.exists():
        log = pd.read_csv(SIGNAL_LOG_PATH)
        log = pd.concat([log, row], ignore_index=True)
    else:
        log = row
    log.to_csv(SIGNAL_LOG_PATH, index=False)


if __name__ == "__main__":
    log = pd.read_csv(PRED_LOG_PATH)
    latest = log.iloc[-1].to_dict()

    signal = generate_signal(latest)
    save_signal(signal)

    print(format_signal_text(signal))
    print("\nFull signal (JSON):")
    print(json.dumps(signal, indent=2, default=str))
