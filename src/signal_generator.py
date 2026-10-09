"""
Signal Generator — formats predict.py output (read from LATEST_RESULT_PATH,
the single-source-of-truth JSON for the current run) into a clean,
human-readable next-day trading signal: entry / high / low / close /
regime-calibrated direction / position size / worth-trading verdict.

NOTE: Does NOT re-read predictions_log.csv's last row. Under the dual-run
architecture (official vs preliminary), predictions_log.csv is only appended
on official, non-cached runs — reading its tail would return a stale prior
day's prediction whenever this script runs under preliminary mode. Instead,
every invocation of predict.py (official or preliminary) freshly overwrites
LATEST_RESULT_PATH, guaranteeing this script always reflects the exact same
numbers produced in that run.
"""

import json
import pandas as pd
from config import SIGNAL_LOG_PATH, LATEST_RESULT_PATH


def generate_signal(prediction: dict) -> dict:
    # raw_direction: 純粹基於 blended return 正負號嘅方向（未經 regime 門檻過濾）。
    # calibrated_signal: ENH#1 regime-calibrated threshold 過濾後嘅實際交易訊號，
    #   值為 "LONG" / "SHORT" / "觀望"（信號落喺兩個門檻之間，不建議進場）。
    raw_direction = "LONG" if prediction["pred_close_return_blended"] > 0 else "SHORT"
    calibrated_signal = prediction.get("calibrated_signal", raw_direction)

    signal = {
        "signal_date": prediction.get("target_trading_date", prediction.get("predict_date")),
        "data_as_of_date": prediction.get("data_as_of_date"),
        "raw_direction": raw_direction,
        "calibrated_signal": calibrated_signal,
        "regime": prediction.get("regime"),
        "p_up": prediction.get("p_up"),
        "confidence_score": round(prediction["signal_confidence"], 4),
        "position_size_pct": prediction.get("position_size_pct"),
        "worth_trading": prediction["worth_trading"],
        "reason": prediction["worth_trading_reason"],
        "last_close": round(prediction["last_close"], 1),
        "entry_price": round(prediction["entry_price"], 1),
        "pred_low": round(prediction["pred_low"], 1),
        "pred_high": round(prediction["pred_high"], 1),
        "pred_close": round(prediction["pred_close"], 1),
        "hit_rate": prediction.get("hit_rate"),
        "data_source": prediction.get("data_source"),
        "is_stale": prediction.get("is_stale"),
        "run_mode": prediction.get("_run_mode"),
        "is_cached_snapshot": prediction.get("_is_cached_snapshot"),
    }

    risk = abs(signal["entry_price"] - signal["pred_low"])
    reward = abs(signal["pred_close"] - signal["entry_price"])
    signal["risk_reward_ratio"] = round(reward / risk, 2) if risk > 0 else None

    return signal


def format_signal_text(signal: dict) -> str:
    verdict_cn = "值博" if signal["worth_trading"] else "不值博"
    calibrated_cn = {"LONG": "做多", "SHORT": "做空", "觀望": "觀望"}.get(
        signal["calibrated_signal"], signal["calibrated_signal"])
    stale_note = " ⚠️ 數據可能非即時" if signal["is_stale"] else ""
    lines = [
        f"=== HSI 次日交易訊號 — {signal['signal_date']} ===",
        f"市場 Regime: {signal['regime']}",
        f"原始方向: {signal['raw_direction']}  |  校準後訊號: {calibrated_cn}",
        f"P(上升): {signal['p_up']}  |  信心分數: {signal['confidence_score']}",
        f"值博與否: {verdict_cn}  ({signal['reason']})",
        f"建議倉位: {signal['position_size_pct']}%" if signal["position_size_pct"] is not None else "建議倉位: N/A",
        f"現價 (Last Close): {signal['last_close']}",
        f"建議入場位: {signal['entry_price']}",
        f"預測低位: {signal['pred_low']}",
        f"預測高位: {signal['pred_high']}",
        f"預測收市位: {signal['pred_close']}",
        f"Risk:Reward = 1:{signal['risk_reward_ratio']}",
        f"歷史命中率 (HSI): {signal['hit_rate']}",
        f"數據來源: {signal['data_source']}{stale_note}",
        f"執行模式: {signal['run_mode']}"
        + (" (讀取今日已凍結 snapshot)" if signal["is_cached_snapshot"] else ""),
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
    with open(LATEST_RESULT_PATH) as f:
        prediction = json.load(f)

    signal = generate_signal(prediction)

    # 只有「official 且非 cached snapshot」（即今日第一次正式 run）先寫入
    # signals_log.csv，邏輯同 predict.py 嘅 append_to_log() 把關一致，
    # 避免 preliminary run 或重複觸發 official workflow 時產生重複紀錄。
    if prediction.get("_run_mode") == "official" and not prediction.get("_is_cached_snapshot"):
        save_signal(signal)

    print(format_signal_text(signal))
    print("\nFull signal (JSON):")
    print(json.dumps(signal, indent=2, default=str))
