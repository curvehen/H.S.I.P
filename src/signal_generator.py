"""
Signal Generator — formats predict.py output (read from LATEST_RESULT_PATH,
the single-source-of-truth JSON for the current run) into a clean,
human-readable next-day trading signal: entry / high / low / close /
regime-calibrated direction / position size / worth-trading verdict.

NOTE: Does NOT re-read predictions_log.csv's last row. Under the dual-run
architecture (official vs preliminary), predictions_log.csv is only appended
on official runs — reading its tail would return a stale prior day's
prediction whenever this script runs under preliminary mode. Instead, every
invocation of predict.py (official or preliminary) freshly overwrites
LATEST_RESULT_PATH when run_mode == "official" ONLY (see predict.py's
log_prediction()). For preliminary runs, this script reads from a SEPARATE
preliminary result file to avoid clobbering the official JSON — see
PRELIMINARY_RESULT_PATH below (pending confirmation/implementation in
predict.py; see FIX LOG).

FIX LOG (this revision):
  - Realigned every field read to predict.py's ACTUAL result dict keys
    (confirmed by reading predict.py's source directly). The previous
    version used a different/older schema (signal_confidence, entry_price,
    pred_low/pred_high/pred_close, worth_trading_reason, _run_mode,
    target_trading_date, calibrated_signal, data_source, is_stale,
    _is_cached_snapshot) — none of which exist under those exact names in
    the current predict.py, and six of them were accessed via direct
    indexing (prediction["..."]), which would raise KeyError and crash this
    script on every single run.
  - calibrated_signal (LONG / SHORT / 觀望) is now computed HERE rather than
    assumed to be supplied by predict.py, since predict.py currently only
    emits a boolean `worth_trading`, not a three-way calibrated label.
  - entry_price is not yet produced by predict.py (no gap_estimator.py
    integration wired into the pipeline yet). As an explicit, flagged
    placeholder, entry_price defaults to last_close. Replace with a true
    next-open estimate once gap_estimator.py is wired into predict.py.
  - hit_rate is now fetched directly via hit_rate_tracker.get_hit_rate(),
    since predict.py's predict_hsi() does not currently compute/attach a
    hit_rate field for the HSI itself (only predict_single_stock() does,
    per-ticker).
  - run_mode key aligned to predict.py's actual key name "run_mode" (no
    underscore prefix). NOTE: email_report.py was previously found to read
    "_run_mode" (underscore) — this mismatch must be fixed when
    email_report.py is next revised, so both scripts agree on one key name.
  - Added the explicit required warning text "⚠️ 初步方向,非交易信號" for
    preliminary runs, per the dual run-time requirement — this text was
    completely absent in the previous version.
  - data_source / is_stale / is_cached_snapshot are not yet surfaced by
    predict.py at all; left as .get()-based soft lookups that degrade to
    None/"N/A" display rather than crashing, pending predict.py exposing
    them (data_sources.py already tags DataFrames with this metadata
    upstream — it is just not yet threaded through predict.py's result
    dict).
"""

import json
import pandas as pd
from config import SIGNAL_LOG_PATH, LATEST_RESULT_PATH, HSI_TICKER
from hit_rate_tracker import get_hit_rate


def _determine_calibrated_signal(prediction: dict, raw_direction: str) -> str:
    """ENH#1: regime-calibrated three-way label. predict.py only emits a
    boolean `worth_trading` verdict (post regime-threshold filtering), so
    the LONG/SHORT/觀望 label is derived here: if the verdict says the move
    doesn't clear the regime-specific threshold, the signal is "觀望"
    (sit out) regardless of raw direction; otherwise it follows raw_direction."""
    if not prediction.get("worth_trading", False):
        return "觀望"
    return raw_direction


def _build_worth_trading_reason(prediction: dict) -> str:
    """predict.py does not emit a pre-built reason string — only the
    `threshold_used` dict (MIN_EXPECTED_MOVE_PCT, MIN_CONFIDENCE, source)
    and the boolean verdict. Reconstructs a human-readable reason from
    those raw numbers so the signal text stays informative."""
    thresh = prediction.get("threshold_used") or {}
    move = prediction.get("expected_move_pct")
    conf = prediction.get("confidence")
    min_move = thresh.get("MIN_EXPECTED_MOVE_PCT")
    min_conf = thresh.get("MIN_CONFIDENCE")
    source = thresh.get("source", "unknown")
    if move is None or conf is None or min_move is None or min_conf is None:
        return "門檻資料不完整,無法生成詳細原因"
    if prediction.get("worth_trading"):
        return (f"預期波幅 {move:.3%} ≥ 門檻 {min_move:.3%} 且信心 {conf:.2f} ≥ 門檻 "
                f"{min_conf:.2f}(門檻來源: {source}")+")"
    reasons = []
    if move < min_move:
        reasons.append(f"預期波幅 {move:.3%} 低於門檻 {min_move:.3%}")
    if conf < min_conf:
        reasons.append(f"信心 {conf:.2f} 低於門檻 {min_conf:.2f}")
    return "、".join(reasons) + f"(門檻來源: {source})"


def generate_signal(prediction: dict) -> dict:
    # raw_direction: 純粹基於 blended return 正負號嘅方向(未經 regime 門檻過濾)。
    raw_direction = "LONG" if prediction["pred_close_return_blended"] > 0 else "SHORT"
    calibrated_signal = _determine_calibrated_signal(prediction, raw_direction)

    hit_rate = prediction.get("hit_rate")
    if hit_rate is None:
        try:
            hit_rate = get_hit_rate(HSI_TICKER)
        except Exception:
            hit_rate = "N/A"

    # entry_price placeholder: predict.py does not yet produce a true
    # next-open estimate (gap_estimator.py is not wired into the pipeline).
    # Falls back to last_close until that integration is confirmed.
    entry_price = prediction.get("entry_price", prediction["last_close"])

    signal = {
        "signal_date": prediction.get("date"),
        "run_mode": prediction.get("run_mode"),
        "raw_direction": raw_direction,
        "calibrated_signal": calibrated_signal,
        "regime": prediction.get("regime"),
        "p_up": prediction.get("p_up"),
        "confidence_score": round(prediction["confidence"], 4) if prediction.get("confidence") is not None else None,
        "position_size_pct": prediction.get("position_pct"),
        "worth_trading": prediction["worth_trading"],
        "reason": _build_worth_trading_reason(prediction),
        "last_close": round(prediction["last_close"], 1),
        "entry_price": round(entry_price, 1),
        "pred_low": round(prediction["pred_low_price"], 1),
        "pred_high": round(prediction["pred_high_price"], 1),
        "pred_close": round(prediction["pred_close_price"], 1),
        "hit_rate": hit_rate,
        "data_source": prediction.get("data_source"),
        "is_stale": prediction.get("is_stale"),
        "is_cached_snapshot": prediction.get("is_cached_snapshot"),
        "llm_commentary": prediction.get("llm_commentary"),
    }

    risk = abs(signal["entry_price"] - signal["pred_low"])
    reward = abs(signal["pred_close"] - signal["entry_price"])
    signal["risk_reward_ratio"] = round(reward / risk, 2) if risk > 0 else None

    return signal


def format_signal_text(signal: dict) -> str:
    is_preliminary = signal.get("run_mode") == "preliminary"
    verdict_cn = "值博" if signal["worth_trading"] else "不值博"
    calibrated_cn = {"LONG": "做多", "SHORT": "做空", "觀望": "觀望"}.get(
        signal["calibrated_signal"], signal["calibrated_signal"])
    stale_note = " ⚠️ 數據可能非即時" if signal.get("is_stale") else ""

    lines = []

    # Required warning banner for preliminary (7pm HKT) runs — must NOT be
    # ambiguous or buried: placed as the very first and very last line.
    if is_preliminary:
        lines.append("⚠️⚠️⚠️ 【初步方向,非交易信號】此為盤前初步參考,非正式交易訊號,正式訊號請以凌晨4時官方版本為準 ⚠️⚠️⚠️")
        lines.append("")

    lines.append(f"=== 恆生指數次日交易訊號 ({signal.get('signal_date', 'N/A')}) ==={stale_note}")
    lines.append(f"方向判斷: {calibrated_cn}(原始方向: {signal['raw_direction']}, Regime: {signal.get('regime', 'N/A')})")
    lines.append(f"上升機率 P(up): {signal['p_up']:.2%}" if signal.get("p_up") is not None else "上升機率: N/A")
    lines.append(f"模型信心分數: {signal['confidence_score']}" if signal.get("confidence_score") is not None else "模型信心分數: N/A")
    lines.append(f"是否值博: {verdict_cn}")
    lines.append(f"判斷原因: {signal.get('reason', 'N/A')}")
    lines.append("")
    lines.append(f"現價: {signal['last_close']}")
    lines.append(f"預估入場價: {signal['entry_price']}" + ("(暫以現價替代,待開盤gap估算模組接入)" if signal.get("entry_price") == signal.get("last_close") else ""))
    lines.append(f"預測低位: {signal['pred_low']}")
    lines.append(f"預測高位: {signal['pred_high']}")
    lines.append(f"預測收市: {signal['pred_close']}")
    if signal.get("risk_reward_ratio") is not None:
        lines.append(f"風險回報比: {signal['risk_reward_ratio']}")
    lines.append("")
    if signal["worth_trading"] and not is_preliminary:
        pos_pct = signal.get("position_size_pct")
        lines.append(f"建議倉位(Fractional Kelly): {pos_pct:.2%}" if pos_pct is not None else "建議倉位: N/A")
    elif is_preliminary:
        lines.append("建議倉位: 不適用(初步訊號不提供倉位建議)")
    else:
        lines.append("建議倉位: 0%(不建議交易)")
    lines.append("")
    lines.append(f"歷史命中率: {signal.get('hit_rate', 'N/A')}")
    if signal.get("llm_commentary"):
        lines.append("")
        lines.append(f"AI市場評論: {signal['llm_commentary']}")

    if is_preliminary:
        lines.append("")
        lines.append("⚠️ 再次提醒:以上為初步方向,非交易信號,請勿據此執行交易。")

    return "\n".join(lines)


def log_signal(signal: dict):
    """Appends the official signal to SIGNAL_LOG_PATH for historical
    auditing. Preliminary signals are explicitly excluded from this log per
    the dual run-time requirement (preliminary runs must not persist to any
    official record)."""
    if signal.get("run_mode") != "official":
        print(f"signal_generator: run_mode='{signal.get('run_mode')}' (preliminary) — "
              f"skipping signal log write.")
        return

    row = {
        "date": signal["signal_date"],
        "calibrated_signal": signal["calibrated_signal"],
        "regime": signal["regime"],
        "p_up": signal["p_up"],
        "confidence_score": signal["confidence_score"],
        "worth_trading": signal["worth_trading"],
        "position_size_pct": signal["position_size_pct"],
        "entry_price": signal["entry_price"],
        "pred_low": signal["pred_low"],
        "pred_high": signal["pred_high"],
        "pred_close": signal["pred_close"],
        "risk_reward_ratio": signal["risk_reward_ratio"],
        "hit_rate": signal["hit_rate"],
    }
    df_row = pd.DataFrame([row])
    if SIGNAL_LOG_PATH.exists():
        df_row.to_csv(SIGNAL_LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(SIGNAL_LOG_PATH, mode="w", header=True, index=False)


def run():
    """Entry point. Reads whichever result file matches the current
    RUN_MODE: official runs read LATEST_RESULT_PATH (written by predict.py's
    log_prediction()); preliminary runs are expected to read a separate
    preliminary-only result file, since predict.py's log_prediction()
    currently skips ALL file writes for preliminary mode entirely (see
    FIX LOG — this is a known open gap, not yet resolved in predict.py)."""
    import os
    run_mode = os.environ.get("RUN_MODE", "official")

    if run_mode == "official":
        result_path = LATEST_RESULT_PATH
    else:
        # PENDING: predict.py currently does not write ANY file on
        # preliminary runs (log_prediction() returns early). This path will
        # raise FileNotFoundError until predict.py is updated to persist a
        # preliminary-only snapshot (e.g. PRED_DIR / "latest_preliminary_result.json").
        from config import PRED_DIR
        result_path = PRED_DIR / "latest_preliminary_result.json"

    if not result_path.exists():
        raise FileNotFoundError(
            f"signal_generator: expected result file not found: {result_path}. "
            f"For preliminary runs, this means predict.py has not yet been updated "
            f"to persist a preliminary-only result snapshot (see FIX LOG)."
        )

    with open(result_path) as f:
        prediction = json.load(f)

    signal = generate_signal(prediction)
    text = format_signal_text(signal)
    print(text)

    log_signal(signal)

    return signal, text


if __name__ == "__main__":
    run()

