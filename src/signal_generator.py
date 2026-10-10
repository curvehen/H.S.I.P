"""
Signal Generator — formats predict.py output (read from LATEST_RESULT_PATH
for official runs, or PRED_DIR/latest_preliminary_result.json for
preliminary runs) into a clean, human-readable next-day trading signal:
entry / high / low / close / regime-calibrated direction / position size /
worth-trading verdict.

FIX LOG (accumulated):
  - Every field read realigned to predict.py's ACTUAL result dict keys
    (confidence, position_pct, pred_low_price/pred_high_price/pred_close_price,
    run_mode — no underscore). Previous schema (signal_confidence,
    entry_price via direct indexing, pred_low/pred_high/pred_close,
    worth_trading_reason, _run_mode, calibrated_signal) did not exist and
    six fields were accessed via direct indexing, guaranteed to raise
    KeyError.
  - calibrated_signal (LONG/SHORT/觀望) computed HERE since predict.py only
    emits a boolean worth_trading, not a three-way label.
  - worth_trading_reason reconstructed HERE from threshold_used +
    expected_move_pct + confidence, since predict.py does not pre-build a
    reason string.
  - entry_price NOW uses predict.py's real estimated_entry_price field
    (wired via gap_estimator.py + macro_features.py's overnight-gap chain),
    with gap_model_status surfaced so the reader knows whether the gap
    model is calibrated ("FITTED") or running on a fallback/default beta
    ("DEFAULT_FALLBACK" / "NOT_CALIBRATED" / "ERROR"). Falls back to
    last_close only if estimated_entry_price is None (gap chain
    unavailable or failed).
  - hit_rate (HSI-level) fetched directly via hit_rate_tracker.get_hit_rate(),
    since predict_hsi() does not attach this field itself.
  - Preliminary runs now get the mandatory "初步方向,非交易信號" banner,
    rendered first and last in the formatted text, and are excluded from
    log_signal() entirely (no write to SIGNAL_LOG_PATH).
"""

import json
import os
import pandas as pd

from config import SIGNAL_LOG_PATH, LATEST_RESULT_PATH, PRED_DIR, HSI_TICKER
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
                f"{min_conf:.2f}(門檻來源: {source})")
    reasons = []
    if move < min_move:
        reasons.append(f"預期波幅 {move:.3%} 低於門檻 {min_move:.3%}")
    if conf < min_conf:
        reasons.append(f"信心 {conf:.2f} 低於門檻 {min_conf:.2f}")
    return "、".join(reasons) + f"(門檻來源: {source})"


def _resolve_entry_price(prediction: dict):
    """Returns (entry_price, note). Uses predict.py's gap-model-based
    estimated_entry_price when available; falls back to last_close with an
    explicit note when the gap chain was unavailable or failed."""
    estimated = prediction.get("estimated_entry_price")
    status = prediction.get("gap_model_status", "UNAVAILABLE")
    last_close = prediction.get("last_close")

    if estimated is not None:
        if status == "FITTED":
            return estimated, ""
        return estimated, f"(開盤gap模型狀態: {status},估算僅供參考)"
    return last_close, "(開盤gap估算不可用,暫以現價替代)"


def generate_signal(prediction: dict) -> dict:
    raw_direction = "LONG" if prediction["pred_close_return_blended"] > 0 else "SHORT"
    calibrated_signal = _determine_calibrated_signal(prediction, raw_direction)

    hit_rate = prediction.get("hit_rate")
    if hit_rate is None:
        try:
            hit_rate = get_hit_rate(HSI_TICKER)
        except Exception:
            hit_rate = "N/A"

    entry_price, entry_price_note = _resolve_entry_price(prediction)

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
        "entry_price": round(entry_price, 1) if entry_price is not None else None,
        "entry_price_note": entry_price_note,
        "gap_model_status": prediction.get("gap_model_status", "UNAVAILABLE"),
        "pred_low": round(prediction["pred_low_price"], 1),
        "pred_high": round(prediction["pred_high_price"], 1),
        "pred_close": round(prediction["pred_close_price"], 1),
        "hit_rate": hit_rate,
        "llm_commentary": prediction.get("llm_commentary"),
    }

    if signal["entry_price"] is not None:
        risk = abs(signal["entry_price"] - signal["pred_low"])
        reward = abs(signal["pred_close"] - signal["entry_price"])
        signal["risk_reward_ratio"] = round(reward / risk, 2) if risk > 0 else None
    else:
        signal["risk_reward_ratio"] = None

    return signal


def format_signal_text(signal: dict) -> str:
    is_preliminary = signal.get("run_mode") == "preliminary"
    verdict_cn = "值博" if signal["worth_trading"] else "不值博"
    calibrated_cn = {"LONG": "做多", "SHORT": "做空", "觀望": "觀望"}.get(
        signal["calibrated_signal"], signal["calibrated_signal"])

    lines = []

    if is_preliminary:
        lines.append("⚠️⚠️⚠️ 【初步方向,非交易信號】此為盤前初步參考,非正式交易訊號,"
                     "正式訊號請以凌晨4時官方版本為準 ⚠️⚠️⚠️")
        lines.append("")

    lines.append(f"=== 恆生指數次日交易訊號 ({signal.get('signal_date', 'N/A')}) ===")
    lines.append(f"方向判斷: {calibrated_cn}(原始方向: {signal['raw_direction']}, Regime: {signal.get('regime', 'N/A')})")
    lines.append(f"上升機率 P(up): {signal['p_up']:.2%}" if signal.get("p_up") is not None else "上升機率: N/A")
    lines.append(f"模型信心分數: {signal['confidence_score']}" if signal.get("confidence_score") is not None else "模型信心分數: N/A")
    lines.append(f"是否值博: {verdict_cn}")
    lines.append(f"判斷原因: {signal.get('reason', 'N/A')}")
    lines.append("")
    lines.append(f"現價: {signal['last_close']}")
    lines.append(f"預估入場價: {signal['entry_price']}{signal.get('entry_price_note', '')}")
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
    auditing. Preliminary signals are explicitly excluded — must not
    pollute the official historical signal record."""
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
        "gap_model_status": signal["gap_model_status"],
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
    RUN_MODE: official reads LATEST_RESULT_PATH; preliminary reads
    PRED_DIR/latest_preliminary_result.json (written by predict.py's
    log_prediction() for non-official runs)."""
    run_mode = os.environ.get("RUN_MODE", "official")

    if run_mode == "official":
        result_path = LATEST_RESULT_PATH
    else:
        result_path = PRED_DIR / "latest_preliminary_result.json"

    if not result_path.exists():
        raise FileNotFoundError(
            f"signal_generator: expected result file not found: {result_path}. "
            f"Ensure predict.py has run with RUN_MODE='{run_mode}' first."
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
