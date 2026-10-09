"""
INFERENCE SCRIPT — run daily by GitHub Actions. CPU-only, lightweight.
1. Loads pre-trained HSI models (Ensemble + quantile + high/low).
2. Generates direct HSI-level next-day prediction.
3. Generates bottom-up prediction (weighted aggregation of stock close predictions).
4. Blends direct + bottom-up.
5. Applies meta-confidence filter.
6. Applies regime-calibrated long/short threshold (ENH#1) to determine signal direction.
7. Computes Fractional Kelly position size (ENH#3), adjusted by regime + rolling accuracy.
8. Outputs entry/high/low/close + worth-trading verdict + position size.
9. Appends everything to the prediction log.

Generates: P(up)/P(down), signal strength, regime, HSI high/low/close,
range stats, position sizing, and full per-stock prediction table with hit rates.
"""

import json
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import os

from config import (MODEL_CLOSE_Q10_PATH, MODEL_CLOSE_Q50_PATH, MODEL_CLOSE_Q90_PATH,
                     MODEL_HIGH_PATH, MODEL_LOW_PATH, FEATURE_LIST_PATH, PRED_LOG_PATH,
                     STOCK_MODEL_DIR, HSI_TICKER, US_FUTURES_TICKER, VIX_TICKER,
                     MIN_EXPECTED_MOVE_PCT, MIN_CONFIDENCE, HSI_PROB_MODEL_PATH,
                     stock_prob_model_path, stock_high_model_path, stock_low_model_path, 
                     SNAPSHOT_DIR, LATEST_RESULT_PATH, LLM_ENABLED_RUN_MODES)
from data_sources import fetch_with_fallback
from features import build_features
from labeling import LABEL_COLUMNS
from confidence import load_meta_model, get_signal_confidence
from news_sentiment import get_daily_market_sentiment, get_stock_sentiment
from stock_universe import get_universe, get_all_tracked_tickers, get_stock_info, is_hsi_constituent
from ensemble_model import HSIEnsembleModel
from regime import detect_regime
from probability_model import load_probability_model, predict_probability_up, classify_signal_strength
from llm_analysis import generate_market_commentary
from hit_rate_tracker import get_hit_rate
from ccass_scraper import get_ccass_change
from data_sources import to_stooq_hk_code
from market_hours import get_latest_usable_row, get_next_trading_day
from threshold_calibrator import RegimeThresholdCalibrator
from position_sizer import PositionSizer
# from gap_estimator import load_gap_model, estimate_next_open_price
# from macro_features import get_market_overnight_return  # 改名修正
# from config import MODEL_OPEN_HIGH_PATH, MODEL_OPEN_LOW_PATH





def align_features(latest_row: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    for col in feature_cols:
        if col not in latest_row.columns:
            latest_row[col] = 0
    return latest_row[feature_cols].astype(float)


def _parse_hit_rate_pct(hit_rate_str: str) -> float:
    """Converts a hit_rate display string like '65.8%' into a 0-1 float.
    Falls back to a neutral 0.5 if unavailable ('N/A')."""
    if not hit_rate_str or hit_rate_str == "N/A":
        return 0.5
    try:
        return float(hit_rate_str.strip("%")) / 100
    except (ValueError, AttributeError):
        return 0.5


def predict_hsi():
    us_futures = fetch_with_fallback(US_FUTURES_TICKER)
    vix = fetch_with_fallback(VIX_TICKER)
    market_sentiment = get_daily_market_sentiment()

    raw = fetch_with_fallback(HSI_TICKER, stooq_ticker="^hsi")
    raw, session = get_latest_usable_row(raw)
    feat_df = build_features(raw, us_futures=us_futures, vix=vix,
                              ccass_change=0.0, market_sentiment=market_sentiment,
                              stock_sentiment=0.0, ticker=HSI_TICKER, include_macro=True)

    m_close_q10 = lgb.Booster(model_file=str(MODEL_CLOSE_Q10_PATH))
    m_close_q90 = lgb.Booster(model_file=str(MODEL_CLOSE_Q90_PATH))
    m_high = lgb.Booster(model_file=str(MODEL_HIGH_PATH))
    m_low  = lgb.Booster(model_file=str(MODEL_LOW_PATH))

    ensemble = HSIEnsembleModel.load(prefix="hsi")
    with open(FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)

    latest_row = feat_df.iloc[[-1]].copy()
    X_latest = align_features(latest_row, feature_cols)

    pred_close_q10 = float(m_close_q10.predict(X_latest)[0])
    pred_close_mid = float(ensemble.predict_loaded(X_latest)[0])
    pred_close_q90 = float(m_close_q90.predict(X_latest)[0])
    pred_high_return = float(m_high.predict(X_latest)[0])
    pred_low_return = float(m_low.predict(X_latest)[0])

    meta_model = load_meta_model()
    confidence = get_signal_confidence(meta_model, X_latest)

    prob_clf = load_probability_model(HSI_PROB_MODEL_PATH)
    p_up = predict_probability_up(prob_clf, X_latest)
    strength = classify_signal_strength(p_up)

    regime = detect_regime(feat_df)
    last_close = float(latest_row["Close"].values[0])

    data_as_of_date = latest_row.index[0].date()
    target_trading_date = get_next_trading_day(data_as_of_date)

    return {
        "last_close": last_close,
        "pred_close_return_q10": pred_close_q10,
        "pred_close_return_mid": pred_close_mid,
        "pred_close_return_q90": pred_close_q90,
        "pred_high_return": pred_high_return,       # 已經係 open-based 還原後嘅數值
        "pred_low_return": pred_low_return,
        "p_up": p_up,
        "signal_strength_label": strength["label"],
        "signal_strength_margin": strength["margin_pct"],
        "regime": regime,
        "signal_confidence": confidence,
        "data_source": latest_row["source"].values[0],
        "is_stale": bool(latest_row["is_stale"].values[0]),
        "predict_date": str(latest_row.index[0].date()),
        "data_as_of_date": str(data_as_of_date),
        "target_trading_date": str(target_trading_date),
    }


def predict_hsi_bottom_up():
    universe = get_universe()
    weighted_return = 0.0
    total_weight_used = 0.0
    for ticker, weight in universe.items():
        model_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_close_q50.txt"
        feat_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_features.json"
        if not model_path.exists() or not feat_path.exists():
            continue
        try:
            stooq_code = to_stooq_hk_code(ticker)
            raw = fetch_with_fallback(ticker, stooq_ticker=stooq_code)
            raw, session = get_latest_usable_row(raw)
            keywords = [ticker.split(".")[0]]
            stock_sentiment = get_stock_sentiment(keywords)
            ccass_change = get_ccass_change(ticker.split(".")[0])

            feat_df = build_features(raw, ccass_change=ccass_change,
                                      stock_sentiment=stock_sentiment, ticker=ticker,
                                      include_macro=False)
            with open(feat_path) as f:
                feature_cols = json.load(f)
            latest_row = feat_df.iloc[[-1]].copy()
            X_latest = align_features(latest_row, feature_cols)

            model = lgb.Booster(model_file=str(model_path))
            pred_return = float(model.predict(X_latest)[0])

            weighted_return += pred_return * weight
            total_weight_used += weight
        except Exception as e:
            print(f"Bottom-up predict failed for {ticker}: {e}")
            continue

    if total_weight_used > 0:
        weighted_return /= total_weight_used
    return {"bottom_up_close_return": weighted_return, "coverage_weight": total_weight_used}


def predict_single_stock(ticker: str):
    """Full prediction for one stock: direction, P(up), strength, high/low/close, RSI, hit rate."""
    model_close_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_close_q50.txt"
    feat_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_features.json"
    prob_path = stock_prob_model_path(ticker)
    high_path = stock_high_model_path(ticker)
    low_path = stock_low_model_path(ticker)

    if not model_close_path.exists() or not feat_path.exists():
        return None

    try:
        # OLD version stooq_code = ticker.replace(".HK", "").zfill(5) + ".hk"
        stooq_code = to_stooq_hk_code(ticker)   # 直接用返已經 import 咗嘅共用函數
        raw = fetch_with_fallback(ticker, stooq_ticker=stooq_code)
        raw, session = get_latest_usable_row(raw)
        keywords = [ticker.split(".")[0]]
        stock_sentiment = get_stock_sentiment(keywords)
        ccass_change = get_ccass_change(ticker.split(".")[0])

        feat_df = build_features(raw, ccass_change=ccass_change,
                                  stock_sentiment=stock_sentiment, ticker=ticker,
                                  include_macro=False)
        with open(feat_path) as f:
            feature_cols = json.load(f)
        latest_row = feat_df.iloc[[-1]].copy()
        X_latest = align_features(latest_row, feature_cols)

        model_close = lgb.Booster(model_file=str(model_close_path))
        pred_close_return = float(model_close.predict(X_latest)[0])

        pred_high_return = 0.0
        if high_path.exists():
            model_high = lgb.Booster(model_file=str(high_path))
            pred_high_return = float(model_high.predict(X_latest)[0])

        pred_low_return = 0.0
        if low_path.exists():
            model_low = lgb.Booster(model_file=str(low_path))
            pred_low_return = float(model_low.predict(X_latest)[0])

        prob_clf = load_probability_model(prob_path)
        p_up = predict_probability_up(prob_clf, X_latest)
        strength = classify_signal_strength(p_up)

        last_close = float(latest_row["Close"].values[0])
        rsi = float(latest_row["RSI14"].values[0]) if "RSI14" in latest_row.columns else None
        info = get_stock_info(ticker)

        return {
            "ticker": ticker,
            "name": info.get("name", ticker),
            "sector": info.get("sector", "未知"),
            "signal": "LONG" if pred_close_return > 0 else "SHORT",
            "p_up": p_up,
            "strength_label": strength["label"],
            "last_close": last_close,
            "pred_high": last_close * (1 + pred_high_return),
            "pred_low": last_close * (1 + pred_low_return),
            "pred_close": last_close * (1 + pred_close_return),
            "pred_return_pct": pred_close_return * 100,
            "rsi": rsi,
            "hit_rate": get_hit_rate(ticker),
        }
    except Exception as e:
        print(f"Single-stock predict failed for {ticker}: {e}")
        return None


def predict_all_stocks():
    # universe = get_universe()
    # HSI 成份股 + 觀察名單（1211.HK, 968.HK 等），不影響 predict_hsi_bottom_up()
    tracked = get_all_tracked_tickers()
    results = []
    for ticker in tracked:
        r = predict_single_stock(ticker)
        if r is not None:
            r["is_constituent"] = is_hsi_constituent(ticker)
            results.append(r)
    return results


def is_worth_trading(predicted_close_return: float, confidence: float) -> dict:
    expected_move = abs(predicted_close_return)
    worth_it = expected_move >= MIN_EXPECTED_MOVE_PCT and confidence >= MIN_CONFIDENCE
    if not worth_it:
        reason = ("信號不明，建議觀望" if expected_move < MIN_EXPECTED_MOVE_PCT
                   else f"模型信心不足 ({confidence:.2f} < {MIN_CONFIDENCE})")
    else:
        reason = "預測幅度同信心度均達標"
    return {"worth_trading": worth_it, "reason": reason}


def _snapshot_path(target_trading_date: str):
    return SNAPSHOT_DIR / f"snapshot_{target_trading_date}.json"
  
def predict_today(run_mode: str = "preliminary"):
    """
    run_mode:
      - "official"    : 凌晨 4:00am HKT 排程專用。抓即時數據、產生 LLM 摘要、
                         寫入 snapshot，並准許寫入 predictions_log.csv。
      - "preliminary" : 晚上 7:00pm HKT 排程專用。若當日 snapshot 已存在
                         （即 official 已跑過），直接讀 snapshot 保證一致；
                         否則抓即時數據，但結果標記為非正式，不寫入任何 log，
                         亦不觸發 LLM（省成本）。
    """
    hsi = predict_hsi()
    target_date = hsi["target_trading_date"]
    snap_path = _snapshot_path(target_date)

    # 若 official run 已經為呢個 target_trading_date 產生過 snapshot，
    # 一律讀返 snapshot，確保同一交易日任何時間 re-run 都得出完全相同結果。
    if snap_path.exists():
        with open(snap_path) as f:
            cached = json.load(f)
        cached["_is_cached_snapshot"] = True
        cached["_run_mode"] = run_mode
        return cached
      

    bottom_up = predict_hsi_bottom_up()

    has_good_coverage = bottom_up["coverage_weight"] > 0.3
    blended_return = ((hsi["pred_close_return_mid"] + bottom_up["bottom_up_close_return"]) / 2
                       if has_good_coverage else hsi["pred_close_return_mid"])

    last_close = hsi["last_close"]
    predicted_close = last_close * (1 + blended_return)
    predicted_high = last_close * (1 + hsi["pred_high_return"])
    predicted_low = last_close * (1 + hsi["pred_low_return"])
    pred_range_points = predicted_high - predicted_low

    verdict = is_worth_trading(blended_return, hsi["signal_confidence"])
    stock_predictions = predict_all_stocks()

    # --- ENH#1: Regime-calibrated long/short threshold ---
    calibrator = RegimeThresholdCalibrator.load()
    calibrated_signal = calibrator.get_signal(prob=hsi["p_up"], regime=hsi["regime"])
    # calibrated_signal: 1 (long) / -1 (short) / 0 (觀望，落在兩個門檻之間)
    calibrated_signal_label = {1: "LONG", -1: "SHORT", 0: "觀望"}[calibrated_signal]
    regime_thresholds_used = calibrator.thresholds.get(hsi["regime"], calibrator.thresholds.get("NEUTRAL"))

    # --- ENH#3: Fractional Kelly position sizing ---
    sizer = PositionSizer(kelly_fraction=0.25, max_position=1.0, min_edge=0.02)
    roll_acc = _parse_hit_rate_pct(get_hit_rate("HSI"))
    position_size_pct = sizer.compute_size(
        prob=hsi["p_up"],
        signal=calibrated_signal if calibrated_signal != 0 else int(np.sign(blended_return)),
        regime=hsi["regime"],
        roll_acc=roll_acc,
    )

    result = {
        "predict_date": hsi["predict_date"],
        "data_as_of_date": hsi["data_as_of_date"],
        "target_trading_date": hsi["target_trading_date"],
        "run_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "last_close": last_close,
        "entry_price": last_close,
        "pred_close_return_blended": blended_return,
        "hit_rate": get_hit_rate("HSI"),
        "p_up": hsi["p_up"],
        "p_down": 1 - hsi["p_up"],
        "signal_strength_label": hsi["signal_strength_label"],
        "signal_strength_margin": hsi["signal_strength_margin"],
        "regime": hsi["regime"],
        "calibrated_signal": calibrated_signal_label,
        "calibrated_signal_thresholds": regime_thresholds_used,
        "position_size_pct": round(position_size_pct * 100, 1),
        "pred_high": predicted_high,
        "pred_high_pct": (predicted_high - last_close) / last_close * 100,
        "pred_low": predicted_low,
        "pred_low_pct": (predicted_low - last_close) / last_close * 100,
        "pred_close": predicted_close,
        "pred_range_points": pred_range_points,
        "pred_range_pct": pred_range_points / last_close * 100,
        "signal_confidence": hsi["signal_confidence"],
        "worth_trading": verdict["worth_trading"],
        "worth_trading_reason": verdict["reason"],
        "bottom_up_coverage_weight": bottom_up["coverage_weight"],
        "data_source": hsi["data_source"],
        "is_stale": hsi["is_stale"],
        "actual_close": None,
        "directional_hit": None,
        "_stock_predictions": stock_predictions,
        "_run_mode": run_mode,
        "_is_cached_snapshot": False,
    }

    # LLM 綜合分析：只喺 official run 先觸發，preliminary 唔叫 API（省成本），
    # 且任何對 DashScope 嘅呼叫失敗都已經喺 llm_analysis.py 內部 fail-safe 處理。
    if run_mode in LLM_ENABLED_RUN_MODES:
        result["llm_commentary"] = generate_market_commentary(result)
    else:
        result["llm_commentary"] = None

    if run_mode == "official":
        with open(snap_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

    return result


def append_to_log(result: dict):
    # calibrated_signal_thresholds is a nested dict — exclude it from the flat
    # CSV log row (same underscore convention as _stock_predictions), but keep
    # it in the JSON output / email report via the full result dict.
    # calibrated_signal_thresholds (nested dict) 同 llm_commentary（長文字）
    # 都唔適合落 CSV flat row，一律排除，JSON/email 層面仍會用完整 result。
  
    flat_result = {k: v for k, v in result.items()
                    if not k.startswith("_") and k not in ("calibrated_signal_thresholds", "llm_commentary")}
                  }
    row = pd.DataFrame([flat_result])
    if PRED_LOG_PATH.exists():
        log = pd.read_csv(PRED_LOG_PATH)
        log = pd.concat([log, row], ignore_index=True)
    else:
        log = row
    log.to_csv(PRED_LOG_PATH, index=False)


if __name__ == "__main__":
    # result = predict_today()
    # append_to_log(result)
    
    run_mode = os.environ.get("RUN_MODE", "preliminary")
    result = predict_today(run_mode=run_mode)

    # 只有「official 且非 cached snapshot」（即今日第一次正式 run）先寫入
    # predictions_log.csv，避免同一交易日 official 被重複觸發時產生重複紀錄。
    if run_mode == "official" and not result.get("_is_cached_snapshot"):
        append_to_log(result)

    # 無論 preliminary 定 official，都將完整結果落盤，
    # 畀同一次 workflow 入面嘅 signal_generator.py / email_report.py
    # 直接讀用，杜絕重複 call predict_today() 攞到唔同即時數據嘅問題。
    with open(LATEST_RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)





    print(json.dumps({k: v for k, v in result.items() if k != "_stock_predictions"},
                      indent=2, default=str))
    print(f"\n個股預測數量: {len(result['_stock_predictions'])}")
