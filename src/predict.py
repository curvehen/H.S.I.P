"""
INFERENCE SCRIPT — run daily by GitHub Actions. CPU-only, lightweight.

[... unchanged docstring intro from prior revisions ...]

FIX LOG (this revision):
  - FIXED (previously tracked as open issue): compute_position_size() call
    corrected to match position_sizer.py's ACTUAL signature:
    compute_position_size(p_up, calibrated_signal, signal_confidence,
    risk_reward_ratio, regime, worth_trading) -> dict with keys
    position_size_pct, direction, p_win, raw_kelly_fraction,
    fractional_kelly, confidence_scaled_kelly, regime_dampener, capped,
    reason. Previous call used nonexistent kwargs (confidence=,
    expected_move_pct=) and silently returned 0% every single call via the
    except branch — Fractional Kelly sizing had NEVER actually executed.
  - To supply calibrated_signal + risk_reward_ratio (which the sizer
    requires but nothing upstream computed), this module now derives:
      raw_direction        = "LONG" if blended_return > 0 else "SHORT"
      calibrated_signal    = raw_direction if worth_trading else "觀望"
      risk_reward_ratio    = direction-aware reward/risk using
                              estimated_entry_price (or last_close fallback)
                              against pred_high_price/pred_low_price.
    These are now stored directly in `result` (calibrated_signal,
    raw_direction, risk_reward_ratio) as the SINGLE source of truth, so
    signal_generator.py / email_report.py should read them from `result`
    rather than re-deriving their own copies (avoids the two reports ever
    disagreeing on direction).
  - FIXED: `today = datetime.date.today()` used the GitHub Actions runner's
    UTC date, which is one calendar day behind Hong Kong time during the
    04:00 HKT official run (HKT 04:00 = UTC 20:00 the PREVIOUS day). Now
    uses `_dt.now(HKT).date()` throughout, so result["date"], the CSV log
    row, and the feature-snapshot filename all use the correct HKT trading
    date regardless of which UTC day the runner's clock reads.
  - result dict key renamed: position_pct -> position_size_pct (matches
    position_sizer.py's actual output key exactly, no translation layer).
    kelly_raw -> raw_kelly_fraction. position_reason now sources the
    sizer's own `reason` field (full diagnostic breakdown string) rather
    than a short ad-hoc string.
"""

import json
import datetime
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from datetime import datetime as _dt

from config import (MODEL_CLOSE_Q10_PATH, MODEL_CLOSE_Q50_PATH, MODEL_CLOSE_Q90_PATH,
                     MODEL_HIGH_PATH, MODEL_LOW_PATH, FEATURE_LIST_PATH, PRED_LOG_PATH,
                     STOCK_MODEL_DIR, HSI_TICKER, US_FUTURES_TICKER, VIX_TICKER,
                     MIN_EXPECTED_MOVE_PCT, MIN_CONFIDENCE, HSI_PROB_MODEL_PATH,
                     stock_prob_model_path, stock_high_model_path, stock_low_model_path)
from data_sources import fetch_with_fallback, to_stooq_hk_code
from features import build_features
from labeling import LABEL_COLUMNS
from confidence import load_meta_model, get_signal_confidence
from news_sentiment import get_daily_market_sentiment, get_stock_sentiment
from stock_universe import get_universe, get_stock_info, get_watchlist, get_watchlist_info
from ensemble_model import HSIEnsembleModel
from regime import detect_regime
from probability_model import load_probability_model, predict_probability_up, classify_signal_strength
from hit_rate_tracker import get_hit_rate
from ccass_scraper import get_ccass_change
from market_hours import get_latest_usable_row, get_next_trading_day, HKT

try:
    from macro_features import get_market_overnight_return
    from gap_estimator import load_gap_model, estimate_next_open_price
    _GAP_ESTIMATOR_AVAILABLE = True
except ImportError:
    _GAP_ESTIMATOR_AVAILABLE = False
    print("predict: gap_estimator/macro_features not available — "
          "estimated_entry_price will be omitted (consumers fall back to last_close).")

try:
    from threshold_calibrator import get_regime_threshold
    _REGIME_THRESHOLDS_AVAILABLE = True
except ImportError:
    _REGIME_THRESHOLDS_AVAILABLE = False
    print("predict: threshold_calibrator not available — using static global thresholds.")

try:
    from dynamic_ensemble_weighter import get_effective_weights
    _DYNAMIC_WEIGHTING_AVAILABLE = True
except ImportError:
    _DYNAMIC_WEIGHTING_AVAILABLE = False
    print("predict: dynamic_ensemble_weighter not available — using static ensemble weights.")

try:
    from position_sizer import compute_position_size
    _POSITION_SIZING_AVAILABLE = True
except ImportError:
    _POSITION_SIZING_AVAILABLE = False
    print("predict: position_sizer not available — position sizing will be omitted from output.")

try:
    from llm_analysis import generate_market_commentary, generate_stock_summaries_batch
    _LLM_ANALYSIS_AVAILABLE = True
except ImportError:
    _LLM_ANALYSIS_AVAILABLE = False
    print("predict: llm_analysis not available — no commentary will be generated.")


def determine_run_mode(now: _dt = None) -> str:
    env_override = os.environ.get("RUN_MODE")
    if env_override in ("preliminary", "official"):
        return env_override
    now = now or _dt.now(HKT)
    hour = now.hour
    if 18 <= hour < 21:
        return "preliminary"
    return "official"


def align_features(latest_row: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    for col in feature_cols:
        if col not in latest_row.columns:
            latest_row[col] = 0
    return latest_row[feature_cols].astype(float)


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
    m_low = lgb.Booster(model_file=str(MODEL_LOW_PATH))
    ensemble = HSIEnsembleModel.load(prefix="hsi")
    with open(FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)

    latest_row = feat_df.iloc[[-1]].copy()
    X_latest = align_features(latest_row, feature_cols)

    pred_close_q10 = float(m_close_q10.predict(X_latest)[0])
    pred_close_q90 = float(m_close_q90.predict(X_latest)[0])
    pred_high = float(m_high.predict(X_latest)[0])
    pred_low = float(m_low.predict(X_latest)[0])
  
    sub_preds = ensemble.predict_submodels_loaded(X_latest)
    static_weights = ensemble.weights  # 鍵名已經係 "lgb"/"rf"/"ridge",同 sub_preds 一致,冇需要再轉換
    static_weights = ensemble.weights

    if _DYNAMIC_WEIGHTING_AVAILABLE:
        try:
            effective_weights, weight_source = get_effective_weights(static_weights)
        except Exception as e:
            print(f"predict: dynamic weight lookup failed ({e}) — using static weights.")
            effective_weights, weight_source = static_weights, "static_fallback"
    else:
        effective_weights, weight_source = static_weights, "static_fallback"

    pred_close_q50 = float(sum(sub_preds[m] * effective_weights[m] for m in effective_weights))

    regime = detect_regime(feat_df.iloc[[-1]])

    prob_model = load_probability_model(HSI_PROB_MODEL_PATH)
    p_up = predict_probability_up(prob_model, X_latest)
    signal_strength = classify_signal_strength(p_up)

    meta_model = load_meta_model()
    confidence = get_signal_confidence(meta_model, X_latest)

    last_close = float(raw["Close"].iloc[-1])
    expected_move_pct = abs(pred_close_q50)

    estimated_entry_price = None
    gap_model_status = "UNAVAILABLE"
    if _GAP_ESTIMATOR_AVAILABLE:
        try:
            us_overnight_return = get_market_overnight_return(US_FUTURES_TICKER)
            gap_model = load_gap_model()
            estimated_entry_price = estimate_next_open_price(last_close, us_overnight_return, gap_model)
            gap_model_status = gap_model.get("status", "UNKNOWN")
        except Exception as e:
            print(f"predict: overnight gap estimation failed ({e}) — estimated_entry_price omitted.")
            estimated_entry_price = None
            gap_model_status = "ERROR"

    return {
        "ticker": HSI_TICKER,
        "session": session,
        "last_close": last_close,
        "pred_close_return": pred_close_q50,
        "pred_close_q10": pred_close_q10,
        "pred_close_q90": pred_close_q90,
        "pred_high_return": pred_high,
        "pred_low_return": pred_low,
        "regime": regime,
        "p_up": p_up,
        "signal_strength": signal_strength,
        "confidence": confidence,
        "expected_move_pct": expected_move_pct,
        "sub_model_preds": sub_preds,
        "ensemble_weights_used": effective_weights,
        "ensemble_weight_source": weight_source,
        "feature_snapshot_row": latest_row,
        "estimated_entry_price": estimated_entry_price,
        "gap_model_status": gap_model_status,
    }


def predict_single_stock(ticker: str):
    stooq_code = to_stooq_hk_code(ticker)
    raw = fetch_with_fallback(ticker, stooq_ticker=stooq_code)
    if raw is None or len(raw) < 50:
        return None
    raw, session = get_latest_usable_row(raw)

    ccass_change = get_ccass_change(ticker)
    stock_sentiment = get_stock_sentiment([ticker.split(".")[0]])
    market_sentiment = get_daily_market_sentiment()

    feat_df = build_features(raw, us_futures=None, vix=None,
                              ccass_change=ccass_change,
                              market_sentiment=market_sentiment,
                              stock_sentiment=stock_sentiment,
                              ticker=ticker, include_macro=False)

    feat_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_features.json"
    model_close_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_close_q50.txt"
    if not feat_path.exists() or not model_close_path.exists():
        return None

    with open(feat_path) as f:
        feature_cols = json.load(f)

    latest_row = feat_df.iloc[[-1]].copy()
    X_latest = align_features(latest_row, feature_cols)

    m_close = lgb.Booster(model_file=str(model_close_path))
    m_high = lgb.Booster(model_file=str(stock_high_model_path(ticker)))
    m_low = lgb.Booster(model_file=str(stock_low_model_path(ticker)))

    pred_close_return = float(m_close.predict(X_latest)[0])
    pred_high_return = float(m_high.predict(X_latest)[0])
    pred_low_return = float(m_low.predict(X_latest)[0])

    prob_path = stock_prob_model_path(ticker)
    p_up = 0.5
    signal_strength = "weak"
    if prob_path.exists():
        prob_model = load_probability_model(prob_path)
        p_up = predict_probability_up(prob_model, X_latest)
        signal_strength = classify_signal_strength(p_up)

    last_close = float(raw["Close"].iloc[-1])
    hit_rate = get_hit_rate(ticker)
    info = get_stock_info(ticker)

    return {
        "ticker": ticker,
        "name": info.get("name", ticker),
        "sector": info.get("sector", "未知"),
        "session": session,
        "last_close": last_close,
        "pred_close_return": pred_close_return,
        "pred_high_return": pred_high_return,
        "pred_low_return": pred_low_return,
        "p_up": p_up,
        "signal_strength": signal_strength,
        "hit_rate": hit_rate,
    }


def predict_bottom_up():
    universe = get_universe()
    weighted_return = 0.0
    stock_rows = []
    for ticker, weight in universe.items():
        result = predict_single_stock(ticker)
        if result is None:
            continue
        weighted_return += result["pred_close_return"] * weight
        result["weight"] = weight
        stock_rows.append(result)
    return weighted_return, stock_rows


def predict_watchlist():
    watchlist_rows = []
    for ticker in get_watchlist():
        result = predict_single_stock(ticker)
        if result is None:
            continue
        info = get_watchlist_info(ticker)
        result["name"] = info.get("name", ticker)
        result["sector"] = info.get("sector", "未知")
        watchlist_rows.append(result)
    return watchlist_rows


def get_effective_threshold(regime: str) -> dict:
    if _REGIME_THRESHOLDS_AVAILABLE:
        try:
            thresh = get_regime_threshold(regime)
            if thresh and "MIN_EXPECTED_MOVE_PCT" in thresh:
                return thresh
        except Exception as e:
            print(f"predict: regime threshold lookup failed ({e}) — using static global thresholds.")
    return {
        "MIN_EXPECTED_MOVE_PCT": MIN_EXPECTED_MOVE_PCT,
        "MIN_CONFIDENCE": MIN_CONFIDENCE,
        "source": "static_fallback",
    }


def compute_verdict(expected_move_pct: float, confidence: float, regime: str) -> dict:
    thresh = get_effective_threshold(regime)
    worth_trading = (expected_move_pct >= thresh["MIN_EXPECTED_MOVE_PCT"]) and \
                     (confidence >= thresh["MIN_CONFIDENCE"])
    return {
        "worth_trading": bool(worth_trading),
        "threshold_used": thresh,
    }


def _compute_risk_reward_ratio(entry_price: float, pred_high_price: float,
                                pred_low_price: float, raw_direction: str) -> float:
    """Direction-aware reward/risk. LONG: reward = upside to pred_high,
    risk = downside to pred_low. SHORT: reward = downside to pred_low,
    risk = upside to pred_high. Returns None if risk leg is zero/invalid."""
    if entry_price is None:
        return None
    if raw_direction == "LONG":
        reward = pred_high_price - entry_price
        risk = entry_price - pred_low_price
    else:
        reward = entry_price - pred_low_price
        risk = pred_high_price - entry_price
    if risk is None or risk <= 0:
        return None
    return round(reward / risk, 2)


def run_daily_prediction(run_mode: str = None):
    run_mode = run_mode or determine_run_mode()
    today = _dt.now(HKT).date()
    print(f"=== Running daily prediction for {today.isoformat()} (run_mode={run_mode}) ===")

    print("[1/7] HSI direct prediction...")
    hsi_direct = predict_hsi()

    print("[2/7] HSI bottom-up prediction (constituents only)...")
    bottom_up_return, stock_rows = predict_bottom_up()

    print("[3/7] Blending direct + bottom-up...")
    blended_return = 0.6 * hsi_direct["pred_close_return"] + 0.4 * bottom_up_return
    pred_close_price = hsi_direct["last_close"] * (1 + blended_return)
    pred_high_price = hsi_direct["last_close"] * (1 + hsi_direct["pred_high_return"])
    pred_low_price = hsi_direct["last_close"] * (1 + hsi_direct["pred_low_return"])

    print("[4/7] Applying regime-aware meta-confidence verdict (ENH#1)...")
    verdict = compute_verdict(hsi_direct["expected_move_pct"], hsi_direct["confidence"], hsi_direct["regime"])

    print("[4b/7] Deriving calibrated signal + risk-reward ratio for position sizing...")
    raw_direction = "LONG" if blended_return > 0 else "SHORT"
    calibrated_signal = raw_direction if verdict["worth_trading"] else "觀望"
    entry_price_for_rr = hsi_direct.get("estimated_entry_price") or hsi_direct["last_close"]
    risk_reward_ratio = _compute_risk_reward_ratio(
        entry_price_for_rr, pred_high_price, pred_low_price, raw_direction
    )

    print("[5/7] Computing fractional-Kelly position size (ENH#3)...")
    position = {
        "position_size_pct": 0.0, "direction": calibrated_signal, "p_win": None,
        "raw_kelly_fraction": 0.0, "fractional_kelly": 0.0,
        "confidence_scaled_kelly": 0.0, "regime_dampener": None, "capped": False,
        "reason": "position_sizer unavailable",
    }
    if _POSITION_SIZING_AVAILABLE:
        if verdict["worth_trading"] and risk_reward_ratio is not None:
            try:
                position = compute_position_size(
                    p_up=hsi_direct["p_up"],
                    calibrated_signal=calibrated_signal,
                    signal_confidence=hsi_direct["confidence"],
                    risk_reward_ratio=risk_reward_ratio,
                    regime=hsi_direct["regime"],
                    worth_trading=verdict["worth_trading"],
                )
            except Exception as e:
                print(f"predict: position sizing failed ({e}) — defaulting to 0% size.")
                position = {
                    "position_size_pct": 0.0, "direction": calibrated_signal, "p_win": None,
                    "raw_kelly_fraction": 0.0, "fractional_kelly": 0.0,
                    "confidence_scaled_kelly": 0.0, "regime_dampener": None, "capped": False,
                    "reason": f"error: {e}",
                }
        elif risk_reward_ratio is None:
            position["reason"] = "risk_reward_ratio unavailable (invalid risk leg) — sizing skipped"
        else:
            position["reason"] = "verdict=not worth trading"

    print("[6/7] Generating watchlist predictions (1211.HK, 0968.HK)...")
    watchlist_rows = predict_watchlist()

    result = {
        "date": today.isoformat(),
        "run_mode": run_mode,
        "session": hsi_direct["session"],
        "last_close": hsi_direct["last_close"],
        "estimated_entry_price": hsi_direct["estimated_entry_price"],
        "gap_model_status": hsi_direct["gap_model_status"],
        "pred_close_return_blended": blended_return,
        "pred_close_price": pred_close_price,
        "pred_high_price": pred_high_price,
        "pred_low_price": pred_low_price,
        "pred_close_q10": hsi_direct["pred_close_q10"],
        "pred_close_q90": hsi_direct["pred_close_q90"],
        "regime": hsi_direct["regime"],
        "p_up": hsi_direct["p_up"],
        "signal_strength": hsi_direct["signal_strength"],
        "confidence": hsi_direct["confidence"],
        "expected_move_pct": hsi_direct["expected_move_pct"],
        "worth_trading": verdict["worth_trading"],
        "threshold_used": verdict["threshold_used"],
        "raw_direction": raw_direction,
        "calibrated_signal": calibrated_signal,
        "risk_reward_ratio": risk_reward_ratio,
        "position_size_pct": position.get("position_size_pct"),
        "raw_kelly_fraction": position.get("raw_kelly_fraction"),
        "fractional_kelly": position.get("fractional_kelly"),
        "confidence_scaled_kelly": position.get("confidence_scaled_kelly"),
        "regime_dampener": position.get("regime_dampener"),
        "position_capped": position.get("capped"),
        "position_reason": position.get("reason"),
        "ensemble_weights_used": hsi_direct["ensemble_weights_used"],
        "ensemble_weight_source": hsi_direct["ensemble_weight_source"],
        "sub_model_preds": hsi_direct["sub_model_preds"],
        "bottom_up_return": bottom_up_return,
        "stock_predictions": stock_rows,
        "watchlist_predictions": watchlist_rows,
        "llm_commentary": None,
        "stock_llm_summaries": {},
    }

    print("[7/7] Generating LLM commentary + per-stock summaries (optional)...")
    if _LLM_ANALYSIS_AVAILABLE:
        try:
            result["llm_commentary"] = generate_market_commentary(result)
        except Exception as e:
            print(f"predict: market commentary failed ({e}) — commentary omitted.")
        try:
            result["stock_llm_summaries"] = generate_stock_summaries_batch(stock_rows + watchlist_rows)
        except Exception as e:
            print(f"predict: stock LLM summaries failed ({e}) — summaries omitted.")

    if run_mode == "official":
        try:
            from config import SNAPSHOT_DIR
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            snapshot_path = SNAPSHOT_DIR / f"{today.isoformat()}_hsi_features.csv"
            hsi_direct["feature_snapshot_row"].to_csv(snapshot_path, index=False)
            print(f"predict: feature snapshot saved to {snapshot_path}")
        except Exception as e:
            print(f"predict: feature snapshot save failed ({e}) — skipped, non-fatal.")

    log_prediction(result)

    print("=== Daily prediction complete ===")
    print(json.dumps({k: v for k, v in result.items()
                       if k not in ("stock_predictions", "watchlist_predictions")},
                      indent=2, default=str))
    return result


def log_prediction(result: dict):
    from config import LATEST_RESULT_PATH, PRED_DIR

    if result.get("run_mode") != "official":
        preliminary_path = PRED_DIR / "latest_preliminary_result.json"
        try:
            with open(preliminary_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"predict: run_mode='{result.get('run_mode')}' (preliminary) — "
                  f"wrote preview snapshot to {preliminary_path}.")
        except Exception as e:
            print(f"predict: failed to write preliminary snapshot ({e}).")
        return

    row = {
        "date": result["date"],
        "last_close": result["last_close"],
        "estimated_entry_price": result.get("estimated_entry_price"),
        "pred_close_return_blended": result["pred_close_return_blended"],
        "pred_close_price": result["pred_close_price"],
        "pred_high_price": result["pred_high_price"],
        "pred_low_price": result["pred_low_price"],
        "regime": result["regime"],
        "p_up": result["p_up"],
        "confidence": result["confidence"],
        "expected_move_pct": result["expected_move_pct"],
        "worth_trading": result["worth_trading"],
        "calibrated_signal": result["calibrated_signal"],
        "risk_reward_ratio": result["risk_reward_ratio"],
        "position_size_pct": result["position_size_pct"],
        "ensemble_weight_source": result["ensemble_weight_source"],
        "sub_pred_lgb": result["sub_model_preds"].get("lgb"),
        "sub_pred_rf": result["sub_model_preds"].get("rf"),
        "sub_pred_ridge": result["sub_model_preds"].get("ridge"),
        "actual_close": None,
    }
    df_row = pd.DataFrame([row])
    if PRED_LOG_PATH.exists():
        df_row.to_csv(PRED_LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(PRED_LOG_PATH, mode="w", header=True, index=False)

    with open(LATEST_RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    run_daily_prediction()

