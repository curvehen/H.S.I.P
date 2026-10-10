"""
INFERENCE SCRIPT — run daily by GitHub Actions. CPU-only, lightweight.

1. Loads pre-trained HSI models (Ensemble + quantile + high/low).
2. Generates direct HSI-level next-day prediction, blending sub-model
   outputs via DYNAMIC (Brier-score) weights when available, falling back
   to the ensemble's static inverse-RMSE weights otherwise. (ENH#2)
3. Generates bottom-up prediction (weighted aggregation of HSI constituent
   stock close predictions — watchlist tickers are NEVER included here).
4. Blends direct + bottom-up.
5. Applies meta-confidence filter, using REGIME-SPECIFIC calibrated
   thresholds when available, falling back to the global static
   MIN_EXPECTED_MOVE_PCT / MIN_CONFIDENCE otherwise. (ENH#1)
6. Computes a fractional-Kelly suggested position size for the verdict. (ENH#3)
7. Generates standalone predictions for watchlist tickers (1211.HK, 0968.HK),
   reported separately from the HSI constituent table.
8. (Optional) Generates a short LLM commentary via DashScope, fail-safe —
   disabled or any failure simply yields a None commentary, never crashes
   the pipeline.
9. Outputs entry/high/low/close + worth-trading verdict + position size.
10. Appends everything (incl. raw sub-model predictions, for next day's
    dynamic weight update by evaluate_drift.py) to the prediction log.

Generates: P(up)/P(down), signal strength, regime, HSI high/low/close,
range stats, position sizing, per-stock prediction table with hit rates,
watchlist table, and optional LLM commentary.
"""

import json
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

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
from market_hours import get_latest_usable_row, get_next_trading_day
import os
from market_hours import HKT
from datetime import datetime as _dt

# ---------------------------------------------------------------------------
# NEW: optional ENH modules — every one of them is wrapped so a missing file,
# a missing models/*.json artifact, or an internal exception degrades to a
# documented static fallback instead of crashing the daily GitHub Actions run.
# ---------------------------------------------------------------------------

# ENH#1 — regime-specific F1-calibrated thresholds.
#   Contract: get_regime_threshold(regime: str) -> dict with keys
#   "MIN_EXPECTED_MOVE_PCT", "MIN_CONFIDENCE", "source" ("calibrated"|"fallback").
try:
    from threshold_calibrator import get_regime_threshold
    _REGIME_THRESHOLDS_AVAILABLE = True
except ImportError:
    _REGIME_THRESHOLDS_AVAILABLE = False
    print("predict: threshold_calibrator not available — using static global thresholds.")

# ENH#2 — dynamic Brier-score ensemble weighting.
#   Contract: get_effective_weights(static_weights: dict) -> (dict, str)
#   where str is "dynamic_brier" or "static_fallback".
try:
    from dynamic_ensemble_weighter import get_effective_weights
    _DYNAMIC_WEIGHTING_AVAILABLE = True
except ImportError:
    _DYNAMIC_WEIGHTING_AVAILABLE = False
    print("predict: dynamic_ensemble_weighter not available — using static ensemble weights.")

# ENH#3 — fractional Kelly position sizing.
#   Contract: compute_position_size(p_up: float, confidence: float,
#   expected_move_pct: float) -> dict with keys "position_pct", "kelly_raw", "reason".
try:
    from position_sizer import compute_position_size
    _POSITION_SIZING_AVAILABLE = True
except ImportError:
    _POSITION_SIZING_AVAILABLE = False
    print("predict: position_sizer not available — position sizing will be omitted from output.")

# LLM commentary (DashScope-compatible).
#   Contract: generate_daily_commentary(context: dict) -> str | None.
#   Must internally respect config.LLM_ANALYSIS_ENABLED and
#   config.LLM_REQUEST_TIMEOUT_SECONDS, returning None on any failure/timeout.
try:
    from llm_analysis import generate_market_commentary, generate_stock_summaries_batch
    _LLM_ANALYSIS_AVAILABLE = True
except ImportError:
    _LLM_ANALYSIS_AVAILABLE = False
    print("predict: llm_analysis not available — no commentary will be generated.")


def determine_run_mode(now: datetime = None) -> str:
    """
    Decides whether this execution is 'preliminary' (7pm HKT scouting run,
    no logging) or 'official' (4am HKT run, full logging + snapshot).
    Priority: explicit RUN_MODE env var (set by daily_predict.yml per cron
    job) > HKT wall-clock fallback for manual/local runs.
    """
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


# ---------------------------------------------------------------------------
# HSI direct prediction (UPDATED for ENH#2)
# ---------------------------------------------------------------------------

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

    # ---- Sub-model raw predictions (needed for dynamic weight blending AND
    #      for logging, so evaluate_drift.py can compute each sub-model's
    #      own rolling Brier score against the next day's actual outcome) ----
    sub_preds = ensemble.predict_submodels(X_latest)  # {"lightgbm": v, "random_forest": v, "ridge": v}
    static_weights = ensemble.weights  # {"lightgbm": w, "random_forest": w, "ridge": w}

    # ---- NEW (ENH#2): dynamic Brier-score weight blending ----
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
        "feature_snapshot_row": latest_row,   # NEW: exposes the feature row used for this prediction, for official-run drift/backtest snapshotting
    }


# ---------------------------------------------------------------------------
# Single-stock prediction (shared by HSI constituents AND watchlist)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Bottom-up aggregation (HSI constituents ONLY — watchlist excluded)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NEW: standalone watchlist predictions (1211.HK, 0968.HK) — reported
# separately, NEVER folded into predict_bottom_up()'s weighted_return.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NEW (ENH#1): regime-aware worth-trading verdict
# ---------------------------------------------------------------------------

def get_effective_threshold(regime: str) -> dict:
    """Returns {"MIN_EXPECTED_MOVE_PCT", "MIN_CONFIDENCE", "source"}.
    Falls back cleanly to static global config values if regime calibration
    is unavailable, uncalibrated for this regime, or raises any error."""
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


# ---------------------------------------------------------------------------
# Main daily prediction routine
# ---------------------------------------------------------------------------

def run_daily_prediction(run_mode: str = None):
    run_mode = run_mode or determine_run_mode()
    today = datetime.date.today()
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

    print("[5/7] Computing fractional-Kelly position size (ENH#3)...")
    position = {"position_pct": 0.0, "kelly_raw": 0.0, "reason": "position_sizer unavailable"}
    if _POSITION_SIZING_AVAILABLE and verdict["worth_trading"]:
        try:
            position = compute_position_size(
                p_up=hsi_direct["p_up"],
                confidence=hsi_direct["confidence"],
                expected_move_pct=hsi_direct["expected_move_pct"],
            )
        except Exception as e:
            print(f"predict: position sizing failed ({e}) — defaulting to 0% size.")
            position = {"position_pct": 0.0, "kelly_raw": 0.0, "reason": f"error: {e}"}
    elif not verdict["worth_trading"]:
        position = {"position_pct": 0.0, "kelly_raw": 0.0, "reason": "verdict=not worth trading"}

    print("[6/7] Generating watchlist predictions (1211.HK, 0968.HK)...")
    watchlist_rows = predict_watchlist()

    # ---- Assemble result dict FIRST (moved ahead of LLM step, since the
    #      corrected LLM functions read real fields off this dict directly) ----
    result = {
        "date": today.isoformat(),
        "run_mode": run_mode,
        "session": hsi_direct["session"],
        "last_close": hsi_direct["last_close"],
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
        "position_pct": position["position_pct"],
        "kelly_raw": position.get("kelly_raw"),
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
            # Covers BOTH HSI constituents and watchlist tickers (ENH#5 scope expansion)
            result["stock_llm_summaries"] = generate_stock_summaries_batch(stock_rows + watchlist_rows)
        except Exception as e:
            print(f"predict: stock LLM summaries failed ({e}) — summaries omitted.")

    # ---- NEW: feature snapshot, official runs only (4am HKT) ----
    if run_mode == "official":
        try:
            from config import SNAPSHOT_DIR
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            snapshot_path = SNAPSHOT_DIR / f"{today.isoformat()}_hsi_features.csv"
            hsi_direct["feature_snapshot_row"].to_csv(snapshot_path, index=False)
            print(f"predict: feature snapshot saved to {snapshot_path}")
        except Exception as e:
            print(f"predict: feature snapshot save failed ({e}) — skipped, non-fatal.")

    # ---- log_prediction() internally no-ops for preliminary runs ----
    log_prediction(result)

    print("=== Daily prediction complete ===")
    print(json.dumps({k: v for k, v in result.items()
                       if k not in ("stock_predictions", "watchlist_predictions")},
                      indent=2, default=str))
    return result



# ---------------------------------------------------------------------------
# Logging (UPDATED: now persists raw sub-model preds + weight source, so
# evaluate_drift.py can compute each sub-model's Brier score and feed
# dynamic_ensemble_weighter.py the next day)
# ---------------------------------------------------------------------------

def log_prediction(result: dict):
    if result.get("run_mode") != "official":
        print(f"predict: run_mode='{result.get('run_mode')}' (preliminary) — "
              f"skipping CSV log / latest_result.json write.")
        return

    row = {
        "date": result["date"],
        "last_close": result["last_close"],
        "pred_close_return_blended": result["pred_close_return_blended"],
        "pred_close_price": result["pred_close_price"],
        "pred_high_price": result["pred_high_price"],
        "pred_low_price": result["pred_low_price"],
        "regime": result["regime"],
        "p_up": result["p_up"],
        "confidence": result["confidence"],
        "expected_move_pct": result["expected_move_pct"],
        "worth_trading": result["worth_trading"],
        "position_pct": result["position_pct"],
        "ensemble_weight_source": result["ensemble_weight_source"],
        "sub_pred_lightgbm": result["sub_model_preds"].get("lightgbm"),
        "sub_pred_random_forest": result["sub_model_preds"].get("random_forest"),
        "sub_pred_ridge": result["sub_model_preds"].get("ridge"),
        "actual_close": None,   # backfilled by evaluate_drift.py once known
    }
    df_row = pd.DataFrame([row])
    if PRED_LOG_PATH.exists():
        df_row.to_csv(PRED_LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(PRED_LOG_PATH, mode="w", header=True, index=False)

    from config import LATEST_RESULT_PATH
    with open(LATEST_RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)



if __name__ == "__main__":
    run_daily_prediction()

