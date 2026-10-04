"""
INFERENCE SCRIPT — run daily by GitHub Actions. CPU-only, lightweight.
1. Loads pre-trained HSI models (Ensemble + quantile + high/low).
2. Generates direct HSI-level next-day prediction.
3. Generates bottom-up prediction (weighted aggregation of stock close predictions).
4. Blends direct + bottom-up.
5. Applies meta-confidence filter.
6. Outputs entry/high/low/close + worth-trading verdict.
7. Appends everything to the prediction log.
"""

import json
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (MODEL_CLOSE_Q10_PATH, MODEL_CLOSE_Q50_PATH, MODEL_CLOSE_Q90_PATH,
                     MODEL_HIGH_PATH, MODEL_LOW_PATH, FEATURE_LIST_PATH, PRED_LOG_PATH,
                     STOCK_MODEL_DIR, HSI_TICKER, US_FUTURES_TICKER, VIX_TICKER,
                     MIN_EXPECTED_MOVE_PCT, MIN_CONFIDENCE)
from data_sources import fetch_with_fallback
from features import build_features
from labeling import LABEL_COLUMNS
from confidence import load_meta_model, get_signal_confidence
from news_sentiment import get_daily_market_sentiment, get_stock_sentiment
from stock_universe import get_universe
from ensemble_model import HSIEnsembleModel


def load_hsi_models():
    m_close_q10 = lgb.Booster(model_file=str(MODEL_CLOSE_Q10_PATH))
    m_close_q50 = lgb.Booster(model_file=str(MODEL_CLOSE_Q50_PATH))
    m_close_q90 = lgb.Booster(model_file=str(MODEL_CLOSE_Q90_PATH))
    m_high = lgb.Booster(model_file=str(MODEL_HIGH_PATH))
    m_low = lgb.Booster(model_file=str(MODEL_LOW_PATH))
    ensemble = HSIEnsembleModel.load(prefix="hsi")
    with open(FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)
    return m_close_q10, m_close_q50, m_close_q90, m_high, m_low, ensemble, feature_cols


def align_features(latest_row: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    for col in feature_cols:
        if col not in latest_row.columns:
            latest_row[col] = 0
    return latest_row[feature_cols].astype(float)


def predict_hsi_direct():
    us_futures = fetch_with_fallback(US_FUTURES_TICKER)
    vix = fetch_with_fallback(VIX_TICKER)
    market_sentiment = get_daily_market_sentiment()

    raw = fetch_with_fallback(HSI_TICKER, stooq_ticker="^hsi")
    feat_df = build_features(raw, us_futures=us_futures, vix=vix,
                              ccass_change=0.0, market_sentiment=market_sentiment,
                              stock_sentiment=0.0, ticker=HSI_TICKER)

    m_close_q10, m_close_q50, m_close_q90, m_high, m_low, ensemble, feature_cols = load_hsi_models()

    latest_row = feat_df.iloc[[-1]].copy()
    X_latest = align_features(latest_row, feature_cols)

    pred_close_q10 = float(m_close_q10.predict(X_latest)[0])
    pred_close_ensemble = float(ensemble.predict_loaded(X_latest)[0])
    pred_close_q90 = float(m_close_q90.predict(X_latest)[0])
    pred_high = float(m_high.predict(X_latest)[0])
    pred_low = float(m_low.predict(X_latest)[0])

    meta_model = load_meta_model()
    confidence = get_signal_confidence(meta_model, X_latest)

    last_close = float(latest_row["Close"].values[0])
    return {
        "last_close": last_close,
        "pred_close_return_q10": pred_close_q10,
        "pred_close_return_mid": pred_close_ensemble,
        "pred_close_return_q90": pred_close_q90,
        "pred_high_return": pred_high,
        "pred_low_return": pred_low,
        "signal_confidence": confidence,
        "data_source": latest_row["source"].values[0],
        "is_stale": bool(latest_row["is_stale"].values[0]),
        "predict_date": str(latest_row.index[0].date()),
    }


def predict_hsi_bottom_up():
    """Aggregate per-stock next-close-return predictions weighted by index weight."""
    universe = get_universe()
    weighted_return = 0.0
    total_weight_used = 0.0
    stock_details = {}

    for ticker, weight in universe.items():
        model_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_close_q50.txt"
        feat_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_features.json"
        if not model_path.exists() or not feat_path.exists():
            continue
        try:
            from ccass_scraper import get_ccass_change
            stooq_code = ticker.replace(".HK", "").zfill(5) + ".hk"
            raw = fetch_with_fallback(ticker, stooq_ticker=stooq_code)
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
            stock_details[ticker] = pred_return
        except Exception as e:
            print(f"Bottom-up predict failed for {ticker}: {e}")
            continue

    if total_weight_used > 0:
        weighted_return = weighted_return / total_weight_used

    return {"bottom_up_close_return": weighted_return, "stock_details": stock_details,
            "coverage_weight": total_weight_used}


def is_worth_trading(predicted_close_return: float, confidence: float) -> dict:
    expected_move = abs(predicted_close_return)
    worth_it = expected_move >= MIN_EXPECTED_MOVE_PCT and confidence >= MIN_CONFIDENCE

    if not worth_it:
        if expected_move < MIN_EXPECTED_MOVE_PCT:
            reason = f"預測波幅太細 ({expected_move:.2%} < {MIN_EXPECTED_MOVE_PCT:.2%})，扣除成本後無意義"
        else:
            reason = f"模型信心不足 ({confidence:.2f} < {MIN_CONFIDENCE})"
    else:
        reason = "預測幅度同信心度均達標"

    return {"worth_trading": worth_it, "reason": reason}


def predict_today():
    direct = predict_hsi_direct()
    bottom_up = predict_hsi_bottom_up()

    has_good_coverage = bottom_up["coverage_weight"] > 0.3
    if has_good_coverage:
        blended_close_return = (direct["pred_close_return_mid"] + bottom_up["bottom_up_close_return"]) / 2
    else:
        blended_close_return = direct["pred_close_return_mid"]

    last_close = direct["last_close"]
    verdict = is_worth_trading(blended_close_return, direct["signal_confidence"])

    predicted_close = last_close * (1 + blended_close_return)
    predicted_high = last_close * (1 + direct["pred_high_return"])
    predicted_low = last_close * (1 + direct["pred_low_return"])
    # Entry suggestion: conservative level between current close and predicted low
    # (gives room for a pullback entry rather than chasing the predicted close directly)
    suggested_entry = last_close + (predicted_low - last_close) * 0.5

    result = {
        "predict_date": direct["predict_date"],
        "run_timestamp": datetime.datetime.utcnow().isoformat(),
        "last_close": last_close,
        "entry_price": suggested_entry,
        "pred_high": predicted_high,
        "pred_low": predicted_low,
        "pred_close": predicted_close,
        "pred_close_return_direct": direct["pred_close_return_mid"],
        "pred_close_return_bottom_up": bottom_up["bottom_up_close_return"],
        "pred_close_return_blended": blended_close_return,
        "signal_confidence": direct["signal_confidence"],
        "worth_trading": verdict["worth_trading"],
        "worth_trading_reason": verdict["reason"],
        "bottom_up_coverage_weight": bottom_up["coverage_weight"],
        "data_source": direct["data_source"],
        "is_stale": direct["is_stale"],
        "actual_close": None,
        "directional_hit": None,
    }
    return result


def append_to_log(result: dict):
    row = pd.DataFrame([{k: v for k, v in result.items() if k != "stock_details"}])
    if PRED_LOG_PATH.exists():
        log = pd.read_csv(PRED_LOG_PATH)
        log = pd.concat([log, row], ignore_index=True)
    else:
        log = row
    log.to_csv(PRED_LOG_PATH, index=False)


if __name__ == "__main__":
    result = predict_today()
    append_to_log(result)
    print(json.dumps(result, indent=2, default=str))
