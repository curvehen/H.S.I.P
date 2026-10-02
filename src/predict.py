"""
INFERENCE SCRIPT — run daily by GitHub Actions. CPU-only, lightweight.
1. Loads pre-trained HSI + per-stock models (trained in Colab).
2. Generates HSI-level prediction (direct model).
3. Generates bottom-up prediction (weighted aggregation of stock predictions).
4. Applies meta-labeling confidence filter.
5. Outputs entry/exit/target/stop-loss levels.
6. Appends everything to the prediction log.
"""

import json
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (MODEL_Q10_PATH, MODEL_Q50_PATH, MODEL_Q90_PATH,
                     FEATURE_LIST_PATH, PRED_LOG_PATH, STOCK_MODEL_DIR,
                     HSI_TICKER, US_FUTURES_TICKER, VIX_TICKER)
from data_sources import fetch_with_fallback
from features import build_features
from meta_labeling import load_meta_model, get_signal_confidence
from ccass_scraper import get_ccass_change
from news_sentiment import get_daily_market_sentiment, get_stock_sentiment
from stock_universe import get_universe


def load_hsi_models():
    m10 = lgb.Booster(model_file=str(MODEL_Q10_PATH))
    m50 = lgb.Booster(model_file=str(MODEL_Q50_PATH))
    m90 = lgb.Booster(model_file=str(MODEL_Q90_PATH))
    with open(FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)
    return m10, m50, m90, feature_cols


def align_features(latest_row: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    for col in feature_cols:
        if col not in latest_row.columns:
            latest_row[col] = 0
    return latest_row[feature_cols].select_dtypes(include=[np.number])


def predict_hsi_direct():
    us_futures = fetch_with_fallback(US_FUTURES_TICKER)
    vix = fetch_with_fallback(VIX_TICKER)
    market_sentiment = get_daily_market_sentiment()

    raw = fetch_with_fallback(HSI_TICKER, stooq_ticker="^hsi")
    feat_df = build_features(raw, us_futures, vix, ccass_change=0.0,
                              market_sentiment=market_sentiment, stock_sentiment=0.0)

    m10, m50, m90, feature_cols = load_hsi_models()
    latest_row = feat_df.iloc[[-1]].copy()
    X_latest = align_features(latest_row, feature_cols)

    pred_q10 = float(m10.predict(X_latest)[0])
    pred_q50 = float(m50.predict(X_latest)[0])
    pred_q90 = float(m90.predict(X_latest)[0])

    meta_model = load_meta_model()
    confidence = get_signal_confidence(meta_model, X_latest)

    last_close = float(latest_row["Close"].values[0])
    return {
        "last_close": last_close,
        "pred_return_q10": pred_q10,
        "pred_return_q50": pred_q50,
        "pred_return_q90": pred_q90,
        "signal_confidence": confidence,
        "data_source": latest_row["source"].values[0],
        "is_stale": bool(latest_row["is_stale"].values[0]),
        "predict_date": str(latest_row.index[0].date()),
    }


def predict_hsi_bottom_up():
    """Aggregate per-stock predictions weighted by index weight -> implied HSI return."""
    universe = get_universe()
    weighted_return = 0.0
    total_weight_used = 0.0
    stock_details = {}

    for ticker, weight in universe.items():
        model_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_q50.txt"
        feat_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_features.json"
        if not model_path.exists() or not feat_path.exists():
            continue
        try:
            stooq_code = ticker.replace(".HK", "").zfill(5) + ".hk"
            raw = fetch_with_fallback(ticker, stooq_ticker=stooq_code)
            keywords = [ticker.split(".")[0]]
            stock_sentiment = get_stock_sentiment(keywords)
            ccass_change = get_ccass_change(ticker.split(".")[0])

            feat_df = build_features(raw, ccass_change=ccass_change,
                                      stock_sentiment=stock_sentiment)
            model = lgb.Booster(model_file=str(model_path))
            with open(feat_path) as f:
                feature_cols = json.load(f)

            latest_row = feat_df.iloc[[-1]].copy()
            X_latest = align_features(latest_row, feature_cols)
            pred_return = float(model.predict(X_latest)[0])

            weighted_return += pred_return * weight
            total_weight_used += weight
            stock_details[ticker] = pred_return
        except Exception as e:
            print(f"Bottom-up predict failed for {ticker}: {e}")
            continue

    if total_weight_used > 0:
        weighted_return = weighted_return / total_weight_used  # renormalize for missing stocks

    return {"bottom_up_pred_return": weighted_return, "stock_details": stock_details,
            "coverage_weight": total_weight_used}


def predict_today():
    direct = predict_hsi_direct()
    bottom_up = predict_hsi_bottom_up()

    # Blend direct HSI model with bottom-up estimate (simple average; adjust weighting as desired)
    blended_q50 = (direct["pred_return_q50"] + bottom_up["bottom_up_pred_return"]) / 2 \
        if bottom_up["coverage_weight"] > 0.3 else direct["pred_return_q50"]

    last_close = direct["last_close"]
    result = {
        "predict_date": direct["predict_date"],
        "run_timestamp": datetime.datetime.utcnow().isoformat(),
        "last_close": last_close,
        "pred_return_q10": direct["pred_return_q10"],
        "pred_return_q50_direct": direct["pred_return_q50"],
        "pred_return_bottom_up": bottom_up["bottom_up_pred_return"],
        "pred_return_q50_blended": blended_q50,
        "pred_return_q90": direct["pred_return_q90"],
        "pred_price_low": last_close * (1 + direct["pred_return_q10"]),
        "pred_price_mid": last_close * (1 + blended_q50),
        "pred_price_high": last_close * (1 + direct["pred_return_q90"]),
        "entry_price_suggestion": last_close * (1 + direct["pred_return_q10"] * 0.5),
        "target_price": last_close * (1 + direct["pred_return_q90"]),
        "stop_loss_price": last_close * (1 + direct["pred_return_q10"]),
        "signal_confidence": direct["signal_confidence"],
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
