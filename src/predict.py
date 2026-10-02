"""
INFERENCE SCRIPT — run daily by GitHub Actions. CPU-only, lightweight.
1. Loads pre-trained Ensemble + quantile models (trained in Colab).
2. Generates HSI-level prediction (ensemble + quantile band).
3. Generates bottom-up prediction (weighted aggregation of stock q10/q50/q90).
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
from ensemble_model import HSIEnsembleModel
from walk_forward import should_retrain


def load_hsi_models():
    m10 = lgb.Booster(model_file=str(MODEL_Q10_PATH))
    m50 = lgb.Booster(model_file=str(MODEL_Q50_PATH))
    m90 = lgb.Booster(model_file=str(MODEL_Q90_PATH))
    ensemble = HSIEnsembleModel.load(prefix="hsi")
    with open(FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)
    return m10, m50, m90, ensemble, feature_cols


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
                              market_sentiment=market_sentiment, stock_sentiment=0.0,
                              ticker=HSI_TICKER)

    m10, m50, m90, ensemble, feature_cols = load_hsi_models()
    latest_row = feat_df.iloc[[-1]].copy()
    X_latest = align_features(latest_row, feature_cols)

    # Ensemble gives the primary point estimate; q10/q90 give the band
    pred_q10 = float(m10.predict(X_latest)[0])
    pred_ensemble = float(ensemble.predict(X_latest)[0])
    pred_q90 = float(m90.predict(X_latest)[0])

    meta_model = load_meta_model()
    confidence = get_signal_confidence(meta_model, X_latest)

    last_close = float(latest_row["Close"].values[0])
    return {
        "last_close": last_close,
        "pred_return_q10": pred_q10,
        "pred_return_q50": pred_ensemble,   # ensemble replaces plain q50 as point estimate
        "pred_return_q90": pred_q90,
        "signal_confidence": confidence,
        "data_source": latest_row["source"].values[0],
        "is_stale": bool(latest_row["is_stale"].values[0]),
        "predict_date": str(latest_row.index[0].date()),
    }


def predict_hsi_bottom_up():
    """Aggregate per-stock q10/q50/q90 predictions weighted by index weight."""
    universe = get_universe()
    weighted = {"q10": 0.0, "q50": 0.0, "q90": 0.0}
    total_weight_used = 0.0
    stock_details = {}

    for ticker, weight in universe.items():
        q50_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_q50.txt"
        feat_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_features.json"
        if not q50_path.exists() or not feat_path.exists():
            continue
        try:
            stooq_code = ticker.replace(".HK", "").zfill(5) + ".hk"
            raw = fetch_with_fallback(ticker, stooq_ticker=stooq_code)
            keywords = [ticker.split(".")[0]]
            stock_sentiment = get_stock_sentiment(keywords)
            ccass_change = get_ccass_change(ticker.split(".")[0])

            feat_df = build_features(raw, ccass_change=ccass_change,
                                      stock_sentiment=stock_sentiment, ticker=ticker)
            with open(feat_path) as f:
                feature_cols = json.load(f)

            latest_row = feat_df.iloc[[-1]].copy()
            X_latest = align_features(latest_row, feature_cols)

            stock_preds = {}
            for suffix in ["q10", "q50", "q90"]:
                model_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_{suffix}.txt"
                if model_path.exists():
                    model = lgb.Booster(model_file=str(model_path))
                    stock_preds[suffix] = float(model.predict(X_latest)[0])
                else:
                    stock_preds[suffix] = 0.0

            for k in weighted:
                weighted[k] += stock_preds[k] * weight
            total_weight_used += weight
            stock_details[ticker] = stock_preds
        except Exception as e:
            print(f"Bottom-up predict failed for {ticker}: {e}")
            continue

    if total_weight_used > 0:
        weighted = {k: v / total_weight_used for k, v in weighted.items()}

    return {
        "bottom_up_q10": weighted["q10"],
        "bottom_up_q50": weighted["q50"],
        "bottom_up_q90": weighted["q90"],
        "stock_details": stock_details,
        "coverage_weight": total_weight_used,
    }


def predict_today():
    direct = predict_hsi_direct()
    bottom_up = predict_hsi_bottom_up()

    has_good_coverage = bottom_up["coverage_weight"] > 0.3
    if has_good_coverage:
        blended_q10 = (direct["pred_return_q10"] + bottom_up["bottom_up_q10"]) / 2
        blended_q50 = (direct["pred_return_q50"] + bottom_up["bottom_up_q50"]) / 2
        blended_q90 = (direct["pred_return_q90"] + bottom_up["bottom_up_q90"]) / 2
    else:
        blended_q10 = direct["pred_return_q10"]
        blended_q50 = direct["pred_return_q50"]
        blended_q90 = direct["pred_return_q90"]

    last_close = direct["last_close"]

    # Check walk-forward status for visibility in the log (does not block prediction)
    wf_status = should_retrain()

    result = {
        "predict_date": direct["predict_date"],
        "run_timestamp": datetime.datetime.utcnow().isoformat(),
        "last_close": last_close,
        "pred_return_q10": blended_q10,
        "pred_return_q50_direct": direct["pred_return_q50"],
        "pred_return_bottom_up": bottom_up["bottom_up_q50"],
        "pred_return_q50_blended": blended_q50,
        "pred_return_q90": blended_q90,
        "pred_price_low": last_close * (1 + blended_q10),
        "pred_price_mid": last_close * (1 + blended_q50),
        "pred_price_high": last_close * (1 + blended_q90),
        "entry_price_suggestion": last_close * (1 + blended_q10 * 0.5),
        "target_price": last_close * (1 + blended_q90),
        "stop_loss_price": last_close * (1 + blended_q10),
        "signal_confidence": direct["signal_confidence"],
        "bottom_up_coverage_weight": bottom_up["coverage_weight"],
        "data_source": direct["data_source"],
        "is_stale": direct["is_stale"],
        "days_since_last_train": wf_status["days_since_last_train"],
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
