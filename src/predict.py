"""
INFERENCE SCRIPT — run daily by GitHub Actions.
Loads pre-trained models (trained in Colab), fetches latest data,
generates today's prediction, and appends to the prediction log
for later comparison against actuals.
"""

import json
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

import sys
sys.path.append("..")
from config import (MODEL_Q10_PATH, MODEL_Q50_PATH, MODEL_Q90_PATH,
                     PRED_LOG_PATH, MODEL_DIR)
from data_sources import fetch_with_fallback, save_as_last_good
from features import build_features


def load_models():
    m10 = lgb.Booster(model_file=str(MODEL_Q10_PATH))
    m50 = lgb.Booster(model_file=str(MODEL_Q50_PATH))
    m90 = lgb.Booster(model_file=str(MODEL_Q90_PATH))
    with open(MODEL_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)
    return m10, m50, m90, feature_cols


def predict_today():
    raw = fetch_with_fallback()
    save_as_last_good(raw)
    feat_df = build_features(raw)

    m10, m50, m90, feature_cols = load_models()

    # align columns exactly to training-time feature set (missing cols -> 0)
    latest_row = feat_df.iloc[[-1]].copy()
    for col in feature_cols:
        if col not in latest_row.columns:
            latest_row[col] = 0
    X_latest = latest_row[feature_cols].select_dtypes(include=[np.number])

    pred_q10 = float(m10.predict(X_latest)[0])
    pred_q50 = float(m50.predict(X_latest)[0])
    pred_q90 = float(m90.predict(X_latest)[0])

    last_close = float(latest_row["Close"].values[0])
    result = {
        "predict_date": str(latest_row.index[0].date()),
        "run_timestamp": datetime.datetime.utcnow().isoformat(),
        "last_close": last_close,
        "pred_return_q10": pred_q10,
        "pred_return_q50": pred_q50,
        "pred_return_q90": pred_q90,
        "pred_price_low": last_close * (1 + pred_q10),
        "pred_price_mid": last_close * (1 + pred_q50),
        "pred_price_high": last_close * (1 + pred_q90),
        "entry_price_suggestion": last_close * (1 + pred_q10 * 0.5),
        "target_price": last_close * (1 + pred_q90),
        "stop_loss_price": last_close * (1 + pred_q10),
        "data_source": latest_row["source"].values[0],
        "is_stale": bool(latest_row["is_stale"].values[0]),
        "actual_close": None,        # filled in later once known (see compare step)
        "directional_hit": None,
    }
    return result


def append_to_log(result: dict):
    row = pd.DataFrame([result])
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
