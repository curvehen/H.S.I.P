"""
Backtest module — simulates historical daily predictions vs actuals.

Two modes:
1. run_insample_backtest(): FAST. Uses the already-trained model (which saw
   this period during training) to generate predictions for each day in range.
   CAVEAT: This is in-sample — performance here is optimistic and does NOT
   represent true out-of-sample predictive accuracy. Use only as a system
   sanity-check / demonstration, not as proof of real-world performance.

2. run_walkforward_oos_backtest(): SLOWER but methodologically correct.
   Retrains the model periodically (every N trading days) using only data
   available up to that point, then predicts forward until the next retrain.
   This simulates realistic walk-forward deployment and gives a much more
   honest estimate of real predictive performance.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (HSI_TICKER, US_FUTURES_TICKER, VIX_TICKER, LGB_PARAMS,
                     PRED_DIR, STOCK_MODEL_DIR, FEATURE_LIST_PATH,
                     MODEL_CLOSE_Q50_PATH, MODEL_HIGH_PATH, MODEL_LOW_PATH)
from data_sources import fetch_with_fallback
from features import build_features, get_numeric_feature_columns
from labeling import build_nextday_labels, LABEL_COLUMNS
from confidence import load_meta_model, get_signal_confidence
from stock_universe import get_universe


# ---------------------------------------------------------------------------
# Mode 1: Fast in-sample backtest (HSI)
# ---------------------------------------------------------------------------

def run_insample_backtest_hsi(start_date: str = "2026-01-01") -> pd.DataFrame:
    """
    Uses the CURRENTLY SAVED trained models to predict every day in
    [start_date, today]. IMPORTANT: in-sample — model was trained on
    data including this period. Results here are illustrative only.
    """
    us_futures = fetch_with_fallback(US_FUTURES_TICKER)
    vix = fetch_with_fallback(VIX_TICKER)

    raw = fetch_with_fallback(HSI_TICKER, stooq_ticker="^hsi")
    feat_df = build_features(raw, us_futures=us_futures, vix=vix,
                              ccass_change=0.0, market_sentiment=0.0,
                              stock_sentiment=0.0, ticker=HSI_TICKER, include_macro=True)
    labeled_df = build_nextday_labels(feat_df)

    with open(FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)

    # Align columns (add missing as 0) in case feature set evolved
    for col in feature_cols:
        if col not in labeled_df.columns:
            labeled_df[col] = 0
    X_all = labeled_df[feature_cols].astype(float)

    m_close = lgb.Booster(model_file=str(MODEL_CLOSE_Q50_PATH))
    m_high = lgb.Booster(model_file=str(MODEL_HIGH_PATH))
    m_low = lgb.Booster(model_file=str(MODEL_LOW_PATH))
    meta_model = load_meta_model()

    mask = labeled_df.index >= pd.Timestamp(start_date)
    subset = labeled_df[mask]
    X_subset = X_all[mask]

    if subset.empty:
        raise ValueError(f"No data available from {start_date} onwards for backtest.")

    pred_close_return = m_close.predict(X_subset)
    pred_high_return = m_high.predict(X_subset)
    pred_low_return = m_low.predict(X_subset)

    confidences = []
    for i in range(len(X_subset)):
        row = X_subset.iloc[[i]]
        conf = get_signal_confidence(meta_model, row)
        confidences.append(conf)

    results = pd.DataFrame({
        "date": subset.index,
        "last_close": subset["Close"].values,
        "pred_close_return": pred_close_return,
        "actual_close_return": subset["next_close_return"].values,
        "pred_high_return": pred_high_return,
        "actual_high_return": subset["next_high_return"].values,
        "pred_low_return": pred_low_return,
        "actual_low_return": subset["next_low_return"].values,
        "confidence": confidences,
    })

    results["pred_close"] = results["last_close"] * (1 + results["pred_close_return"])
    results["actual_close"] = results["last_close"] * (1 + results["actual_close_return"])
    results["pred_high"] = results["last_close"] * (1 + results["pred_high_return"])
    results["actual_high"] = results["last_close"] * (1 + results["actual_high_return"])
    results["pred_low"] = results["last_close"] * (1 + results["pred_low_return"])
    results["actual_low"] = results["last_close"] * (1 + results["actual_low_return"])

    results["directional_hit"] = (
        np.sign(results["pred_close_return"]) == np.sign(results["actual_close_return"])
    ).astype(int)

    results["abs_error_pct"] = (
        (results["pred_close"] - results["actual_close"]).abs() / results["actual_close"]
    )

    return results


# ---------------------------------------------------------------------------
# Mode 1: Fast in-sample backtest (per-stock)
# ---------------------------------------------------------------------------

def run_insample_backtest_stock(ticker: str, start_date: str = "2026-01-01") -> pd.DataFrame:
    model_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_close_q50.txt"
    feat_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_features.json"

    if not model_path.exists() or not feat_path.exists():
        return pd.DataFrame()

    stooq_code = ticker.replace(".HK", "").zfill(5) + ".hk"
    raw = fetch_with_fallback(ticker, stooq_ticker=stooq_code)
    feat_df = build_features(raw, ccass_change=0.0, stock_sentiment=0.0,
                              ticker=ticker, include_macro=False)
    labeled_df = build_nextday_labels(feat_df)

    with open(feat_path) as f:
        feature_cols = json.load(f)
    for col in feature_cols:
        if col not in labeled_df.columns:
            labeled_df[col] = 0
    X_all = labeled_df[feature_cols].astype(float)

    mask = labeled_df.index >= pd.Timestamp(start_date)
    subset = labeled_df[mask]
    X_subset = X_all[mask]

    if subset.empty:
        return pd.DataFrame()

    model = lgb.Booster(model_file=str(model_path))
    pred_close_return = model.predict(X_subset)

    results = pd.DataFrame({
        "date": subset.index,
        "ticker": ticker,
        "last_close": subset["Close"].values,
        "pred_close_return": pred_close_return,
        "actual_close_return": subset["next_close_return"].values,
    })
    results["pred_close"] = results["last_close"] * (1 + results["pred_close_return"])
    results["actual_close"] = results["last_close"] * (1 + results["actual_close_return"])
    results["directional_hit"] = (
        np.sign(results["pred_close_return"]) == np.sign(results["actual_close_return"])
    ).astype(int)

    return results


def run_insample_backtest_all_stocks(start_date: str = "2026-01-01") -> pd.DataFrame:
    universe = get_universe()
    all_results = []
    for ticker in universe:
        try:
            r = run_insample_backtest_stock(ticker, start_date)
            if not r.empty:
                all_results.append(r)
        except Exception as e:
            print(f"Backtest failed for {ticker}: {e}")
            continue
    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


# ---------------------------------------------------------------------------
# Mode 2: Walk-forward out-of-sample backtest (HSI) — slower, more honest
# ---------------------------------------------------------------------------

def run_walkforward_oos_backtest_hsi(start_date: str = "2026-01-01",
                                       retrain_every_n_days: int = 10) -> pd.DataFrame:
    """
    Methodologically correct backtest: retrains the model every N trading
    days using ONLY data available up to that point (no future leakage),
    then predicts forward until the next retrain point. Much slower than
    in-sample mode due to repeated training, but gives an honest estimate
    of real-world predictive performance.
    """
    us_futures = fetch_with_fallback(US_FUTURES_TICKER)
    vix = fetch_with_fallback(VIX_TICKER)
    raw = fetch_with_fallback(HSI_TICKER, stooq_ticker="^hsi")
    feat_df = build_features(raw, us_futures=us_futures, vix=vix,
                              ccass_change=0.0, market_sentiment=0.0,
                              stock_sentiment=0.0, ticker=HSI_TICKER, include_macro=True)
    labeled_df = build_nextday_labels(feat_df)

    feature_cols = get_numeric_feature_columns(labeled_df, exclude=LABEL_COLUMNS)
    X_all = labeled_df[feature_cols]
    y_all = labeled_df["next_close_return"]
    y_high_all = labeled_df["next_high_return"]
    y_low_all = labeled_df["next_low_return"]

    cutoff_idx = labeled_df.index.searchsorted(pd.Timestamp(start_date))
    if cutoff_idx >= len(labeled_df):
        raise ValueError(f"start_date {start_date} is beyond available data.")

    results = []
    i = cutoff_idx
    model_close, model_high, model_low = None, None, None

    while i < len(labeled_df):
        # Retrain at the start and every N days thereafter, using only past data
        if (i - cutoff_idx) % retrain_every_n_days == 0:
            X_train = X_all.iloc[:i]
            y_train = y_all.iloc[:i]
            y_high_train = y_high_all.iloc[:i]
            y_low_train = y_low_all.iloc[:i]

            model_close = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
            model_close.fit(X_train, y_train)
            model_high = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
            model_high.fit(X_train, y_high_train)
            model_low = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
            model_low.fit(X_train, y_low_train)

        X_today = X_all.iloc[[i]]
        pred_close_return = float(model_close.predict(X_today)[0])
        pred_high_return = float(model_high.predict(X_today)[0])
        pred_low_return = float(model_low.predict(X_today)[0])

        last_close = float(labeled_df["Close"].iloc[i])
        actual_close_return = float(y_all.iloc[i])

        results.append({
            "date": labeled_df.index[i],
            "last_close": last_close,
            "pred_close": last_close * (1 + pred_close_return),
            "actual_close": last_close * (1 + actual_close_return),
            "pred_high": last_close * (1 + pred_high_return),
            "actual_high": last_close * (1 + float(y_high_all.iloc[i])),
            "pred_low": last_close * (1 + pred_low_return),
            "actual_low": last_close * (1 + float(y_low_all.iloc[i])),
            "directional_hit": int(np.sign(pred_close_return) == np.sign(actual_close_return)),
        })
        i += 1

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_backtest(results: pd.DataFrame, label: str = "HSI") -> dict:
    if results.empty:
        return {"label": label, "status": "NO_DATA"}

    return {
        "label": label,
        "n_days": int(len(results)),
        "date_range": f"{results['date'].min()} to {results['date'].max()}",
        "directional_accuracy": round(float(results["directional_hit"].mean()), 4),
        "mean_abs_error_pct": round(float(
            ((results["pred_close"] - results["actual_close"]).abs() / results["actual_close"]).mean() * 100
        ), 3),
        "rmse": round(float(np.sqrt(((results["pred_close"] - results["actual_close"]) ** 2).mean())), 2),
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_full_backtest(start_date: str = "2026-01-01", mode: str = "insample"):
    """
    mode: 'insample' (fast, optimistic) or 'walkforward' (slow, honest)
    """
    print(f"=== Running {mode} backtest for HSI from {start_date} ===")
    if mode == "insample":
        hsi_results = run_insample_backtest_hsi(start_date)
    elif mode == "walkforward":
        hsi_results = run_walkforward_oos_backtest_hsi(start_date)
    else:
        raise ValueError("mode must be 'insample' or 'walkforward'")

    hsi_results.to_csv(PRED_DIR / "backtest_hsi.csv", index=False)
    hsi_summary = summarize_backtest(hsi_results, "HSI")

    print(f"=== Running in-sample backtest for all stocks from {start_date} ===")
    stock_results = run_insample_backtest_all_stocks(start_date)
    if not stock_results.empty:
        stock_results.to_csv(PRED_DIR / "backtest_stocks.csv", index=False)

    stock_summaries = []
    if not stock_results.empty:
        for ticker, grp in stock_results.groupby("ticker"):
            stock_summaries.append(summarize_backtest(grp, ticker))

    full_summary = {
        "mode": mode,
        "hsi_summary": hsi_summary,
        "stock_summaries": stock_summaries,
    }

    with open(PRED_DIR / "backtest_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2, default=str)

    print(json.dumps(full_summary, indent=2, default=str))
    return hsi_results, stock_results, full_summary


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "insample"
    start_date = sys.argv[2] if len(sys.argv) > 2 else "2026-01-01"
    run_full_backtest(start_date=start_date, mode=mode)
