%%writefile src/run_daily_backtest.py
"""
Daily backtest runner: simulates every trading day's prediction from
start_date to today, for both HSI and per-stock models, then produces
a full performance report (accuracy, RMSE, calibration) with charts.

Usage:
    python run_daily_backtest.py insample 2026-01-01
    python run_daily_backtest.py walkforward 2026-01-01
"""

import sys
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from config import (HSI_TICKER, US_FUTURES_TICKER, VIX_TICKER, LGB_PARAMS,
                     PRED_DIR, STOCK_MODEL_DIR, FEATURE_LIST_PATH,
                     MODEL_CLOSE_Q50_PATH, MODEL_HIGH_PATH, MODEL_LOW_PATH,
                     HSI_PROB_MODEL_PATH, DIRECTIONAL_ACC_MIN)
from data_sources import fetch_with_fallback
from features import build_features, get_numeric_feature_columns
from labeling import build_nextday_labels, LABEL_COLUMNS
from confidence import load_meta_model, get_signal_confidence
from probability_model import load_probability_model, predict_probability_up
from regime import detect_regime
from stock_universe import get_universe


# ---------------------------------------------------------------------------
# Build full labeled dataset once (reused across all days)
# ---------------------------------------------------------------------------

def build_hsi_dataset():
    us_futures = fetch_with_fallback(US_FUTURES_TICKER)
    vix = fetch_with_fallback(VIX_TICKER)
    raw = fetch_with_fallback(HSI_TICKER, stooq_ticker="^hsi")
    feat_df = build_features(raw, us_futures=us_futures, vix=vix,
                              ccass_change=0.0, market_sentiment=0.0,
                              stock_sentiment=0.0, ticker=HSI_TICKER, include_macro=True)
    labeled_df = build_nextday_labels(feat_df)
    return labeled_df


# ---------------------------------------------------------------------------
# Mode 1: In-sample daily backtest (fast, uses currently saved trained models)
# ---------------------------------------------------------------------------

def daily_backtest_insample_hsi(start_date: str) -> pd.DataFrame:
    labeled_df = build_hsi_dataset()

    with open(FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)
    for col in feature_cols:
        if col not in labeled_df.columns:
            labeled_df[col] = 0
    X_all = labeled_df[feature_cols].astype(float)

    m_close = lgb.Booster(model_file=str(MODEL_CLOSE_Q50_PATH))
    m_high = lgb.Booster(model_file=str(MODEL_HIGH_PATH))
    m_low = lgb.Booster(model_file=str(MODEL_LOW_PATH))
    meta_model = load_meta_model()
    prob_clf = load_probability_model(HSI_PROB_MODEL_PATH)

    mask = labeled_df.index >= pd.Timestamp(start_date)
    subset = labeled_df[mask]
    X_subset = X_all[mask]

    if subset.empty:
        raise ValueError(f"No data from {start_date} onwards.")

    rows = []
    for i in range(len(subset)):
        row_X = X_subset.iloc[[i]]
        date = subset.index[i]
        last_close = float(subset["Close"].iloc[i])

        pred_close_return = float(m_close.predict(row_X)[0])
        pred_high_return = float(m_high.predict(row_X)[0])
        pred_low_return = float(m_low.predict(row_X)[0])
        actual_close_return = float(subset["next_close_return"].iloc[i])
        actual_high_return = float(subset["next_high_return"].iloc[i])
        actual_low_return = float(subset["next_low_return"].iloc[i])

        confidence = get_signal_confidence(meta_model, row_X)
        p_up = predict_probability_up(prob_clf, row_X)

        rows.append({
            "date": date,
            "last_close": last_close,
            "pred_close": last_close * (1 + pred_close_return),
            "actual_close": last_close * (1 + actual_close_return),
            "pred_high": last_close * (1 + pred_high_return),
            "actual_high": last_close * (1 + actual_high_return),
            "pred_low": last_close * (1 + pred_low_return),
            "actual_low": last_close * (1 + actual_low_return),
            "p_up": p_up,
            "confidence": confidence,
            "directional_hit": int(np.sign(pred_close_return) == np.sign(actual_close_return)),
            "abs_error_pct": abs(pred_close_return - actual_close_return) * 100,
        })

    return pd.DataFrame(rows)


def daily_backtest_insample_stock(ticker: str, start_date: str) -> pd.DataFrame:
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
    pred_returns = model.predict(X_subset)

    df = pd.DataFrame({
        "date": subset.index,
        "ticker": ticker,
        "last_close": subset["Close"].values,
        "pred_close": subset["Close"].values * (1 + pred_returns),
        "actual_close": subset["Close"].values * (1 + subset["next_close_return"].values),
    })
    df["directional_hit"] = (
        np.sign(pred_returns) == np.sign(subset["next_close_return"].values)
    ).astype(int)
    return df


# ---------------------------------------------------------------------------
# Mode 2: Walk-forward daily backtest (honest, retrains periodically)
# ---------------------------------------------------------------------------

def daily_backtest_walkforward_hsi(start_date: str, retrain_every_n_days: int = 10) -> pd.DataFrame:
    labeled_df = build_hsi_dataset()
    feature_cols = get_numeric_feature_columns(labeled_df, exclude=LABEL_COLUMNS)
    X_all = labeled_df[feature_cols]
    y_close_all = labeled_df["next_close_return"]
    y_high_all = labeled_df["next_high_return"]
    y_low_all = labeled_df["next_low_return"]

    cutoff_idx = labeled_df.index.searchsorted(pd.Timestamp(start_date))
    if cutoff_idx >= len(labeled_df):
        raise ValueError(f"start_date {start_date} is beyond available data.")

    rows = []
    model_close, model_high, model_low, prob_clf = None, None, None, None
    i = cutoff_idx

    while i < len(labeled_df):
        if (i - cutoff_idx) % retrain_every_n_days == 0:
            X_train = X_all.iloc[:i]
            y_close_train = y_close_all.iloc[:i]
            y_high_train = y_high_all.iloc[:i]
            y_low_train = y_low_all.iloc[:i]

            model_close = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
            model_close.fit(X_train, y_close_train)
            model_high = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
            model_high.fit(X_train, y_high_train)
            model_low = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
            model_low.fit(X_train, y_low_train)

            prob_clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                                            max_depth=5, verbosity=-1)
            prob_clf.fit(X_train, (y_close_train > 0).astype(int))

        X_today = X_all.iloc[[i]]
        last_close = float(labeled_df["Close"].iloc[i])

        pred_close_return = float(model_close.predict(X_today)[0])
        pred_high_return = float(model_high.predict(X_today)[0])
        pred_low_return = float(model_low.predict(X_today)[0])

        proba = prob_clf.predict_proba(X_today)[0]
        classes = list(prob_clf.classes_)
        p_up = float(proba[classes.index(1)]) if 1 in classes else 0.5

        actual_close_return = float(y_close_all.iloc[i])
        actual_high_return = float(y_high_all.iloc[i])
        actual_low_return = float(y_low_all.iloc[i])

        rows.append({
            "date": labeled_df.index[i],
            "last_close": last_close,
            "pred_close": last_close * (1 + pred_close_return),
            "actual_close": last_close * (1 + actual_close_return),
            "pred_high": last_close * (1 + pred_high_return),
            "actual_high": last_close * (1 + actual_high_return),
            "pred_low": last_close * (1 + pred_low_return),
            "actual_low": last_close * (1 + actual_low_return),
            "p_up": p_up,
            "directional_hit": int(np.sign(pred_close_return) == np.sign(actual_close_return)),
            "abs_error_pct": abs(pred_close_return - actual_close_return) * 100,
        })
        i += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Performance report
# ---------------------------------------------------------------------------

def summarize_performance(df: pd.DataFrame, label: str = "HSI") -> dict:
    if df.empty:
        return {"label": label, "status": "NO_DATA"}

    rmse = float(np.sqrt(((df["pred_close"] - df["actual_close"]) ** 2).mean()))
    mae_pct = float(((df["pred_close"] - df["actual_close"]).abs() / df["actual_close"]).mean() * 100)

    high_coverage = ((df["actual_close"] <= df["pred_high"]) &
                      (df["actual_close"] >= df["pred_low"])).mean()

    return {
        "label": label,
        "n_days": int(len(df)),
        "date_range": f"{df['date'].min()} to {df['date'].max()}",
        "directional_accuracy": round(float(df["directional_hit"].mean()), 4),
        "rmse": round(rmse, 2),
        "mean_abs_error_pct": round(mae_pct, 3),
        "pred_range_coverage_pct": round(float(high_coverage) * 100, 1),
        "rolling_5d_accuracy_latest": round(float(df["directional_hit"].tail(5).mean()), 4) if len(df) >= 5 else None,
        "below_random_warning": bool(df["directional_hit"].mean() < DIRECTIONAL_ACC_MIN),
    }


def plot_backtest_chart(df: pd.DataFrame, save_path, title: str = "HSI Backtest"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(df["date"], df["actual_close"], label="Actual Close", marker="o", markersize=3)
    axes[0].plot(df["date"], df["pred_close"], label="Predicted Close", marker="x", markersize=3)
    axes[0].fill_between(df["date"], df["pred_low"], df["pred_high"], alpha=0.15, label="Pred Range")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=45)

    rolling_acc = df["directional_hit"].rolling(10, min_periods=1).mean()
    axes[1].plot(df["date"], rolling_acc, label="Rolling 10-day Directional Accuracy", color="green")
    axes[1].axhline(y=0.5, color="gray", linestyle="--", label="Random (50%)")
    axes[1].axhline(y=DIRECTIONAL_ACC_MIN, color="red", linestyle="--", label=f"Min threshold ({DIRECTIONAL_ACC_MIN})")
    axes[1].set_title("Rolling Directional Accuracy")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run(mode: str = "insample", start_date: str = "2026-01-01"):
    print(f"=== 每日回測 [{mode}模式] 由 {start_date} 至今 ===\n")

    print(">>> HSI 大盤回測中...")
    if mode == "insample":
        hsi_df = daily_backtest_insample_hsi(start_date)
    elif mode == "walkforward":
        hsi_df = daily_backtest_walkforward_hsi(start_date)
    else:
        raise ValueError("mode must be 'insample' or 'walkforward'")

    hsi_csv_path = PRED_DIR / f"daily_backtest_hsi_{mode}.csv"
    hsi_df.to_csv(hsi_csv_path, index=False)
    hsi_summary = summarize_performance(hsi_df, "HSI")

    chart_path = PRED_DIR / f"daily_backtest_hsi_{mode}_chart.png"
    plot_backtest_chart(hsi_df, chart_path, title=f"HSI Daily Backtest ({mode})")

    print(f"HSI回測完成: {hsi_summary}\n")

    print(">>> 個股回測中 (只做in-sample，速度考量)...")
    universe = get_universe()
    stock_summaries = []
    all_stock_rows = []

    for ticker in universe:
        try:
            df = daily_backtest_insample_stock(ticker, start_date)
            if not df.empty:
                all_stock_rows.append(df)
                stock_summaries.append(summarize_performance(df, ticker))
                print(f"  {ticker}: 方向準確度 = {stock_summaries[-1]['directional_accuracy']}")
        except Exception as e:
            print(f"  {ticker} 回測失敗: {e}")
            continue

    if all_stock_rows:
        stock_df = pd.concat(all_stock_rows, ignore_index=True)
        stock_df.to_csv(PRED_DIR / "daily_backtest_stocks.csv", index=False)

    full_report = {
        "mode": mode,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hsi_summary": hsi_summary,
        "stock_summaries": stock_summaries,
    }

    report_path = PRED_DIR / f"daily_backtest_summary_{mode}.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    print("\n=== 完整報告 ===")
    print(json.dumps(full_report, indent=2, default=str))
    print(f"\n已儲存: {hsi_csv_path}, {chart_path}, {report_path}")

    if hsi_summary.get("below_random_warning"):
        print("\n⚠️ 警告: HSI方向準確度低於隨機水平門檻，建議檢視模型或重新訓練。")

    return hsi_df, full_report


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "insample"
    start_date = sys.argv[2] if len(sys.argv) > 2 else "2026-01-01"
    run(mode=mode, start_date=start_date)
