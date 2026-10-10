"""
Daily backtest runner: simulates every trading day's prediction from
start_date to today, for both HSI and per-stock models, then produces
a full performance report (accuracy, RMSE, calibration) with charts.

Now includes transaction cost + slippage simulation and Kelly-based
position sizing, to report NET (after-cost) strategy performance
alongside raw directional accuracy.

Usage:
    python run_daily_backtest.py insample 2026-01-01
    python run_daily_backtest.py walkforward 2026-01-01

FIX LOG (this revision):
  - PositionSizer(kelly_fraction=0.25, max_position=1.0, min_edge=0.02)
    used kwarg names that never existed on PositionSizer.__init__()
    (real names: kelly_fraction, max_position_pct, min_position_pct_floor,
    kelly_min_edge) — this raised TypeError on every run.
  - sizer.compute_size(prob=..., signal=..., regime=..., roll_acc=...)
    called a method that does not exist at all on PositionSizer. Replaced
    with the module-level compute_position_size(p_up, calibrated_signal,
    signal_confidence, risk_reward_ratio, regime, worth_trading) -> dict
    with key "position_size_pct", matching position_sizer.py's actual
    contract (same one predict.py now uses).
  - p_up is now derived directionally from the confidence column: a
    confidence value is "probability the direction is correct", not raw
    P(up) — for SHORT signals this must be inverted (1 - confidence),
    otherwise Kelly sizing would have been computed backwards for every
    short signal.
  - risk_reward_ratio is now computed directionally from pred_high/
    pred_low against last_close when available (HSI always has these;
    stocks have them only when their high/low models exist on disk);
    falls back to a flagged neutral 1.0 for stock rows missing high/low
    models (see daily_backtest_insample_stock()'s FIX LOG below).
  - daily_backtest_insample_stock() now also loads
    stock_high_model_path(ticker) / stock_low_model_path(ticker) (these
    model artifacts already exist on disk, produced by train_model.py's
    train_stock_models() — they were simply never wired into this backtest
    function before), adding real pred_high/pred_low columns so
    apply_cost_simulation() can compute a genuine risk_reward_ratio for
    stocks too, instead of always using the 1.0 placeholder.
  - Added missing imports: stock_high_model_path, stock_low_model_path
    (from config) — daily_backtest_insample_stock() referenced these
    without importing them, which would have raised NameError.
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
                     DIRECTIONAL_ACC_MIN, TRANSACTION_COST, SLIPPAGE_POINTS,
                     stock_high_model_path, stock_low_model_path)
from data_sources import fetch_with_fallback, to_stooq_hk_code
from features import build_features, get_numeric_feature_columns
from labeling import build_nextday_labels, LABEL_COLUMNS
from confidence import load_meta_model, get_signal_confidence
from stock_universe import get_universe
from position_sizer import compute_position_size


# ---------------------------------------------------------------------------
# Cost + position-sizing helpers (shared logic with backtest.py)
# ---------------------------------------------------------------------------

def apply_cost_simulation(df: pd.DataFrame, confidence_col: str = "confidence",
                           regime_col: str = None, use_position_sizing: bool = True) -> pd.DataFrame:
    """Adds gross/net strategy return columns using Kelly position sizing
    and transaction cost + slippage simulation."""
    df = df.copy()
    df["signal"] = np.sign(df["pred_close_return"])

    has_high_low = "pred_high" in df.columns and "pred_low" in df.columns

    if use_position_sizing:
        sizes = []
        for _, row in df.iterrows():
            confidence = row[confidence_col] if confidence_col in df.columns and pd.notna(row[confidence_col]) else 0.55
            regime = row[regime_col] if regime_col and regime_col in df.columns else "NEUTRAL"
            signal = int(row["signal"])

            if signal > 0:
                calibrated_signal = "LONG"
                p_up_proxy = confidence
            elif signal < 0:
                calibrated_signal = "SHORT"
                p_up_proxy = 1 - confidence
            else:
                calibrated_signal = "觀望"
                p_up_proxy = 0.5

            if has_high_low:
                entry = row["last_close"]
                if signal > 0:
                    reward = row["pred_high"] - entry
                    risk = entry - row["pred_low"]
                else:
                    reward = entry - row["pred_low"]
                    risk = row["pred_high"] - entry
                risk_reward_ratio = (reward / risk) if risk and risk > 0 else 1.0
            else:
                # Per-stock backtest rows without high/low models loaded —
                # flagged neutral fallback, not a genuinely computed ratio.
                risk_reward_ratio = 1.0

            result = compute_position_size(
                p_up=p_up_proxy,
                calibrated_signal=calibrated_signal,
                signal_confidence=confidence,
                risk_reward_ratio=risk_reward_ratio,
                regime=regime,
                worth_trading=True,
            )
            sizes.append(result["position_size_pct"])
        df["position_size"] = sizes
    else:
        df["position_size"] = df["signal"].abs()

    df["gross_ret"] = df["signal"] * df["actual_close_return"]

    df["sig_change"] = df["signal"].diff().abs().fillna(abs(df["signal"].iloc[0]) if len(df) else 0)
    slippage_pct = (SLIPPAGE_POINTS / df["last_close"]).fillna(0)
    df["cost"] = df["sig_change"] * (TRANSACTION_COST + slippage_pct) * df["position_size"].clip(lower=0.01)

    df["strategy_ret"] = df["position_size"] * df["signal"] * df["actual_close_return"] - df["cost"]
    df["bh_ret"] = df["actual_close_return"]

    return df


def summarize_cost_adjusted(df: pd.DataFrame) -> dict:
    if df.empty or "strategy_ret" not in df.columns:
        return {}

    n = len(df)
    gross_cum = float((1 + df["gross_ret"]).prod() - 1)
    net_cum = float((1 + df["strategy_ret"]).prod() - 1)
    bh_cum = float((1 + df["bh_ret"]).prod() - 1)

    net_mean = df["strategy_ret"].mean()
    net_std = df["strategy_ret"].std()
    sharpe = float(net_mean / net_std * np.sqrt(252)) if net_std > 0 else 0.0

    total_cost = float(df["cost"].sum())
    n_trades = int((df["sig_change"] > 0).sum())

    return {
        "n_days": n,
        "n_trades": n_trades,
        "gross_cumulative_return_pct": round(gross_cum * 100, 2),
        "net_cumulative_return_pct": round(net_cum * 100, 2),
        "buy_hold_cumulative_return_pct": round(bh_cum * 100, 2),
        "total_cost_pct": round(total_cost * 100, 2),
        "net_sharpe_annualized": round(sharpe, 3),
        "avg_position_size": round(float(df["position_size"].mean()), 3),
    }


# ---------------------------------------------------------------------------
# HSI in-sample daily backtest
# ---------------------------------------------------------------------------

def daily_backtest_insample_hsi(start_date: str) -> pd.DataFrame:
    us_futures = fetch_with_fallback(US_FUTURES_TICKER)
    vix = fetch_with_fallback(VIX_TICKER)
    raw = fetch_with_fallback(HSI_TICKER, stooq_ticker="^hsi")

    feat_df = build_features(raw, us_futures=us_futures, vix=vix,
                              ccass_change=0.0, market_sentiment=0.0,
                              stock_sentiment=0.0, ticker=HSI_TICKER, include_macro=True)
    labeled_df = build_nextday_labels(feat_df)

    with open(FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)
    for col in feature_cols:
        if col not in labeled_df.columns:
            labeled_df[col] = 0
    X_all = labeled_df[feature_cols].astype(float)

    mask = labeled_df.index >= pd.Timestamp(start_date)
    subset = labeled_df[mask]
    X_subset = X_all[mask]

    if subset.empty:
        raise ValueError(f"No data available from {start_date} onwards.")

    m_close = lgb.Booster(model_file=str(MODEL_CLOSE_Q50_PATH))
    m_high = lgb.Booster(model_file=str(MODEL_HIGH_PATH))
    m_low = lgb.Booster(model_file=str(MODEL_LOW_PATH))
    meta_model = load_meta_model()

    pred_close_return = m_close.predict(X_subset)
    pred_high_return = m_high.predict(X_subset)
    pred_low_return = m_low.predict(X_subset)

    confidences = [get_signal_confidence(meta_model, X_subset.iloc[[i]]) for i in range(len(X_subset))]

    df = pd.DataFrame({
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

    df["pred_close"] = df["last_close"] * (1 + df["pred_close_return"])
    df["actual_close"] = df["last_close"] * (1 + df["actual_close_return"])
    df["pred_high"] = df["last_close"] * (1 + df["pred_high_return"])
    df["actual_high"] = df["last_close"] * (1 + df["actual_high_return"])
    df["pred_low"] = df["last_close"] * (1 + df["pred_low_return"])
    df["actual_low"] = df["last_close"] * (1 + df["actual_low_return"])
    df["directional_hit"] = (np.sign(df["pred_close_return"]) == np.sign(df["actual_close_return"])).astype(int)

    df = apply_cost_simulation(df, confidence_col="confidence", use_position_sizing=True)
    return df


# ---------------------------------------------------------------------------
# Per-stock in-sample daily backtest
# ---------------------------------------------------------------------------

def daily_backtest_insample_stock(ticker: str, start_date: str) -> pd.DataFrame:
    """
    In-sample backtest for a single stock (constituent or watchlist ticker).
    Loads close AND high/low models (when available) so apply_cost_simulation()
    can compute a genuine direction-aware risk_reward_ratio, instead of
    always falling back to a flagged neutral 1.0.
    """
    raw = fetch_with_fallback(ticker, stooq_ticker=to_stooq_hk_code(ticker))
    if raw is None or len(raw) < 100:
        return pd.DataFrame()

    feat_df = build_features(raw, us_futures=None, vix=None,
                              ccass_change=0.0, market_sentiment=0.0,
                              stock_sentiment=0.0, ticker=ticker, include_macro=False)
    labeled_df = build_nextday_labels(feat_df)
    labeled_df = labeled_df[labeled_df.index >= pd.to_datetime(start_date)]
    if len(labeled_df) == 0:
        return pd.DataFrame()

    feat_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_features.json"
    model_close_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_close_q50.txt"
    if not feat_path.exists() or not model_close_path.exists():
        return pd.DataFrame()

    with open(feat_path) as f:
        feature_cols = json.load(f)

    X = labeled_df.reindex(columns=feature_cols, fill_value=0).astype(float)

    model_close = lgb.Booster(model_file=str(model_close_path))
    pred_close_return = model_close.predict(X)

    high_path = stock_high_model_path(ticker)
    low_path = stock_low_model_path(ticker)
    has_high_low = high_path.exists() and low_path.exists()

    if has_high_low:
        model_high = lgb.Booster(model_file=str(high_path))
        model_low = lgb.Booster(model_file=str(low_path))
        pred_high_return = model_high.predict(X)
        pred_low_return = model_low.predict(X)
    else:
        print(f"run_daily_backtest: high/low models not found for {ticker} — "
              f"risk_reward_ratio for this ticker will fall back to the neutral 1.0 default.")
        pred_high_return = None
        pred_low_return = None

    last_close = labeled_df["Close"].values if "Close" in labeled_df.columns else raw.loc[labeled_df.index, "Close"].values

    df = pd.DataFrame({
        "date": labeled_df.index,
        "ticker": ticker,
        "last_close": last_close,
        "pred_close_return": pred_close_return,
        "actual_close_return": labeled_df["next_close_return"].values,
        "confidence": 0.55,  # per-stock backtest has no meta-confidence model wired in; neutral default
    })

    if has_high_low:
        df["pred_high"] = last_close * (1 + pred_high_return)
        df["pred_low"] = last_close * (1 + pred_low_return)

    return df


# ---------------------------------------------------------------------------
# Performance summary (with optional pred_high/pred_low coverage + cost stats)
# ---------------------------------------------------------------------------

def summarize_performance(df: pd.DataFrame, label: str = "HSI") -> dict:
    if df.empty:
        return {"label": label, "status": "NO_DATA"}

    rmse = float(np.sqrt(((df["pred_close"] - df["actual_close"]) ** 2).mean()))
    mae_pct = float(((df["pred_close"] - df["actual_close"]).abs() / df["actual_close"]).mean() * 100)

    if "pred_high" in df.columns and "pred_low" in df.columns:
        high_coverage = ((df["actual_close"] <= df["pred_high"]) &
                          (df["actual_close"] >= df["pred_low"])).mean()
        coverage_pct = round(float(high_coverage) * 100, 1)
    else:
        coverage_pct = None

    summary = {
        "label": label,
        "n_days": int(len(df)),
        "date_range": f"{df['date'].min()} to {df['date'].max()}",
        "directional_accuracy": round(float(df["directional_hit"].mean()), 4),
        "rmse": round(rmse, 2),
        "mean_abs_error_pct": round(mae_pct, 3),
        "pred_range_coverage_pct": coverage_pct,
        "rolling_5d_accuracy_latest": round(float(df["directional_hit"].tail(5).mean()), 4) if len(df) >= 5 else None,
        "below_random_warning": bool(df["directional_hit"].mean() < DIRECTIONAL_ACC_MIN),
    }

    if "strategy_ret" in df.columns:
        summary["cost_adjusted"] = summarize_cost_adjusted(df)

    return summary


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_chart(df: pd.DataFrame, output_path, label: str = "HSI"):
    if df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df["date"], df["actual_close"], label="Actual Close", color="black", linewidth=1.2)
    axes[0].plot(df["date"], df["pred_close"], label="Predicted Close", color="blue", linestyle="--", linewidth=1)
    if "pred_low" in df.columns and "pred_high" in df.columns:
        axes[0].fill_between(df["date"], df["pred_low"], df["pred_high"], alpha=0.15, label="Pred Range")
    axes[0].set_title(f"{label} — Predicted vs Actual Close")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if "strategy_ret" in df.columns:
        cum_net = (1 + df["strategy_ret"]).cumprod() - 1
        cum_gross = (1 + df["gross_ret"]).cumprod() - 1
        cum_bh = (1 + df["bh_ret"]).cumprod() - 1
        axes[1].plot(df["date"], cum_net * 100, label="Net Strategy Return (after cost)", color="green")
        axes[1].plot(df["date"], cum_gross * 100, label="Gross Strategy Return (no cost)", color="orange", linestyle="--")
        axes[1].plot(df["date"], cum_bh * 100, label="Buy & Hold", color="gray", linestyle=":")
        axes[1].set_title(f"{label} — Cumulative Return Comparison (%)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run(mode: str = "insample", start_date: str = "2026-01-01"):
    print(f"=== 每日回測 [{mode}模式] 由 {start_date} 至今 ===\n")

    print(">>> HSI 大盤回測中...")
    if mode == "insample":
        hsi_df = daily_backtest_insample_hsi(start_date)
    else:
        from backtest import run_walkforward_oos_backtest_hsi
        hsi_df = run_walkforward_oos_backtest_hsi(start_date)

    hsi_csv_path = PRED_DIR / f"daily_backtest_hsi_{mode}.csv"
    hsi_df.to_csv(hsi_csv_path, index=False)
    hsi_df.to_csv(PRED_DIR / "backtest_hsi.csv", index=False)
    hsi_summary = summarize_performance(hsi_df, "HSI")
    print(f"HSI回測完成: {json.dumps(hsi_summary, indent=2, default=str)}")

    chart_path = PRED_DIR / f"daily_backtest_hsi_{mode}_chart.png"
    generate_chart(hsi_df, chart_path, "HSI")

    print("\n>>> 個股回測中 (只做in-sample，速度考量)...")
    universe = get_universe()
    all_stock_rows = []
    stock_summaries = []
    for ticker in universe:
        try:
            df = daily_backtest_insample_stock(ticker, start_date)
            if df.empty:
                continue
            all_stock_rows.append(df)
            summary = summarize_performance(df, ticker)
            stock_summaries.append(summary)
            print(f"  {ticker}: 方向準確度 = {summary['directional_accuracy']}")
        except Exception as e:
            print(f"  {ticker} 回測失敗: {e}")
            continue

    if all_stock_rows:
        stock_df = pd.concat(all_stock_rows, ignore_index=True)
        stock_df.to_csv(PRED_DIR / "daily_backtest_stocks.csv", index=False)
        stock_df.to_csv(PRED_DIR / "backtest_stocks.csv", index=False)

    full_report = {
        "mode": mode,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hsi_summary": hsi_summary,
        "stock_summaries": stock_summaries,
    }

    summary_path = PRED_DIR / f"daily_backtest_summary_{mode}.json"
    with open(summary_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    print(f"\n=== 完整報告 ===")
    print(json.dumps(full_report, indent=2, default=str))
    print(f"\n已儲存: {hsi_csv_path}, {chart_path}, {summary_path}")

    return full_report


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "insample"
    start_date = sys.argv[2] if len(sys.argv) > 2 else "2026-01-01"
    run(mode=mode, start_date=start_date)
