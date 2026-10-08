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

Both modes now support optional transaction cost + slippage simulation and
Kelly-based position sizing, to estimate NET (after-cost) strategy returns
rather than just raw directional accuracy.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (HSI_TICKER, US_FUTURES_TICKER, VIX_TICKER, LGB_PARAMS,
                     PRED_DIR, STOCK_MODEL_DIR, FEATURE_LIST_PATH,
                     MODEL_CLOSE_Q50_PATH, MODEL_HIGH_PATH, MODEL_LOW_PATH,
                     TRANSACTION_COST, SLIPPAGE_POINTS)
from data_sources import fetch_with_fallback, to_stooq_hk_code
from features import build_features, get_numeric_feature_columns
from labeling import build_nextday_labels, LABEL_COLUMNS
from confidence import load_meta_model, get_signal_confidence
from stock_universe import get_universe
from position_sizer import PositionSizer


# ---------------------------------------------------------------------------
# Cost + position-sizing helpers
# ---------------------------------------------------------------------------

def apply_cost_simulation(df: pd.DataFrame, confidence_col: str = "confidence",
                           regime_col: str = None, use_position_sizing: bool = True) -> pd.DataFrame:
    """
    Adds gross/net strategy return columns to a backtest results DataFrame.
    Requires columns: pred_close_return, actual_close_return, last_close.

    - gross_ret: raw signal * actual return (no costs)
    - position_size: Kelly-derived fraction of capital risked (0 if use_position_sizing=False, uses fixed 1.0)
    - cost: transaction cost + slippage charged whenever the signal changes direction
    - net_ret: position_size * actual_return - cost
    """
    df = df.copy()
    df["signal"] = np.sign(df["pred_close_return"])

    if use_position_sizing:
        sizer = PositionSizer(kelly_fraction=0.25, max_position=1.0, min_edge=0.02)
        sizes = []
        for _, row in df.iterrows():
            # Approximate win-probability from confidence score when available,
            # otherwise fall back to a neutral 0.5 + small edge based on signal direction.
            prob = row[confidence_col] if confidence_col in df.columns and pd.notna(row[confidence_col]) else 0.55
            regime = row[regime_col] if regime_col and regime_col in df.columns else "NEUTRAL"
            size = sizer.compute_size(prob=prob, signal=int(row["signal"]), regime=regime, roll_acc=0.5)
            sizes.append(size if size > 0 else 0.0)
        df["position_size"] = sizes
    else:
        df["position_size"] = df["signal"].abs()  # fixed full-size whenever there's a signal

    df["gross_ret"] = df["signal"] * df["actual_close_return"]

    # Cost charged whenever position changes (entry/exit), proportional to trade size.
    df["sig_change"] = df["signal"].diff().abs().fillna(abs(df["signal"].iloc[0]) if len(df) else 0)
    slippage_pct = (SLIPPAGE_POINTS / df["last_close"]).fillna(0)
    df["cost"] = df["sig_change"] * (TRANSACTION_COST + slippage_pct) * df["position_size"].clip(lower=0.01)

    df["strategy_ret"] = df["position_size"] * df["signal"] * df["actual_close_return"] - df["cost"]
    df["bh_ret"] = df["actual_close_return"]  # Buy & Hold benchmark

    return df


def summarize_cost_adjusted(df: pd.DataFrame) -> dict:
    """Summary stats comparing gross vs net (after-cost) strategy performance."""
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

    # NEW: cost + position-sizing simulation
    results = apply_cost_simulation(results, confidence_col="confidence", use_position_sizing=True)

    return results


# ---------------------------------------------------------------------------
# Mode 1: Fast in-sample backtest (per-stock)
# ---------------------------------------------------------------------------

def run_insample_backtest_stock(ticker: str, start_date: str = "2026-01-01") -> pd.DataFrame:
    model_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_close_q50.txt"
    feat_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_features.json"

    if not model_path.exists() or not feat_path.exists():
        return pd.DataFrame()

    stooq_code = to_stooq_hk_code(ticker)
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

    # NEW: cost simulation (no confidence column available at per-stock level,
    # so position sizing falls back to a neutral default inside apply_cost_simulation)
    results = apply_cost_simulation(results, confidence_col="confidence", use_position_sizing=True)

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

    meta_model = load_meta_model()

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
        confidence = get_signal_confidence(meta_model, X_today)

        last_close = float(labeled_df["Close"].iloc[i])
        actual_close_return = float(y_all.iloc[i])

        results.append({
            "date": labeled_df.index[i],
            "last_close": last_close,
            "pred_close_return": pred_close_return,
            "confidence": confidence,
            "pred_close": last_close * (1 + pred_close_return),
            "actual_close": last_close * (1 + actual_close_return),
            "actual_close_return": actual_close_return,
            "pred_high": last_close * (1 + pred_high_return),
            "actual_high": last_close * (1 + float(y_high_all.iloc[i])),
            "pred_low": last_close * (1 + pred_low_return),
            "actual_low": last_close * (1 + float(y_low_all.iloc[i])),
            "directional_hit": int(np.sign(pred_close_return) == np.sign(actual_close_return)),
        })
        i += 1

    df = pd.DataFrame(results)

    # NEW: cost + position-sizing simulation
    df = apply_cost_simulation(df, confidence_col="confidence", use_position_sizing=True)

    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_backtest(results: pd.DataFrame, label: str = "HSI") -> dict:
    if results.empty:
        return {"label": label, "status": "NO_DATA"}

    summary = {
        "label": label,
        "n_days": int(len(results)),
        "date_range": f"{results['date'].min()} to {results['date'].max()}",
        "directional_accuracy": round(float(results["directional_hit"].mean()), 4),
        "mean_abs_error_pct": round(float(
            ((results["pred_close"] - results["actual_close"]).abs() / results["actual_close"]).mean() * 100
        ), 3),
        "rmse": round(float(np.sqrt(((results["pred_close"] - results["actual_close"]) ** 2).mean())), 2),
    }

    # NEW: merge in cost-adjusted performance stats if available
    if "strategy_ret" in results.columns:
        summary["cost_adjusted"] = summarize_cost_adjusted(results)

    return summary


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
