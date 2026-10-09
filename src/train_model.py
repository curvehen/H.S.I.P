"""
TRAINING SCRIPT — run manually in Google Colab only.
GitHub Actions never calls this file; it only runs predict.py.

Full pipeline:
1. Check walk-forward retrain trigger (calendar + drift) — informational only
2. Fetch HSI + cross-market + macro (Stock Connect, GARCH, ADR) data
3. Feature engineering (candlesticks, technicals, CCASS, news, macro)
4. Next-day labeling (close/high/low returns)
5. Optuna hyperparameter search (Purged K-Fold objective)
6. Expanding-window walk-forward validation report (using tuned params)
7. Train final Ensemble model (LightGBM + RandomForest + Ridge) for close
8. Train standalone quantile models (q10/q50/q90) for close band
9. Train direct High / Low regressors
10. Train Meta-Labeling (confidence) model
11. Train HSI probability model (P-up/P-down)
12. Generate Purged-K-Fold out-of-fold predictions + regime tags, fit
    regime-specific F1-optimized trading thresholds (ENH#1)
13. Initialize rolling-Brier-score dynamic ensemble weights file (ENH#2 seed)
14. Train per-stock bottom-up models for ALL tracked tickers
    (HSI constituents + watchlist, e.g. 1211.HK / 0968.HK)
15. Save everything + metrics.json + mark_trained_today()
"""

import json
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

from config import (MODEL_CLOSE_Q10_PATH, MODEL_CLOSE_Q50_PATH, MODEL_CLOSE_Q90_PATH,
                     MODEL_HIGH_PATH, MODEL_LOW_PATH, FEATURE_LIST_PATH, METRICS_PATH,
                     STOCK_MODEL_DIR, HSI_TICKER, US_FUTURES_TICKER, VIX_TICKER,
                     LGB_PARAMS, MODEL_DIR, PURGE_EMBARGO_DAYS)
from data_sources import fetch_with_fallback, to_stooq_hk_code
from features import build_features, get_numeric_feature_columns
from labeling import build_nextday_labels, LABEL_COLUMNS
from confidence import build_meta_labels, train_meta_model
from news_sentiment import get_daily_market_sentiment, get_stock_sentiment
from ccass_scraper import get_ccass_change
from stock_universe import get_universe, get_all_tracked_tickers, is_hsi_constituent
from ensemble_model import HSIEnsembleModel
from optuna_tuning import run_optuna_search, purged_kfold_indices
from walk_forward import should_retrain, mark_trained_today, expanding_window_validation

# NOTE: the following two modules implement features discussed and designed
# earlier in this project but have not yet been written as standalone files.
# This script calls their intended public interface; it will raise
# ImportError until those files exist. See the confirmation note at the end
# of this message.
from threshold_calibrator import fit_and_save_regime_thresholds      # ENH#1 (TODO: not yet delivered)
from dynamic_ensemble_weighter import init_dynamic_weights           # ENH#2 (TODO: not yet delivered)


DYNAMIC_WEIGHTS_SEED_PATH = MODEL_DIR / "dynamic_ensemble_weights.json"
REGIME_THRESHOLDS_PATH = MODEL_DIR / "regime_thresholds.json"


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_labeled_hsi_dataset():
    """Fetches HSI + cross-market data, builds features, applies next-day labeling."""
    us_futures = fetch_with_fallback(US_FUTURES_TICKER)
    vix = fetch_with_fallback(VIX_TICKER)
    market_sentiment = get_daily_market_sentiment()

    raw = fetch_with_fallback(HSI_TICKER, stooq_ticker="^hsi")
    feat_df = build_features(raw, us_futures=us_futures, vix=vix,
                              ccass_change=0.0, market_sentiment=market_sentiment,
                              stock_sentiment=0.0, ticker=HSI_TICKER, include_macro=True)
    labeled_df = build_nextday_labels(feat_df)
    return labeled_df


# ---------------------------------------------------------------------------
# Regime tagging (vectorized version of regime.py's row-wise detect_regime,
# needed here to tag an entire historical dataframe for threshold
# calibration — regime.py itself only evaluates the single latest row).
# ---------------------------------------------------------------------------

def compute_regime_series(feat_df: pd.DataFrame) -> pd.Series:
    """Vectorized BULL/BEAR/NEUTRAL tagging across full history, using the
    same Close vs MA20/MA60 logic as regime.detect_regime()."""
    close, ma20, ma60 = feat_df["Close"], feat_df["MA20"], feat_df["MA60"]
    regime = pd.Series("NEUTRAL", index=feat_df.index)
    bull_mask = (close > ma60) & (ma20 > ma60)
    bear_mask = (close < ma60) & (ma20 < ma60)
    regime[bull_mask] = "BULL"
    regime[bear_mask] = "BEAR"
    return regime


# ---------------------------------------------------------------------------
# Out-of-fold predictions for regime-threshold calibration (ENH#1)
# ---------------------------------------------------------------------------

def generate_oof_predictions(X: pd.DataFrame, y_close: pd.Series, feat_df: pd.DataFrame,
                              best_params: dict, n_splits: int = 5) -> pd.DataFrame:
    """
    Builds out-of-fold (Purged K-Fold) primary return predictions AND
    out-of-fold meta-confidence scores, so that threshold calibration is
    evaluated on predictions the model never saw during its own fold's
    training — mirroring live inference honestly instead of calibrating
    thresholds on in-sample fitted values (which would be overly optimistic).

    Each fold trains its own throwaway primary regressor + meta classifier
    on that fold's training split only; neither model is persisted, they
    exist purely to produce honest OOF confidence/return estimates.
    """
    oof_pred = pd.Series(index=X.index, dtype=float)
    oof_confidence = pd.Series(index=X.index, dtype=float)

    for train_idx, test_idx in purged_kfold_indices(len(X), n_splits=n_splits, embargo=PURGE_EMBARGO_DAYS):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y_close.iloc[train_idx]

        fold_model = lgb.LGBMRegressor(**best_params)
        fold_model.fit(X_train, y_train)
        fold_pred = fold_model.predict(X_test)
        oof_pred.iloc[test_idx] = fold_pred

        # Fold-local meta-label + meta-model, purely for honest OOF confidence
        fold_meta_labels = build_meta_labels(
            pred_returns=pd.Series(fold_model.predict(X_train), index=X_train.index),
            actual_returns=y_train)
        fold_meta_model = train_meta_model(X_train, fold_meta_labels)
        if fold_meta_model is not None:
            oof_confidence.iloc[test_idx] = fold_meta_model.predict_proba(X_test)[:, 1]
        else:
            oof_confidence.iloc[test_idx] = 0.5  # neutral fallback if a fold has no valid labels

    oof_df = pd.DataFrame({
        "date": feat_df.index,
        "regime": compute_regime_series(feat_df).values,
        "actual_return": y_close.values,
        "pred_return": oof_pred.values,
        "confidence": oof_confidence.values,
    }).dropna(subset=["pred_return"])

    return oof_df


# ---------------------------------------------------------------------------
# Per-stock bottom-up training
# ---------------------------------------------------------------------------

def train_stock_model(ticker: str):
    """Trains and saves a standalone close-return model for one ticker.
    Called for every ticker returned by get_all_tracked_tickers() — this
    includes both HSI constituents (used for bottom-up aggregation) AND
    watchlist-only tickers like 1211.HK/0968.HK (standalone display only).
    is_hsi_constituent() is NOT checked here: every tracked ticker gets a
    model regardless of its role, aggregation-vs-display is decided later
    at prediction time in predict.py."""
    try:
        stooq_code = to_stooq_hk_code(ticker)
        raw = fetch_with_fallback(ticker, stooq_ticker=stooq_code)
        if raw is None or len(raw) < 250:
            print(f"TRAIN_STOCK[{ticker}]: insufficient data ({0 if raw is None else len(raw)} rows) — skipped.")
            return None

        ccass_change = get_ccass_change(ticker)
        stock_sentiment = get_stock_sentiment(ticker)

        feat_df = build_features(raw, us_futures=None, vix=None,
                                  ccass_change=ccass_change,
                                  market_sentiment=get_daily_market_sentiment(),
                                  stock_sentiment=stock_sentiment,
                                  ticker=ticker, include_macro=False)
        labeled_df = build_nextday_labels(feat_df)

        feature_cols = get_numeric_feature_columns(labeled_df)
        X = labeled_df[feature_cols]
        y = labeled_df["next_close_return"]

        split = int(len(X) * 0.85)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        model = HSIEnsembleModel()
        model.fit(X_train, y_train, X_val, y_val)
        model.save(STOCK_MODEL_DIR / ticker.replace(".", "_"))

        val_pred = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
        directional_acc = float(np.mean(np.sign(val_pred) == np.sign(y_val)))

        print(f"TRAIN_STOCK[{ticker}]: RMSE={rmse:.5f}, DirAcc={directional_acc:.3f}, "
              f"is_constituent={is_hsi_constituent(ticker)}")

        return {
            "ticker": ticker,
            "is_constituent": is_hsi_constituent(ticker),
            "rmse": rmse,
            "directional_accuracy": directional_acc,
            "n_train": len(X_train),
            "n_val": len(X_val),
        }
    except Exception as e:
        print(f"TRAIN_STOCK[{ticker}]: failed — {e}")
        return None


def train_all_stock_models() -> list:
    """Iterates get_all_tracked_tickers() — HSI constituents + watchlist
    combined — training one model per ticker. get_universe()'s weights are
    never consulted here; this loop is agnostic to index-weight, it simply
    trains every ticker that predict.py will later need a model for."""
    results = []
    for ticker in get_all_tracked_tickers():
        result = train_stock_model(ticker)
        if result is not None:
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(f"TRAIN_MODEL run started: {datetime.datetime.utcnow().isoformat()}Z")
    print("=" * 70)

    trigger = should_retrain()
    print(f"Retrain trigger check (informational — this script runs regardless): {trigger}")

    # 1. Dataset construction
    print("\n[1/10] Building labeled HSI dataset...")
    labeled_df = build_labeled_hsi_dataset()
    feature_cols = get_numeric_feature_columns(labeled_df)
    X = labeled_df[feature_cols]
    y_close = labeled_df["next_close_return"]
    y_high = labeled_df["next_high_return"]
    y_low = labeled_df["next_low_return"] if "next_low_return" in labeled_df.columns else None

    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  -> {len(labeled_df)} rows, {len(feature_cols)} features.")

    # 2. Optuna hyperparameter search
    print("\n[2/10] Running Optuna hyperparameter search...")
    best_params = run_optuna_search(X, y_close)
    print(f"  -> Best params: {best_params}")

    # 3. Walk-forward expanding-window validation report (diagnostic only)
    print("\n[3/10] Running expanding-window walk-forward validation...")
    wf_report = expanding_window_validation(X, y_close, params=best_params)
    print(f"  -> Walk-forward avg RMSE: {wf_report.get('avg_rmse')}, "
          f"avg DirAcc: {wf_report.get('avg_directional_accuracy')}")

    # 4. Chronological train/val split for final models
    split = int(len(X) * 0.85)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_close_train, y_close_val = y_close.iloc[:split], y_close.iloc[split:]
    y_high_train, y_high_val = y_high.iloc[:split], y_high.iloc[split:]

    # 5. Final ensemble model (close)
    print("\n[4/10] Training final Ensemble model (close)...")
    ensemble = HSIEnsembleModel()
    ensemble.fit(X_train, y_close_train, X_val, y_close_val)
    ensemble.save(MODEL_DIR / "hsi_ensemble_close")
    ensemble_rmse = float(np.sqrt(np.mean((ensemble.predict(X_val) - y_close_val) ** 2)))
    print(f"  -> Ensemble close RMSE: {ensemble_rmse:.5f}")

    # 6. Quantile models (q10/q50/q90) for close
    print("\n[5/10] Training quantile models (close)...")
    quantile_models = {}
    for q, path in [(0.10, MODEL_CLOSE_Q10_PATH), (0.50, MODEL_CLOSE_Q50_PATH), (0.90, MODEL_CLOSE_Q90_PATH)]:
        q_params = {**best_params, "objective": "quantile", "alpha": q}
        q_model = lgb.LGBMRegressor(**q_params)
        q_model.fit(X_train, y_close_train)
        q_model.booster_.save_model(str(path))
        quantile_models[q] = q_model
    print("  -> Quantile models saved.")

    # 7. Direct High / Low regressors
    print("\n[6/10] Training High/Low regressors...")
    high_model = lgb.LGBMRegressor(**best_params)
    high_model.fit(X_train, y_high_train)
    high_model.booster_.save_model(str(MODEL_HIGH_PATH))
    high_rmse = float(np.sqrt(np.mean((high_model.predict(X_val) - y_high_val) ** 2)))

    low_rmse = None
    if y_low is not None:
        y_low_train, y_low_val = y_low.iloc[:split], y_low.iloc[split:]
        low_model = lgb.LGBMRegressor(**best_params)
        low_model.fit(X_train, y_low_train)
        low_model.booster_.save_model(str(MODEL_LOW_PATH))
        low_rmse = float(np.sqrt(np.mean((low_model.predict(X_val) - y_low_val) ** 2)))
    print(f"  -> High RMSE: {high_rmse:.5f}, Low RMSE: {low_rmse}")

    # 8. Meta-labeling (confidence) model — trained on full-sample in-sample
    # ensemble predictions (final deployed model, not OOF — OOF variant is
    # used separately below purely for threshold calibration honesty).
    print("\n[7/10] Training meta-labeling (confidence) model...")
    full_pred = ensemble.predict(X)
    meta_labels = build_meta_labels(pred_returns=pd.Series(full_pred, index=X.index), actual_returns=y_close)
    meta_model = train_meta_model(X, meta_labels)
    print("  -> Meta-labeling model trained and saved.")

    # 9. HSI probability model (P-up/P-down)
    print("\n[8/10] Training probability model...")
    from probability_model import train_probability_model
    prob_model = train_probability_model(X_train, y_close_train)
    print("  -> Probability model trained and saved.")

    # 10. Regime-threshold calibration (ENH#1) + dynamic ensemble weight seed (ENH#2)
    print("\n[9/10] Generating OOF predictions for regime threshold calibration...")
    oof_df = generate_oof_predictions(X, y_close, labeled_df, best_params)
    fit_and_save_regime_thresholds(oof_df, output_path=REGIME_THRESHOLDS_PATH)
    print(f"  -> Regime thresholds saved to {REGIME_THRESHOLDS_PATH}")

    init_dynamic_weights(models=["lightgbm", "random_forest", "ridge"],
                          output_path=DYNAMIC_WEIGHTS_SEED_PATH)
    print(f"  -> Dynamic ensemble weights seeded at {DYNAMIC_WEIGHTS_SEED_PATH}")

    # 11. Per-stock bottom-up models (HSI constituents + watchlist)
    print("\n[10/10] Training per-stock models (constituents + watchlist)...")
    stock_results = train_all_stock_models()
    n_constituents = sum(1 for r in stock_results if r["is_constituent"])
    n_watchlist = len(stock_results) - n_constituents
    print(f"  -> {len(stock_results)} stock models trained "
          f"({n_constituents} constituents, {n_watchlist} watchlist).")

    # 12. Save metrics + mark trained
    metrics = {
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
        "n_rows": len(labeled_df),
        "n_features": len(feature_cols),
        "best_params": best_params,
        "ensemble_close_rmse": ensemble_rmse,
        "high_rmse": high_rmse,
        "low_rmse": low_rmse,
        "walk_forward_report": wf_report,
        "stock_results": stock_results,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    mark_trained_today()
    print("\n" + "=" * 70)
    print("TRAIN_MODEL run complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
