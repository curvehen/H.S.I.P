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
8. Generate true walk-forward OOF predictions + fit regime-based
   dynamic thresholds (ENH#1) + initialize dynamic ensemble weights (ENH#2)
8.5. (NEW) Calibrate overnight gap model (US futures -> HSI open gap),
   wiring gap_estimator.py + macro_features.py into training for the first
   time — produces the calibration artifact predict.py's predict_hsi()
   consumes to compute estimated_entry_price.
9. Train standalone quantile models (q10/q50/q90) for close band
10. Train direct High / Low regressors
11. Train Meta-Labeling (confidence) model
12. Train per-stock bottom-up models (constituents + watchlist)
13. Save everything + metrics.json + mark_trained_today()

FIX LOG (this revision):
  - CORRECTION: an earlier review pass claimed Step 4's `def model_fn(...)`
    had a stray indentation error. On re-reading the full file verbatim,
    this was incorrect — the original indentation was always valid Python.
    No change was needed or made to Step 4; flagged here only to retract
    that earlier inaccurate claim.
  - build_labeled_hsi_dataset() now returns (labeled_df, raw) instead of
    just labeled_df. The raw OHLC DataFrame (with Open/Close columns) is
    required by gap_estimator.calibrate_gap_model() to compute realized
    historical HSI gaps — it was previously being fetched and then
    discarded, since only the already-feature-engineered labeled_df was
    returned.
  - NEW Step 5.7: wires gap_estimator.calibrate_gap_model() +
    macro_features.get_market_overnight_return_history() into the training
    pipeline. Without this step, predict.py's gap_model_status permanently
    stays "NOT_CALIBRATED"/"DEFAULT_FALLBACK" and estimated_entry_price
    only ever uses the hardcoded DEFAULT_BETA, never a real fitted
    regression. Wrapped fail-safe like the other optional ENH steps —
    any failure (missing futures data, too few samples) logs a warning and
    leaves predict.py on its existing fallback behavior, without blocking
    the rest of training.
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
                     LGB_PARAMS, REGIME_CALIBRATION_MIN_SAMPLES_PER_REGIME)
from data_sources import fetch_with_fallback
from features import build_features, get_numeric_feature_columns
from labeling import build_nextday_labels, LABEL_COLUMNS
from confidence import build_meta_labels, train_meta_model
from news_sentiment import get_daily_market_sentiment, get_stock_sentiment
from ccass_scraper import get_ccass_change
from stock_universe import get_universe
from ensemble_model import HSIEnsembleModel
from optuna_tuning import run_optuna_search
from walk_forward import should_retrain, mark_trained_today, expanding_window_validation
from regime import detect_regime

# NEW: ENH#1 / ENH#2 modules — both optional; training must still succeed
# (producing a fully usable static-weight, static-threshold pipeline) even
# if these experimental calibration layers are unavailable or fail.
try:
    from threshold_calibrator import fit_and_save_regime_thresholds
    _REGIME_CALIBRATION_AVAILABLE = True
except ImportError:
    _REGIME_CALIBRATION_AVAILABLE = False
    print("train_model: threshold_calibrator not available — skipping regime threshold calibration "
          "(predict.py will fall back to the static MIN_EXPECTED_MOVE_PCT / MIN_CONFIDENCE).")

try:
    from dynamic_ensemble_weighter import init_dynamic_weights
    _DYNAMIC_WEIGHTING_AVAILABLE = True
except ImportError:
    _DYNAMIC_WEIGHTING_AVAILABLE = False
    print("train_model: dynamic_ensemble_weighter not available — skipping dynamic weight init "
          "(predict.py will fall back to static inverse-RMSE ensemble weights).")

# NEW — overnight gap calibration (see Step 5.7 in train_hsi_models()).
# Previously defined in gap_estimator.py/macro_features.py but never called
# from anywhere in the training pipeline, leaving predict.py's gap_model
# permanently stuck on its DEFAULT_BETA fallback.
try:
    from gap_estimator import calibrate_gap_model
    from macro_features import get_market_overnight_return_history
    _GAP_CALIBRATION_AVAILABLE = True
except ImportError:
    _GAP_CALIBRATION_AVAILABLE = False
    print("train_model: gap_estimator/macro_features not available — "
          "skipping overnight gap calibration (predict.py will use the "
          "DEFAULT_BETA fallback for estimated_entry_price).")


# ---------------------------------------------------------------------------
# Dataset construction (UPDATED: now also returns raw OHLC, needed by
# calibrate_gap_model() in Step 5.7 — previously only labeled_df was
# returned, discarding the Open/Close columns required for gap calibration)
# ---------------------------------------------------------------------------

def build_labeled_hsi_dataset():
    """Fetches HSI + cross-market data, builds features, applies next-day labeling.
    Returns (labeled_df, raw) — raw retains the original OHLC columns
    (Open/Close) needed downstream for overnight gap calibration."""
    us_futures = fetch_with_fallback(US_FUTURES_TICKER)
    vix = fetch_with_fallback(VIX_TICKER)
    market_sentiment = get_daily_market_sentiment()

    raw = fetch_with_fallback(HSI_TICKER, stooq_ticker="^hsi")
    feat_df = build_features(raw, us_futures=us_futures, vix=vix,
                              ccass_change=0.0, market_sentiment=market_sentiment,
                              stock_sentiment=0.0, ticker=HSI_TICKER, include_macro=True)
    labeled_df = build_nextday_labels(feat_df)
    return labeled_df, raw


# ---------------------------------------------------------------------------
# NEW (ENH#1 support): true walk-forward OOF prediction generation
# ---------------------------------------------------------------------------

def generate_oof_predictions(labeled_df: pd.DataFrame, feature_cols: list, best_params: dict,
                              n_windows: int = 5) -> pd.DataFrame:
    """
    Produces GENUINE out-of-fold predictions for every row across
    `n_windows` expanding walk-forward windows — i.e. each row's prediction
    comes from a model that was trained strictly on data BEFORE that row's
    date, never on the row itself. This is required for regime threshold
    calibration: fitting F1-optimal thresholds on in-sample predictions
    would silently overstate achievable precision/recall, because the
    primary model has already memorized those very rows' relationship
    between features and outcome.

    For each window:
      1. A primary LGBM regressor (predicting next_close_return) is fit on
         the train slice and used to predict the held-out test slice.
      2. A meta-confidence RandomForest classifier is fit OOF-consistently:
         its own training labels (build_meta_labels) are built from the
         TRAIN slice's in-sample primary predictions (mirroring the
         production confidence.py methodology exactly), then the fitted
         classifier scores the held-out test slice — so the confidence
         score for every test row is also a genuine out-of-sample value.
      3. Market regime (BULL/BEAR/NEUTRAL) is tagged per test row via
         regime.py, using that row's own pre-computed MA20/MA60 (regime
         detection only looks at current price vs trailing averages, so
         this does not leak future information).

    Returns a DataFrame indexed like labeled_df, restricted to rows that
    fell inside a test fold (the first window's train slice is never
    scored, by construction), with columns:
        regime, expected_move_pct, confidence, actual_hit, pred_return, actual_return
    """
    n = len(labeled_df)
    fold_edges = np.linspace(int(n * 0.5), n, n_windows + 1, dtype=int)
    rows = []

    for i in range(n_windows):
        train_end = fold_edges[i]
        test_end = fold_edges[i + 1]
        if train_end >= test_end:
            continue

        train_df = labeled_df.iloc[:train_end]
        test_df = labeled_df.iloc[train_end:test_end]
        if len(train_df) < 50 or len(test_df) == 0:
            continue

        X_train, y_train = train_df[feature_cols], train_df["next_close_return"]
        X_test, y_test = test_df[feature_cols], test_df["next_close_return"]

        # 1) Primary model — OOF close-return prediction
        primary_model = lgb.LGBMRegressor(objective="regression", **best_params)
        primary_model.fit(X_train, y_train)
        pred_return_test = primary_model.predict(X_test)

        # 2) Meta-confidence model — trained on TRAIN-fold in-sample labels
        #    only, then scored OOF on the TEST fold (never sees test labels
        #    or test predictions during its own fit).
        primary_pred_train = primary_model.predict(X_train)
        meta_labels_train = build_meta_labels(pd.Series(primary_pred_train, index=X_train.index), y_train)
        meta_clf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=10, random_state=42)
        meta_clf.fit(X_train, meta_labels_train)
        confidence_test = meta_clf.predict_proba(X_test)[:, 1]

        # 3) Regime tag per test row (current-row MA-based, no lookahead)
        regimes_test = [detect_regime(test_df.iloc[[j]]) for j in range(len(test_df))]

        fold_result = pd.DataFrame({
            "regime": regimes_test,
            "expected_move_pct": np.abs(pred_return_test),
            "confidence": confidence_test,
            "pred_return": pred_return_test,
            "actual_return": y_test.values,
        }, index=test_df.index)
        fold_result["actual_hit"] = (np.sign(fold_result["pred_return"]) == np.sign(fold_result["actual_return"]))
        rows.append(fold_result)

    if not rows:
        return pd.DataFrame(columns=["regime", "expected_move_pct", "confidence", "actual_hit",
                                      "pred_return", "actual_return"])
    return pd.concat(rows).sort_index()


# ---------------------------------------------------------------------------
# HSI-level model training
# ---------------------------------------------------------------------------

def train_hsi_models():
    print("=== [Step 1] Checking walk-forward retrain trigger (informational) ===")
    retrain_status = should_retrain()
    print(json.dumps(retrain_status, indent=2))

    print("=== [Step 2] Building HSI dataset (features + next-day labels) ===")
    labeled_df, raw = build_labeled_hsi_dataset()

    feature_cols = get_numeric_feature_columns(labeled_df, exclude=LABEL_COLUMNS)
    X = labeled_df[feature_cols]
    y_close = labeled_df["next_close_return"]
    y_high = labeled_df["next_high_return"]
    y_low = labeled_df["next_low_return"]

    print(f"Dataset ready: {len(X)} samples, {len(feature_cols)} features")

    print("=== [Step 3] Optuna hyperparameter search (close target) ===")
    best_params = run_optuna_search(X, y_close)

    print("=== [Step 4] Expanding-window walk-forward validation (tuned params) ===")
    def model_fn(X_tr, y_tr):
        m = lgb.LGBMRegressor(objective="regression", **best_params)
        m.fit(X_tr, y_tr)
        return m

    wf_report = expanding_window_validation(X, y_close, model_fn, n_windows=5)
    print("Walk-forward validation report:")
    print(json.dumps(wf_report, indent=2))

    print("=== [Step 5] Training Ensemble model (close) ===")
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y_close.iloc[:split_idx], y_close.iloc[split_idx:]

    ensemble = HSIEnsembleModel(lgb_params=best_params)
    ensemble_report = ensemble.fit(X_train, y_train, X_val, y_val)
    print("Ensemble weights:", ensemble_report["weights"])
    ensemble.save(prefix="hsi")

    # =========================================================================
    # ENH#1 & ENH#2: Regime threshold calibration & dynamic weight seed
    # =========================================================================
    if _REGIME_CALIBRATION_AVAILABLE:
        print("=== [Step 5.5] Generating true OOF predictions for regime threshold calibration (ENH#1)... ===")
        try:
            oof_df = generate_oof_predictions(labeled_df, feature_cols, best_params, n_windows=5)
            if len(oof_df) >= REGIME_CALIBRATION_MIN_SAMPLES_PER_REGIME * 2:
                calib_summary = fit_and_save_regime_thresholds(oof_df)
                print(f"  -> Regime thresholds successfully calibrated and saved over {len(oof_df)} OOF samples.")
                print(f"  -> Per-regime summary: {json.dumps(calib_summary, indent=2, default=str)}")
            else:
                print(f"  -> OOF dataset too small ({len(oof_df)} rows) — skipping calibration, "
                      f"fallback to global static defaults remains active.")
        except Exception as e:
            print(f"  -> Regime threshold calibration failed unexpectedly: {e} — falling back to global defaults.")
    else:
        print("=== [Step 5.5] Skipping regime threshold calibration (threshold_calibrator.py not found) ===")

    if _DYNAMIC_WEIGHTING_AVAILABLE:
        print("=== [Step 5.6] Initializing dynamic ensemble weight tracker (ENH#2)... ===")
        try:
            init_dynamic_weights(models=["lightgbm", "random_forest", "ridge"])
            print("  -> Dynamic weight tracker initialized (empty rolling history; "
                  "predict.py will fall back to static weights until enough live days accumulate).")
        except Exception as e:
            print(f"  -> Dynamic weight initialization failed: {e} — predict.py will use static weights only.")
    else:
        print("=== [Step 5.6] Skipping dynamic weight initialization (dynamic_ensemble_weighter.py not found) ===")
    # =========================================================================

    # =========================================================================
    # NEW — overnight gap model calibration (wires gap_estimator.py +
    # macro_features.py into the training pipeline for the first time).
    # Produces the gap calibration artifact consumed by predict.py's
    # predict_hsi() via gap_estimator.load_gap_model() to compute
    # estimated_entry_price. Without this step, that field permanently
    # stays on gap_model_status="NOT_CALIBRATED"/"DEFAULT_FALLBACK".
    # =========================================================================
    if _GAP_CALIBRATION_AVAILABLE:
        print("=== [Step 5.7] Calibrating overnight gap model (US futures -> HSI open gap)... ===")
        try:
            start_date = raw.index.min().strftime("%Y-%m-%d")
            end_date = (raw.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            us_overnight_history = get_market_overnight_return_history(
                start_date=start_date, end_date=end_date, us_futures_ticker=US_FUTURES_TICKER
            )
            gap_calib_result = calibrate_gap_model(raw, us_overnight_history)
            print(f"  -> Gap model calibration result: {json.dumps(gap_calib_result, indent=2, default=str)}")
        except Exception as e:
            print(f"  -> Gap model calibration failed: {e} — predict.py will keep using "
                  f"the DEFAULT_BETA fallback (gap_model_status='DEFAULT_FALLBACK'/'NOT_CALIBRATED').")
    else:
        print("=== [Step 5.7] Skipping overnight gap calibration (gap_estimator.py/macro_features.py not found) ===")
    # =========================================================================

    print("=== [Step 6] Training quantile models (close q10/q50/q90) ===")
    def train_quantile(y, alpha):
        params = {k: v for k, v in best_params.items()}
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
        m.fit(X, y)
        return m

    model_close_q10 = train_quantile(y_close, 0.1)
    model_close_q50 = train_quantile(y_close, 0.5)
    model_close_q90 = train_quantile(y_close, 0.9)

    model_close_q10.booster_.save_model(str(MODEL_CLOSE_Q10_PATH))
    model_close_q50.booster_.save_model(str(MODEL_CLOSE_Q50_PATH))
    model_close_q90.booster_.save_model(str(MODEL_CLOSE_Q90_PATH))

    print("=== [Step 7] Training direct High / Low regressors ===")
    model_high = lgb.LGBMRegressor(objective="regression", **best_params)
    model_high.fit(X, y_high)
    model_high.booster_.save_model(str(MODEL_HIGH_PATH))

    model_low = lgb.LGBMRegressor(objective="regression", **best_params)
    model_low.fit(X, y_low)
    model_low.booster_.save_model(str(MODEL_LOW_PATH))

    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_cols, f)

    print("=== [Step 8] Training Meta-Labeling (confidence) model ===")
    primary_pred = model_close_q50.predict(X)
    meta_labels = build_meta_labels(pd.Series(primary_pred, index=X.index), y_close)
    train_meta_model(X, meta_labels)

    print("=== [Step 9] Training HSI probability model (P升/P跌) ===")
    from probability_model import train_probability_model
    from config import HSI_PROB_MODEL_PATH
    train_probability_model(X, y_close, HSI_PROB_MODEL_PATH)

    return {
        "best_hyperparams": best_params,
        "ensemble_report": ensemble_report,
        "walk_forward_report": wf_report,
        "retrain_trigger_status": retrain_status,
        "n_samples": int(len(X)),
        "n_features": len(feature_cols),
        "date_range": {"start": str(labeled_df.index.min()), "end": str(labeled_df.index.max())},
    }


# ---------------------------------------------------------------------------
# Per-stock bottom-up model training (constituents + watchlist)
# ---------------------------------------------------------------------------

def train_stock_models():
    """
    Trains standalone models for ALL tracked tickers — HSI constituents (used
    for bottom-up index aggregation) plus watchlist tickers (e.g. 1211.HK,
    0968.HK, used for standalone per-stock predictions and email display).

    Uses get_universe() + get_watchlist() (NOT a single merged dict) so the
    "is_constituent" flag is derived explicitly per ticker, keeping the
    watchlist/constituent distinction visible all the way through to
    metrics.json — predict.py and email_report.py rely on this same
    distinction downstream to route results into separate report sections.
    """
    print("=== Training per-stock bottom-up models (constituents + watchlist) ===")
    universe = get_universe()  # {ticker: weight} — HSI constituents only
    watchlist_tickers = get_universe.__globals__.get("get_watchlist", None)
    from stock_universe import get_watchlist  # explicit import, avoids relying on globals() lookup above
    watchlist = get_watchlist()  # [ticker, ...] — independent, no weight

    constituent_tickers = list(universe.keys())
    all_tickers = list(dict.fromkeys(constituent_tickers + watchlist))  # de-duplicated, order-preserving

    print(f"Tracking {len(constituent_tickers)} HSI constituents + {len(watchlist)} watchlist tickers "
          f"= {len(all_tickers)} unique tickers total.")

    stock_metrics = {}

    for ticker in all_tickers:
        is_constituent = ticker in universe
        print(f"--- Training models for {ticker} ({'constituent' if is_constituent else 'watchlist'}) ---")

        try:
            raw = fetch_with_fallback(ticker)
            if raw is None or len(raw) < 100:
                print(f"  -> Insufficient data for {ticker} ({0 if raw is None else len(raw)} rows) — skipped.")
                stock_metrics[ticker] = {"status": "skipped_insufficient_data", "is_constituent": is_constituent}
                continue

            ccass_change = get_ccass_change(ticker)
            stock_sentiment = get_stock_sentiment([ticker.split(".")[0]])
            market_sentiment = get_daily_market_sentiment()

            feat_df = build_features(raw, us_futures=None, vix=None,
                                      ccass_change=ccass_change,
                                      market_sentiment=market_sentiment,
                                      stock_sentiment=stock_sentiment,
                                      ticker=ticker, include_macro=False)
            labeled_df = build_nextday_labels(feat_df)

            if len(labeled_df) < 80:
                print(f"  -> Insufficient labeled rows for {ticker} ({len(labeled_df)}) — skipped.")
                stock_metrics[ticker] = {"status": "skipped_insufficient_labels", "is_constituent": is_constituent}
                continue

            feature_cols = get_numeric_feature_columns(labeled_df, exclude=LABEL_COLUMNS)
            X = labeled_df[feature_cols]
            y_close = labeled_df["next_close_return"]
            y_high = labeled_df["next_high_return"]
            y_low = labeled_df["next_low_return"]

            stock_lgb_params = {k: v for k, v in LGB_PARAMS.items()}

            model_close = lgb.LGBMRegressor(objective="regression", **stock_lgb_params)
            model_close.fit(X, y_close)

            model_high = lgb.LGBMRegressor(objective="regression", **stock_lgb_params)
            model_high.fit(X, y_high)

            model_low = lgb.LGBMRegressor(objective="regression", **stock_lgb_params)
            model_low.fit(X, y_low)

            safe_name = ticker.replace(".", "_")
            model_close.booster_.save_model(str(STOCK_MODEL_DIR / f"{safe_name}_close_q50.txt"))
            model_high.booster_.save_model(str(stock_high_model_path(ticker)))
            model_low.booster_.save_model(str(stock_low_model_path(ticker)))

            with open(STOCK_MODEL_DIR / f"{safe_name}_features.json", "w") as f:
                json.dump(feature_cols, f)

            print(f"  -> Training probability model (P-up) for {ticker}...")
            from probability_model import train_probability_model
            train_probability_model(X, y_close, stock_prob_model_path(ticker))

            stock_metrics[ticker] = {
                "status": "trained",
                "is_constituent": is_constituent,
                "weight": universe.get(ticker),  # None for watchlist tickers, by design
                "n_samples": int(len(X)),
                "n_features": len(feature_cols),
                "date_range": {"start": str(labeled_df.index.min()), "end": str(labeled_df.index.max())},
            }

        except Exception as e:
            print(f"  -> Training failed for {ticker}: {e} — skipped, other tickers unaffected.")
            stock_metrics[ticker] = {"status": f"failed: {e}", "is_constituent": is_constituent}

    n_trained = sum(1 for v in stock_metrics.values() if v.get("status") == "trained")
    print(f"=== Per-stock training complete: {n_trained}/{len(all_tickers)} tickers trained successfully ===")

    return stock_metrics


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def main():
    print("############################################################")
    print("# HSI PREDICTION SYSTEM — FULL TRAINING PIPELINE")
    print(f"# Started: {datetime.datetime.now().isoformat()}")
    print("############################################################\n")

    hsi_metrics = train_hsi_models()
    stock_metrics = train_stock_models()

    full_metrics = {
        "trained_at": datetime.datetime.now().isoformat(),
        "hsi": hsi_metrics,
        "stocks": stock_metrics,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(full_metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to {METRICS_PATH}")

    mark_trained_today()
    print("Training date recorded for walk-forward calendar trigger.")

    print("\n############################################################")
    print("# TRAINING PIPELINE COMPLETE")
    print("############################################################")

    return full_metrics


if __name__ == "__main__":
    main()

