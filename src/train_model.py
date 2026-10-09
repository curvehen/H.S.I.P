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
8. (NEW) Generate true walk-forward OOF predictions + fit regime-based
   dynamic thresholds (ENH#1) + initialize dynamic ensemble weights (ENH#2)
9. Train standalone quantile models (q10/q50/q90) for close band
10. Train direct High / Low regressors
11. Train Meta-Labeling (confidence) model
12. Train per-stock bottom-up models (close q50)
13. Save everything + metrics.json + mark_trained_today()
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


# ---------------------------------------------------------------------------
# Dataset construction (UNCHANGED)
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
    labeled_df = build_labeled_hsi_dataset()

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
    # NEW (ENH#1 & ENH#2): Regime threshold calibration & dynamic weight seed
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
    from probability_model import train_probability_model
    from config import stock_prob_model_path, stock_high_model_path, stock_low_model_path
    from stock_universe import get_universe, get_watchlist

    constituent_tickers = list(get_universe().keys())
    watchlist_tickers = get_watchlist()
    # Union while preserving explicit origin, de-duplicated in case a ticker
    # is ever accidentally present in both lists.
    all_tickers = list(dict.fromkeys(constituent_tickers + watchlist_tickers))

    stock_metrics = {}

    for ticker in all_tickers:
        try:
            is_constituent = ticker in constituent_tickers
            print(f"Training {ticker} (is_constituent={is_constituent})")

            raw = fetch_with_fallback(ticker)
            if raw is None or len(raw) < 250:
                print(f"Skipping {ticker}: insufficient raw data ({0 if raw is None else len(raw)} rows)")
                continue

            ccass_change = get_ccass_change(ticker)
            stock_sentiment = get_stock_sentiment([ticker.split(".")[0]])

            feat_df = build_features(raw, us_futures=None, vix=None,
                                      ccass_change=ccass_change,
                                      market_sentiment=get_daily_market_sentiment(),
                                      stock_sentiment=stock_sentiment,
                                      ticker=ticker, include_macro=False)
            labeled_df = build_nextday_labels(feat_df)

            if len(labeled_df) < 100:
                print(f"Skipping {ticker}: insufficient labeled data ({len(labeled_df)} rows)")
                continue

            feature_cols = get_numeric_feature_columns(labeled_df, exclude=LABEL_COLUMNS)
            X = labeled_df[feature_cols]
            y_close = labeled_df["next_close_return"]
            y_high = labeled_df["next_high_return"]
            y_low = labeled_df["next_low_return"]

            model_close = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
            model_close.fit(X, y_close)
            model_close.booster_.save_model(
                str(STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_close_q50.txt"))

            model_high = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
            model_high.fit(X, y_high)
            model_high.booster_.save_model(str(stock_high_model_path(ticker)))

            model_low = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
            model_low.fit(X, y_low)
            model_low.booster_.save_model(str(stock_low_model_path(ticker)))

            train_probability_model(X, y_close, stock_prob_model_path(ticker))

            feat_path = STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_features.json"
            with open(feat_path, "w") as f:
                json.dump(feature_cols, f)

            stock_metrics[ticker] = {
                "is_constituent": is_constituent,
                "n_samples": int(len(X)),
            }
        except Exception as e:
            print(f"Failed training {ticker}: {e}")
            continue

    return stock_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    hsi_metrics = train_hsi_models()
    stock_metrics = train_stock_models()

    metrics = {
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
        "hsi_model": hsi_metrics,
        "stock_models": stock_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    mark_trained_today()

    print("=== Training complete ===")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()

