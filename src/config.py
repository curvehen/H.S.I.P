"""
Central configuration — shared by Colab (training) and GitHub Actions (inference).
All paths resolve relative to the repo root (parent of this src/ folder),
regardless of the current working directory the scripts are run from.
"""
import json
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
PRED_DIR = ROOT_DIR / "predictions"
LOG_DIR = ROOT_DIR / "logs"
CCASS_DIR = DATA_DIR / "ccass"
NEWS_DIR = DATA_DIR / "news"
SNAPSHOT_DIR = PRED_DIR / "snapshots"   # NEW: feature-snapshot/freeze mechanism (reproducibility)

for d in [DATA_DIR, MODEL_DIR, PRED_DIR, LOG_DIR, CCASS_DIR, NEWS_DIR, SNAPSHOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- Tickers ----
HSI_TICKER = "^HSI"
US_FUTURES_TICKER = "ES=F"
VIX_TICKER = "^VIX"

# ---- Model artifact paths (HSI-level) ----
MODEL_CLOSE_Q10_PATH = MODEL_DIR / "hsi_next_close_q10.txt"
MODEL_CLOSE_Q50_PATH = MODEL_DIR / "hsi_next_close_q50.txt"
MODEL_CLOSE_Q90_PATH = MODEL_DIR / "hsi_next_close_q90.txt"
MODEL_HIGH_PATH = MODEL_DIR / "hsi_next_high.txt"
MODEL_LOW_PATH = MODEL_DIR / "hsi_next_low.txt"

ENSEMBLE_RF_PATH = MODEL_DIR / "hsi_ensemble_rf.pkl"
ENSEMBLE_RIDGE_PATH = MODEL_DIR / "hsi_ensemble_ridge.pkl"
ENSEMBLE_WEIGHTS_PATH = MODEL_DIR / "hsi_ensemble_weights.json"

META_MODEL_PATH = MODEL_DIR / "meta_confidence_model.pkl"
FEATURE_LIST_PATH = MODEL_DIR / "feature_columns.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

STOCK_MODEL_DIR = MODEL_DIR / "stocks"
STOCK_MODEL_DIR.mkdir(exist_ok=True)

PRED_LOG_PATH = LOG_DIR / "predictions_log.csv"
SIGNAL_LOG_PATH = PRED_DIR / "signals_log.csv"
RETRAIN_FLAG_PATH = ROOT_DIR / "RETRAIN_NEEDED.flag"
LAST_TRAIN_DATE_PATH = MODEL_DIR / "last_train_date.txt"

# NEW: reproducibility — dual-run inference (preliminary vs. official)
LATEST_RESULT_PATH = PRED_DIR / "latest_result.json"
# Snapshot file name pattern consumed by predict.py: SNAPSHOT_DIR / f"{date}_features.parquet"

# ---- Drift detection thresholds ----
DIRECTIONAL_ACC_MIN = 0.45
ROLLING_WINDOW = 5
PAGE_HINKLEY_DELTA = 0.005
PAGE_HINKLEY_THRESHOLD = 10
ROLLING_ACCURACY_WINDOW = ROLLING_WINDOW  # NEW: alias — evaluate_drift.py's preferred name, same value
DRIFT_FLAG_PATH = MODEL_DIR / "drift_status.json"  # NEW: structured drift/accuracy summary (distinct from RETRAIN_FLAG_PATH trigger file)

# ---- Worth-trading verdict thresholds ----
# Auto-tuned via threshold_tuning.py (walk-forward OOS grid search)
_THRESHOLD_PARAMS_PATH = MODEL_DIR / "threshold_best_params.json"
if _THRESHOLD_PARAMS_PATH.exists():
    with open(_THRESHOLD_PARAMS_PATH) as _f:
        _tuned = json.load(_f)
    MIN_EXPECTED_MOVE_PCT = _tuned["MIN_EXPECTED_MOVE_PCT"]
    MIN_CONFIDENCE = _tuned["MIN_CONFIDENCE"]
else:
    MIN_EXPECTED_MOVE_PCT = 0.006
    MIN_CONFIDENCE = 0.75

# ---- LightGBM default hyperparameters (fixed, stable, no external tuning dependency) ----
LGB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbosity": -1,
}

# ---- Purged K-Fold embargo (days) ----
PURGE_EMBARGO_DAYS = 2

# ---- Walk-forward retraining ----
WALK_FORWARD_MIN_DAYS_SINCE_RETRAIN = 14

# ---- News ----
NEWS_RSS_FEEDS = [
    "https://www.hkex.com.hk/eng/rss/newsrelease.xml",
    "https://feeds.reuters.com/reuters/HongKongNews",
]

# ---- HSI constituents (representative subset; expand as needed) ----
HSI_CONSTITUENTS = {
    "0700.HK": 0.081, "9988.HK": 0.062, "0941.HK": 0.058, "1299.HK": 0.055,
    "0388.HK": 0.052, "3690.HK": 0.045, "0005.HK": 0.044, "1810.HK": 0.030,
    "2318.HK": 0.028, "0016.HK": 0.025, "0027.HK": 0.020, "0883.HK": 0.020,
    "1398.HK": 0.019, "3988.HK": 0.018, "0002.HK": 0.017, "0001.HK": 0.016,
}

# ---- ADR proxies ----
ADR_PROXIES = {
    "9988.HK": {"adr_ticker": "BABA", "ratio": 8},
    "9618.HK": {"adr_ticker": "JD", "ratio": 2},
}

# ---- Probability models ----
HSI_PROB_MODEL_PATH = MODEL_DIR / "hsi_probability_model.pkl"

# ---- Per-stock model path helpers ----
def stock_prob_model_path(ticker: str):
    return STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_prob_model.pkl"

def stock_high_model_path(ticker: str):
    return STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_high.txt"

def stock_low_model_path(ticker: str):
    return STOCK_MODEL_DIR / f"{ticker.replace('.', '_')}_low.txt"

# ---- Hit rate tracking ----
HIT_RATE_PATH = MODEL_DIR / "hit_rates.json"

# =============================================================================
# NEW — ENH#1: Regime-based dynamic threshold calibration (threshold_calibrator.py)
# =============================================================================
REGIME_THRESHOLDS_PATH = MODEL_DIR / "regime_thresholds.json"
REGIME_CALIBRATION_MIN_SAMPLES_PER_REGIME = 30   # min OOF samples required to fit a regime-specific threshold
REGIME_CALIBRATION_F1_GRID_STEPS = 25            # grid resolution for expected-move / confidence threshold search
REGIME_CALIBRATION_FALLBACK = {                   # used when a regime has too few samples to calibrate
    "MIN_EXPECTED_MOVE_PCT": MIN_EXPECTED_MOVE_PCT,
    "MIN_CONFIDENCE": MIN_CONFIDENCE,
}

# =============================================================================
# NEW — ENH#2: Dynamic ensemble weighting via rolling Brier score (dynamic_ensemble_weighter.py)
# =============================================================================
DYNAMIC_WEIGHTS_PATH = MODEL_DIR / "dynamic_ensemble_weights.json"
DYNAMIC_WEIGHT_ROLLING_WINDOW = 60      # trailing trading days kept per sub-model
DYNAMIC_WEIGHT_MIN_SAMPLES = 10         # min live days before trusting dynamic over static weights
DYNAMIC_WEIGHT_PROB_SCALE = 0.01        # logistic steepness for return->prob-up transform
DYNAMIC_WEIGHT_MIN_HISTORY = 15
DYNAMIC_WEIGHT_SIGMOID_K = 50

# =============================================================================
# NEW — ENH#3: Fractional Kelly position sizing (position_sizer.py)
# =============================================================================
KELLY_FRACTION = 0.5            # Half-Kelly — industry-standard dampening of full-Kelly volatility
MAX_POSITION_PCT = 0.25         # Hard cap: never size above 25% of allocated trading capital
MIN_POSITION_PCT_FLOOR = 0.02   # If Kelly says "trade" but sizes <2%, round up to 2% or treat as noise
KELLY_MIN_EDGE = 0.0            # Minimum raw Kelly fraction required to size any position at all
REGIME_VOL_DAMPENING = True     # Enable/disable the NEUTRAL-regime secondary size trim

# =============================================================================
# NEW — ENH#4: Transaction cost / slippage simulation (backtest.py, run_daily_backtest.py)
# =============================================================================
TRANSACTION_COST = 0.002        # Round-trip cost as a fraction of notional (e.g. commission + spread)
SLIPPAGE_POINTS = 3             # Fixed HSI index-point slippage applied per side in backtests

# =============================================================================
# NEW — Watchlist (standalone monitoring, excluded from HSI bottom-up weight aggregation)
# =============================================================================
WATCHLIST_TICKERS = ["1211.HK", "0968.HK"]   # BYD, Xinyi Solar — see stock_universe.get_watchlist()

# =============================================================================
# NEW — DashScope-compatible LLM integration (llm_analysis.py)
# =============================================================================
DASHSCOPE_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")   # must be injected via GitHub Actions secret
DASHSCOPE_MODEL_NAME = os.environ.get("DASHSCOPE_MODEL_NAME", "qwen-plus")
LLM_REQUEST_TIMEOUT_SECONDS = 20
LLM_MAX_TOKENS = 400
LLM_ANALYSIS_ENABLED = bool(DASHSCOPE_API_KEY)   # auto-disables cleanly if secret not configured
