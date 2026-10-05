"""
Central configuration — shared by Colab (training) and GitHub Actions (inference).
All paths resolve relative to the repo root (parent of this src/ folder),
regardless of the current working directory the scripts are run from.
"""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
PRED_DIR = ROOT_DIR / "predictions"
LOG_DIR = ROOT_DIR / "logs"
CCASS_DIR = DATA_DIR / "ccass"
NEWS_DIR = DATA_DIR / "news"

for d in [DATA_DIR, MODEL_DIR, PRED_DIR, LOG_DIR, CCASS_DIR, NEWS_DIR]:
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

# ---- Drift detection thresholds ----
DIRECTIONAL_ACC_MIN = 0.45
ROLLING_WINDOW = 5
PAGE_HINKLEY_DELTA = 0.005
PAGE_HINKLEY_THRESHOLD = 10

# ---- Worth-trading verdict thresholds ----
MIN_EXPECTED_MOVE_PCT = 0.003
MIN_CONFIDENCE = 0.60

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
