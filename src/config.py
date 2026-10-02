"""Central config shared by Colab (training) and GitHub Actions (inference)."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
PRED_DIR = ROOT_DIR / "predictions"
LOG_DIR = ROOT_DIR / "logs"
CCASS_DIR = DATA_DIR / "ccass"
NEWS_DIR = DATA_DIR / "news"
INTRADAY_DIR = DATA_DIR / "intraday"

for d in [DATA_DIR, MODEL_DIR, PRED_DIR, LOG_DIR, CCASS_DIR, NEWS_DIR, INTRADAY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HSI_TICKER = "^HSI"
US_FUTURES_TICKER = "ES=F"
VIX_TICKER = "^VIX"

# ---- ADR proxies (HK stocks with US-listed ADRs, ratio approximate) ----
ADR_PROXIES = {
    "9988.HK": {"adr_ticker": "BABA", "ratio": 8},   # 1 ADR = 8 ordinary shares
    "9618.HK": {"adr_ticker": "JD",   "ratio": 2},
    "1810.HK": {"adr_ticker": "XIACF", "ratio": 1},
}

# ---- Model artifact paths ----
MODEL_Q10_PATH = MODEL_DIR / "hsi_lgb_q10.txt"
MODEL_Q50_PATH = MODEL_DIR / "hsi_lgb_q50.txt"
MODEL_Q90_PATH = MODEL_DIR / "hsi_lgb_q90.txt"
ENSEMBLE_META_PATH = MODEL_DIR / "ensemble_weights.json"
META_MODEL_PATH = MODEL_DIR / "meta_model.pkl"
FEATURE_LIST_PATH = MODEL_DIR / "feature_columns.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
OPTUNA_BEST_PARAMS_PATH = MODEL_DIR / "optuna_best_params.json"
STOCK_MODEL_DIR = MODEL_DIR / "stocks"
STOCK_MODEL_DIR.mkdir(exist_ok=True)

PRED_LOG_PATH = LOG_DIR / "predictions_log.csv"
RETRAIN_FLAG_PATH = ROOT_DIR / "RETRAIN_NEEDED.flag"
LAST_TRAIN_DATE_PATH = MODEL_DIR / "last_train_date.txt"

# ---- Drift thresholds ----
DIRECTIONAL_ACC_MIN = 0.45
ROLLING_WINDOW = 5
PAGE_HINKLEY_DELTA = 0.005
PAGE_HINKLEY_THRESHOLD = 10

# ---- Triple-barrier params ----
BARRIER_HOLDING_DAYS = 3
PROFIT_TAKE_ATR_MULT = 1.5
STOP_LOSS_ATR_MULT = 1.0

# ---- Walk-forward retraining ----
WALK_FORWARD_MIN_DAYS_SINCE_RETRAIN = 30   # force retrain after N calendar days regardless of drift
WALK_FORWARD_N_WINDOWS = 5                 # number of expanding-window folds for validation during training

# ---- Optuna ----
OPTUNA_N_TRIALS = 40
OPTUNA_TIMEOUT_SEC = 1800   # 30 min safety cap (Colab session)

# ---- Intraday ----
INTRADAY_INTERVAL = "15m"
INTRADAY_LOOKBACK_DAYS = 59   # yfinance limits 15m data to ~60 days

# ---- News ----
NEWS_RSS_FEEDS = [
    "https://www.hkex.com.hk/eng/rss/newsrelease.xml",
    "https://feeds.reuters.com/reuters/HongKongNews",
]
