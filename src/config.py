"""Central config shared by Colab (training) and GitHub Actions (inference)."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
PRED_DIR = ROOT_DIR / "predictions"
LOG_DIR = ROOT_DIR / "logs"
CCASS_DIR = DATA_DIR / "ccass"
NEWS_DIR = DATA_DIR / "news"

for d in [DATA_DIR, MODEL_DIR, PRED_DIR, LOG_DIR, CCASS_DIR, NEWS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HSI_TICKER = "^HSI"
US_FUTURES_TICKER = "ES=F"
VIX_TICKER = "^VIX"

# ---- Model artifact paths ----
MODEL_Q10_PATH = MODEL_DIR / "hsi_lgb_q10.txt"
MODEL_Q50_PATH = MODEL_DIR / "hsi_lgb_q50.txt"
MODEL_Q90_PATH = MODEL_DIR / "hsi_lgb_q90.txt"
META_MODEL_PATH = MODEL_DIR / "meta_model.pkl"
FEATURE_LIST_PATH = MODEL_DIR / "feature_columns.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
STOCK_MODEL_DIR = MODEL_DIR / "stocks"
STOCK_MODEL_DIR.mkdir(exist_ok=True)

PRED_LOG_PATH = LOG_DIR / "predictions_log.csv"
RETRAIN_FLAG_PATH = ROOT_DIR / "RETRAIN_NEEDED.flag"

# ---- Drift thresholds ----
DIRECTIONAL_ACC_MIN = 0.45
ROLLING_WINDOW = 5
PAGE_HINKLEY_DELTA = 0.005
PAGE_HINKLEY_THRESHOLD = 10

# ---- Triple-barrier params ----
BARRIER_HOLDING_DAYS = 3
PROFIT_TAKE_ATR_MULT = 1.5
STOP_LOSS_ATR_MULT = 1.0

# ---- News ----
NEWS_RSS_FEEDS = [
    "https://www.hkex.com.hk/eng/rss/newsrelease.xml",
    "https://feeds.reuters.com/reuters/HongKongNews",
]
