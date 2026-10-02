"""
Central configuration file — shared by both Colab (training) 
and GitHub Actions (inference).
Keeping all paths/params here avoids hardcoding in multiple scripts.
"""

from pathlib import Path

# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
PRED_DIR = ROOT_DIR / "predictions"
LOG_DIR = ROOT_DIR / "logs"

for d in [DATA_DIR, MODEL_DIR, PRED_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

# ---- Ticker settings ----
HSI_TICKER = "^HSI"
US_FUTURES_TICKER = "ES=F"   # S&P500 futures as overnight proxy
VIX_TICKER = "^VIX"

# ---- Model files ----
MODEL_Q10_PATH = MODEL_DIR / "lgb_q10.txt"
MODEL_Q50_PATH = MODEL_DIR / "lgb_q50.txt"
MODEL_Q90_PATH = MODEL_DIR / "lgb_q90.txt"
METRICS_PATH = MODEL_DIR / "metrics.json"

# ---- Logging files ----
PRED_LOG_PATH = LOG_DIR / "predictions_log.csv"

# ---- Drift detection thresholds ----
DIRECTIONAL_ACC_MIN = 0.45     # below this over rolling window -> flag retrain
ROLLING_WINDOW = 5              # trading days
PAGE_HINKLEY_DELTA = 0.005
PAGE_HINKLEY_THRESHOLD = 10

# ---- Triple-barrier labeling params ----
BARRIER_HOLDING_DAYS = 3
PROFIT_TAKE_ATR_MULT = 1.5
STOP_LOSS_ATR_MULT = 1.0
