"""
Central configuration — shared by Colab (training) and GitHub Actions (inference).
All paths resolve relative to the repo root (parent of this src/ folder),
regardless of the current working directory the scripts are run from.
"""

import os
import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
PRED_DIR = ROOT_DIR / "predictions"
LOG_DIR = ROOT_DIR / "logs"
CCASS_DIR = DATA_DIR / "ccass"
NEWS_DIR = DATA_DIR / "news"
SNAPSHOT_DIR = PRED_DIR / "snapshots"

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

# ---- Daily shared-result / snapshot paths (dual-run reproducibility) ----
# LATEST_RESULT_PATH: 每次 predict.py 執行完即時覆寫一次，供同一次 workflow
#   入面嘅 signal_generator.py / email_report.py 直接讀用，避免三個腳本各自
#   重新抓即時數據，導致同一次 run 內部結果不一致。
# SNAPSHOT_DIR: 按 target_trading_date 存檔（snapshot_YYYY-MM-DD.json），
#   official run 寫入一次後即凍結；preliminary run 或任何同日 re-run 一律
#   優先讀返 snapshot，保證全日結果可重現、一致。
LATEST_RESULT_PATH = PRED_DIR / "latest_result.json"

# ---- Drift detection thresholds ----
DIRECTIONAL_ACC_MIN = 0.45
ROLLING_WINDOW = 5
PAGE_HINKLEY_DELTA = 0.005
PAGE_HINKLEY_THRESHOLD = 10

# ---- Worth-trading verdict thresholds ----
# Auto-tuned via threshold_tuning.py (walk-forward OOS grid search);
# falls back to safe defaults if tuning has never been run yet.
_THRESHOLD_PARAMS_PATH = MODEL_DIR / "threshold_best_params.json"
if _THRESHOLD_PARAMS_PATH.exists():
    with open(_THRESHOLD_PARAMS_PATH) as _f:
        _tuned = json.load(_f)
    MIN_EXPECTED_MOVE_PCT = _tuned["MIN_EXPECTED_MOVE_PCT"]
    MIN_CONFIDENCE = _tuned["MIN_CONFIDENCE"]
else:
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
# 注意：呢個係 flat {ticker: weight} 字典，供部分舊腳本/特徵工程直接引用權重；
# stock_universe.py 嘅 HSI_CONSTITUENTS_INFO（含 sector/name）係獨立、更完整
# 嘅資料來源，predict_hsi_bottom_up() 一律以 stock_universe.get_universe() 為準。
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

# ---- LLM-powered commentary (DashScope OpenAI-compatible endpoint) ----
# API Key 經 GitHub Secrets 注入 (DASHSCOPE_API_KEY)，不可寫死喺代碼入面。
LLM_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
LLM_MODEL = "qwen-plus"          # 待確認實際型號（qwen-plus / qwen-max / qwen-turbo）
LLM_TIMEOUT_SECONDS = 20
LLM_ENABLED_RUN_MODES = {"official"}   # 只喺 4am official run 觸發，preliminary 不叫 LLM

DYNAMIC_WEIGHTS_PATH = MODEL_DIR / "dynamic_ensemble_weights.json"
DYNAMIC_WEIGHT_ROLLING_WINDOW = 60      # trailing trading days kept per sub-model
DYNAMIC_WEIGHT_MIN_SAMPLES = 10         # min live days before trusting dynamic over static weights
DYNAMIC_WEIGHT_PROB_SCALE = 0.01        # logistic steepness for return->prob-up transform


KELLY_FRACTION = 0.5            # Half-Kelly — industry-standard dampening of full-Kelly volatility
MAX_POSITION_PCT = 0.25         # Hard cap: never size above 25% of allocated trading capital
MIN_POSITION_PCT_FLOOR = 0.02   # If Kelly says "trade" but sizes <2%, round up to 2% or treat as noise
KELLY_MIN_EDGE = 0.0            # Minimum raw Kelly fraction required to size any position at all
REGIME_VOL_DAMPENING = True     # Enable/disable the NEUTRAL-regime secondary size trim

