"""
Overnight gap estimator: estimates tomorrow's HSI open-to-today's-close gap
using overnight US futures / ADR-implied returns as a leading indicator.

Why this is needed:
Open-based high/low labels (labeling.py's use_open_base=True) express
next-day high/low relative to NEXT DAY'S OPEN, not today's close. But at
prediction time (after today's close), tomorrow's open is unknown. This
module estimates it using the historical relationship between overnight
US market moves and HSI's actual open gap, calibrated via simple linear
regression (HSI gap tends to partially track US overnight moves, but not
1:1 — Hong Kong has its own open-market dynamics, China-specific news,
and a dampening/amplifying factor that varies over time).

Usage:
    1. Training: call calibrate_gap_model() once after accumulating enough
       historical (us_overnight_return, hsi_actual_gap) pairs, save the
       calibration coefficients.
    2. Prediction: call estimate_next_open_gap() with the latest overnight
       return to get an estimated gap percentage, then use it to construct
       a virtual "next_open" price for open-based high/low reconstruction.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

from config import MODEL_DIR

GAP_MODEL_PATH = MODEL_DIR / "gap_calibration.json"

# Fallback: if no calibration data is available yet, assume HSI gap tracks
# ~55% of the overnight US futures move (a commonly observed historical
# dampening factor for HK vs US overnight correlation) with zero intercept.
DEFAULT_BETA = 0.55
DEFAULT_INTERCEPT = 0.0


def compute_historical_gaps(hsi_df: pd.DataFrame) -> pd.Series:
    """
    Computes the realized overnight gap for each trading day:
        gap_t = (Open_t - Close_{t-1}) / Close_{t-1}
    """
    # 防禦：若 columns 係 MultiIndex（yfinance 新版常見），先攤平
    if isinstance(hsi_df.columns, pd.MultiIndex):
        hsi_df = hsi_df.copy()
        hsi_df.columns = [c[0] for c in hsi_df.columns]

    if not {"Open", "Close"}.issubset(hsi_df.columns):
        raise ValueError("compute_historical_gaps() requires 'Open' and 'Close' columns.")

    open_col = hsi_df["Open"]
    close_col = hsi_df["Close"]

    # 二重防禦：若因重複欄位名導致仍為 DataFrame，強制取第一欄並轉 Series
    if isinstance(open_col, pd.DataFrame):
        open_col = open_col.iloc[:, 0]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]

    prev_close = close_col.shift(1)
    gap = (open_col - prev_close) / prev_close
    return gap


def calibrate_gap_model(hsi_df: pd.DataFrame, us_overnight_return: pd.Series,
                          min_samples: int = 60) -> dict:
    """
    Fits a simple linear model: hsi_gap = beta * us_overnight_return + intercept

    hsi_df: HSI OHLC dataframe (DatetimeIndex), used to compute realized gaps.
    us_overnight_return: Series aligned to the SAME dates as hsi_df, representing
                          the prior US session's overnight return (e.g. S&P 500
                          futures % change from HSI's previous close to its own open).

    Saves the calibration to GAP_MODEL_PATH and returns the fitted coefficients
    plus diagnostic stats (R^2, n_samples).
    """
    realized_gap = compute_historical_gaps(hsi_df)

    aligned = pd.DataFrame({
        "us_overnight_return": us_overnight_return,
        "realized_gap": realized_gap,
    }).dropna()

    if len(aligned) < min_samples:
        print(f"  [GapEstimator] 樣本不足 ({len(aligned)} < {min_samples})，"
              f"使用 default beta={DEFAULT_BETA}")
        result = {
            "beta": DEFAULT_BETA,
            "intercept": DEFAULT_INTERCEPT,
            "r_squared": None,
            "n_samples": len(aligned),
            "status": "DEFAULT_FALLBACK",
        }
    else:
        X = aligned[["us_overnight_return"]].values
        y = aligned["realized_gap"].values

        reg = LinearRegression()
        reg.fit(X, y)
        r_squared = float(reg.score(X, y))

        result = {
            "beta": float(reg.coef_[0]),
            "intercept": float(reg.intercept_),
            "r_squared": round(r_squared, 4),
            "n_samples": int(len(aligned)),
            "status": "FITTED",
        }
        print(f"  [GapEstimator] Calibrated: beta={result['beta']:.4f} "
              f"intercept={result['intercept']:.5f} R²={r_squared:.4f} n={len(aligned)}")

    with open(GAP_MODEL_PATH, "w") as f:
        json.dump(result, f, indent=2)

    return result


def load_gap_model() -> dict:
    if GAP_MODEL_PATH.exists():
        with open(GAP_MODEL_PATH) as f:
            return json.load(f)
    return {"beta": DEFAULT_BETA, "intercept": DEFAULT_INTERCEPT,
            "r_squared": None, "n_samples": 0, "status": "NOT_CALIBRATED"}


def estimate_next_open_gap(us_overnight_return: float, gap_model: dict = None) -> float:
    """
    Estimates tomorrow's HSI open gap (as a decimal return, e.g. 0.005 = +0.5%)
    given the latest overnight US market move.

    Returns the estimated gap percentage relative to today's close.
    """
    if gap_model is None:
        gap_model = load_gap_model()

    beta = gap_model.get("beta", DEFAULT_BETA)
    intercept = gap_model.get("intercept", DEFAULT_INTERCEPT)

    estimated_gap = beta * us_overnight_return + intercept
    return float(estimated_gap)


def estimate_next_open_price(last_close: float, us_overnight_return: float,
                               gap_model: dict = None) -> float:
    """
    Convenience wrapper: returns an estimated absolute price for tomorrow's
    open, given today's close and the latest overnight US return.
    """
    gap_pct = estimate_next_open_gap(us_overnight_return, gap_model)
    return last_close * (1 + gap_pct)
