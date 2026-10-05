"""
Hit-rate tracker: maintains rolling historical directional accuracy per
ticker (including HSI itself), computed from logged predictions vs actuals.
Used to populate the "命中率" column in the email report.
"""

import json
import pandas as pd
from config import HIT_RATE_PATH, PRED_DIR


def compute_hit_rates(min_samples: int = 5) -> dict:
    """
    Reads predictions/backtest_stocks.csv (or live stock predictions log, if
    available) and computes per-ticker directional hit rate. Falls back to
    None (displayed as "N/A") if insufficient history exists.
    """
    hit_rates = {}

    backtest_path = PRED_DIR / "backtest_stocks.csv"
    if backtest_path.exists():
        df = pd.read_csv(backtest_path)
        for ticker, grp in df.groupby("ticker"):
            if len(grp) >= min_samples:
                hit_rates[ticker] = round(float(grp["directional_hit"].mean()) * 100, 1)

    hsi_backtest_path = PRED_DIR / "backtest_hsi.csv"
    if hsi_backtest_path.exists():
        hsi_df = pd.read_csv(hsi_backtest_path)
        if len(hsi_df) >= min_samples:
            hit_rates["HSI"] = round(float(hsi_df["directional_hit"].mean()) * 100, 1)

    with open(HIT_RATE_PATH, "w") as f:
        json.dump(hit_rates, f, indent=2)

    return hit_rates


def get_hit_rate(ticker: str) -> str:
    if HIT_RATE_PATH.exists():
        with open(HIT_RATE_PATH) as f:
            rates = json.load(f)
        rate = rates.get(ticker)
        return f"{rate}%" if rate is not None else "N/A"
    return "N/A"


if __name__ == "__main__":
    rates = compute_hit_rates()
    print(json.dumps(rates, indent=2))
