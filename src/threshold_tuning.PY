"""
Grid search for MIN_EXPECTED_MOVE_PCT x MIN_CONFIDENCE thresholds,
using honest walk-forward out-of-sample backtest results.
"""
import json
import numpy as np
import pandas as pd
from itertools import product

from config import MODEL_DIR
from backtest import run_walkforward_oos_backtest_hsi

THRESHOLD_BEST_PARAMS_PATH = MODEL_DIR / "threshold_best_params.json"

# 搜尋範圍：可按需要收窄/擴闊
MOVE_PCT_GRID = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006]
CONFIDENCE_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
MIN_TRADE_COUNT = 30   # 樣本太少嘅組合直接剔除，避免 overfit 去幾單交易


def evaluate_threshold(df: pd.DataFrame, move_th: float, conf_th: float) -> dict:
    """對單一門檻組合，計算在 walk-forward 回測全期入面嘅表現。"""
    mask = (df["pred_close_return"].abs() >= move_th) & (df["confidence"] >= conf_th)
    selected = df[mask]
    n_trades = len(selected)

    if n_trades < MIN_TRADE_COUNT:
        return {
            "move_pct": move_th, "confidence": conf_th,
            "n_trades": n_trades, "status": "INSUFFICIENT_SAMPLES",
        }

    # signed_return：跟訓練方向落注時嘅真實回報（正=賺，負=蝕）
    signed_return = np.sign(selected["pred_close_return"]) * selected["actual_close_return"]

    win_rate = float(selected["directional_hit"].mean())
    expectancy = float(signed_return.mean())          # 每次交易平均回報
    std_return = float(signed_return.std())
    sharpe = expectancy / std_return if std_return > 0 else 0.0
    coverage = n_trades / len(df)                      # 呢個門檻下，有幾多%日子會「值博」

    return {
        "move_pct": move_th,
        "confidence": conf_th,
        "n_trades": n_trades,
        "coverage_pct": round(coverage * 100, 1),
        "win_rate": round(win_rate, 4),
        "expectancy_pct": round(expectancy * 100, 4),
        "sharpe": round(sharpe, 4),
        "status": "OK",
    }


def run_grid_search(start_date: str = "2026-01-01",
                     objective: str = "sharpe") -> dict:
    """
    objective: 'sharpe' (risk-adjusted) | 'expectancy' (raw avg return) | 'win_rate'
    """
    print("Running walk-forward backtest with confidence tracking...")
    df = run_walkforward_oos_backtest_hsi(start_date=start_date)

    if "confidence" not in df.columns:
        raise RuntimeError(
            "backtest.py::run_walkforward_oos_backtest_hsi() 未輸出 'confidence' 欄，"
            "請先套用修正 A，加入 meta-confidence 計算。"
        )

    results = []
    for move_th, conf_th in product(MOVE_PCT_GRID, CONFIDENCE_GRID):
        results.append(evaluate_threshold(df, move_th, conf_th))

    grid_df = pd.DataFrame(results)
    valid = grid_df[grid_df["status"] == "OK"].copy()

    if valid.empty:
        raise RuntimeError(
            f"所有門檻組合嘅交易數都少於 MIN_TRADE_COUNT={MIN_TRADE_COUNT}，"
            "數據期太短或門檻範圍太嚴，請擴大回測期或放寬 MIN_TRADE_COUNT。"
        )

    best_row = valid.sort_values(objective, ascending=False).iloc[0]

    best_params = {
        "MIN_EXPECTED_MOVE_PCT": float(best_row["move_pct"]),
        "MIN_CONFIDENCE": float(best_row["confidence"]),
        "objective_used": objective,
        "objective_value": float(best_row[objective]),
        "n_trades": int(best_row["n_trades"]),
        "coverage_pct": float(best_row["coverage_pct"]),
        "win_rate": float(best_row["win_rate"]),
        "expectancy_pct": float(best_row["expectancy_pct"]),
        "sharpe": float(best_row["sharpe"]),
        "backtest_start_date": start_date,
        "backtest_mode": "walkforward_oos",
    }

    with open(THRESHOLD_BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)

    print("\n=== Grid Search Results (sorted by objective) ===")
    print(valid.sort_values(objective, ascending=False).to_string(index=False))
    print(f"\n=== Best threshold combo (by {objective}) ===")
    print(json.dumps(best_params, indent=2))

    return best_params


if __name__ == "__main__":
    import sys
    objective = sys.argv[1] if len(sys.argv) > 1 else "sharpe"
    start_date = sys.argv[2] if len(sys.argv) > 2 else "2026-01-01"
    run_grid_search(start_date=start_date, objective=objective)
