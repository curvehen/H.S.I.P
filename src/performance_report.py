"""
Performance Report Generator.
Analyzes predictions_log.csv (predicted vs actual) and produces:
1. Core accuracy metrics (directional accuracy, RMSE, MAPE)
2. Quantile calibration check (is q10/q90 actually covering ~80% of outcomes?)
3. Confidence calibration (does high meta-label confidence = higher accuracy?)
4. Direct model vs Bottom-up model comparison
5. Data quality impact (stale data vs live data accuracy)
6. Concrete parameter tuning suggestions based on the above
7. Saves a markdown report + charts to predictions/

Run manually (Colab or Actions) anytime you want to review performance:
    python performance_report.py
"""

import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import PRED_LOG_PATH, PRED_DIR, BARRIER_HOLDING_DAYS, \
    PROFIT_TAKE_ATR_MULT, STOP_LOSS_ATR_MULT, DIRECTIONAL_ACC_MIN


# ---------------------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------------------

def load_completed_log() -> pd.DataFrame:
    """Loads prediction log, keeps only rows where actual outcome is known."""
    if not PRED_LOG_PATH.exists():
        raise FileNotFoundError(f"No prediction log found at {PRED_LOG_PATH}")

    log = pd.read_csv(PRED_LOG_PATH)
    log["predict_date"] = pd.to_datetime(log["predict_date"])
    completed = log.dropna(subset=["actual_close"]).copy().sort_values("predict_date")

    if completed.empty:
        raise ValueError("No completed predictions (with actual_close) yet to evaluate.")

    completed["actual_return"] = (completed["actual_close"] - completed["last_close"]) / completed["last_close"]
    completed["abs_error"] = (completed["actual_close"] - completed["pred_price_mid"]).abs()
    completed["pct_error"] = completed["abs_error"] / completed["actual_close"]
    completed["signed_error"] = completed["pred_price_mid"] - completed["actual_close"]
    return completed


# ---------------------------------------------------------------------------
# 2. Core accuracy metrics
# ---------------------------------------------------------------------------

def compute_core_metrics(df: pd.DataFrame) -> dict:
    rmse = float(np.sqrt((df["signed_error"] ** 2).mean()))
    mae = float(df["abs_error"].mean())
    mape = float(df["pct_error"].mean() * 100)

    directional_acc_overall = float(df["directional_hit"].mean())
    directional_acc_rolling5 = float(df["directional_hit"].tail(5).mean()) if len(df) >= 5 else None
    directional_acc_rolling20 = float(df["directional_hit"].tail(20).mean()) if len(df) >= 20 else None

    return {
        "n_predictions_evaluated": int(len(df)),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape_pct": round(mape, 3),
        "directional_accuracy_overall": round(directional_acc_overall, 4),
        "directional_accuracy_last5": round(directional_acc_rolling5, 4) if directional_acc_rolling5 is not None else None,
        "directional_accuracy_last20": round(directional_acc_rolling20, 4) if directional_acc_rolling20 is not None else None,
    }


# ---------------------------------------------------------------------------
# 3. Quantile calibration — are q10/q90 actually capturing ~80% of outcomes?
# ---------------------------------------------------------------------------

def compute_quantile_calibration(df: pd.DataFrame) -> dict:
    within_band = (
        (df["actual_close"] >= df["pred_price_low"]) &
        (df["actual_close"] <= df["pred_price_high"])
    )
    coverage_pct = float(within_band.mean() * 100)

    # Expected coverage for q10-q90 band is 80%
    expected_coverage = 80.0
    deviation = coverage_pct - expected_coverage

    return {
        "actual_coverage_pct": round(coverage_pct, 2),
        "expected_coverage_pct": expected_coverage,
        "deviation_pct_points": round(deviation, 2),
        "band_too_narrow": deviation < -10,   # actual outcomes escape band too often
        "band_too_wide": deviation > 10,      # band rarely binding, low information value
    }


# ---------------------------------------------------------------------------
# 4. Confidence calibration — does meta-label confidence correlate with accuracy?
# ---------------------------------------------------------------------------

def compute_confidence_calibration(df: pd.DataFrame) -> dict:
    if "signal_confidence" not in df.columns or df["signal_confidence"].isna().all():
        return {"status": "NO_CONFIDENCE_DATA"}

    high_conf = df[df["signal_confidence"] >= 0.6]
    low_conf = df[df["signal_confidence"] < 0.6]

    high_conf_acc = float(high_conf["directional_hit"].mean()) if len(high_conf) > 0 else None
    low_conf_acc = float(low_conf["directional_hit"].mean()) if len(low_conf) > 0 else None

    meta_model_useful = (
        high_conf_acc is not None and low_conf_acc is not None
        and high_conf_acc > low_conf_acc
    )

    return {
        "n_high_confidence": int(len(high_conf)),
        "n_low_confidence": int(len(low_conf)),
        "accuracy_high_confidence": round(high_conf_acc, 4) if high_conf_acc is not None else None,
        "accuracy_low_confidence": round(low_conf_acc, 4) if low_conf_acc is not None else None,
        "meta_model_adding_value": meta_model_useful,
    }


# ---------------------------------------------------------------------------
# 5. Direct model vs Bottom-up model — which is more reliable?
# ---------------------------------------------------------------------------

def compute_model_comparison(df: pd.DataFrame) -> dict:
    required = {"pred_return_q50_direct", "pred_return_bottom_up", "actual_return"}
    if not required.issubset(df.columns):
        return {"status": "MISSING_COLUMNS"}

    valid = df.dropna(subset=["pred_return_q50_direct", "pred_return_bottom_up"])
    if valid.empty:
        return {"status": "NO_VALID_ROWS"}

    direct_hit = (np.sign(valid["pred_return_q50_direct"]) == np.sign(valid["actual_return"])).mean()
    bottom_up_hit = (np.sign(valid["pred_return_bottom_up"]) == np.sign(valid["actual_return"])).mean()

    return {
        "n_compared": int(len(valid)),
        "direct_model_directional_accuracy": round(float(direct_hit), 4),
        "bottom_up_model_directional_accuracy": round(float(bottom_up_hit), 4),
        "better_model": "direct" if direct_hit > bottom_up_hit else (
            "bottom_up" if bottom_up_hit > direct_hit else "tie"
        ),
    }


# ---------------------------------------------------------------------------
# 6. Data quality impact — does stale data hurt accuracy?
# ---------------------------------------------------------------------------

def compute_data_quality_impact(df: pd.DataFrame) -> dict:
    if "is_stale" not in df.columns:
        return {"status": "NO_STALENESS_DATA"}

    stale = df[df["is_stale"] == True]
    fresh = df[df["is_stale"] == False]

    stale_acc = float(stale["directional_hit"].mean()) if len(stale) > 0 else None
    fresh_acc = float(fresh["directional_hit"].mean()) if len(fresh) > 0 else None

    return {
        "n_stale_predictions": int(len(stale)),
        "n_fresh_predictions": int(len(fresh)),
        "accuracy_when_stale": round(stale_acc, 4) if stale_acc is not None else None,
        "accuracy_when_fresh": round(fresh_acc, 4) if fresh_acc is not None else None,
        "stale_data_significantly_worse": (
            stale_acc is not None and fresh_acc is not None and (fresh_acc - stale_acc) > 0.1
        ),
    }


# ---------------------------------------------------------------------------
# 7. Parameter tuning suggestions — the actionable output
# ---------------------------------------------------------------------------

def generate_tuning_suggestions(core: dict, calib: dict, conf: dict,
                                 model_cmp: dict, data_quality: dict) -> list:
    suggestions = []

    # --- Directional accuracy too low overall ---
    if core["directional_accuracy_overall"] < DIRECTIONAL_ACC_MIN:
        suggestions.append({
            "issue": "Overall directional accuracy below threshold",
            "current_value": core["directional_accuracy_overall"],
            "threshold": DIRECTIONAL_ACC_MIN,
            "suggested_action": "Trigger retrain in Colab. Consider adding more features "
                                 "(e.g. additional cross-market lags) before retraining.",
        })

    # --- Quantile band miscalibrated ---
    if calib.get("band_too_narrow"):
        new_profit_mult = round(PROFIT_TAKE_ATR_MULT * 1.2, 2)
        new_stop_mult = round(STOP_LOSS_ATR_MULT * 1.2, 2)
        suggestions.append({
            "issue": f"Prediction band too narrow (actual coverage {calib['actual_coverage_pct']}% vs expected 80%)",
            "suggested_action": f"Widen barriers in config.py: "
                                 f"PROFIT_TAKE_ATR_MULT {PROFIT_TAKE_ATR_MULT} -> {new_profit_mult}, "
                                 f"STOP_LOSS_ATR_MULT {STOP_LOSS_ATR_MULT} -> {new_stop_mult}",
        })
    elif calib.get("band_too_wide"):
        new_profit_mult = round(PROFIT_TAKE_ATR_MULT * 0.85, 2)
        new_stop_mult = round(STOP_LOSS_ATR_MULT * 0.85, 2)
        suggestions.append({
            "issue": f"Prediction band too wide (actual coverage {calib['actual_coverage_pct']}% vs expected 80%), "
                     f"reducing practical usefulness of entry/exit levels",
            "suggested_action": f"Tighten barriers in config.py: "
                                 f"PROFIT_TAKE_ATR_MULT {PROFIT_TAKE_ATR_MULT} -> {new_profit_mult}, "
                                 f"STOP_LOSS_ATR_MULT {STOP_LOSS_ATR_MULT} -> {new_stop_mult}",
        })

    # --- Meta-labeling not adding value ---
    if conf.get("meta_model_adding_value") is False:
        suggestions.append({
            "issue": "Meta-labeling confidence score does not correlate with actual accuracy "
                     f"(high-conf acc={conf.get('accuracy_high_confidence')}, "
                     f"low-conf acc={conf.get('accuracy_low_confidence')})",
            "suggested_action": "Retrain meta_model with more samples, or add features specific "
                                 "to signal quality (e.g. recent volatility regime) before relying on confidence filtering.",
        })

    # --- Direct vs Bottom-up blending weight ---
    if model_cmp.get("better_model") == "bottom_up":
        suggestions.append({
            "issue": f"Bottom-up model outperforming direct HSI model "
                     f"({model_cmp['bottom_up_model_directional_accuracy']} vs "
                     f"{model_cmp['direct_model_directional_accuracy']})",
            "suggested_action": "In predict.py, increase weight on bottom_up_pred_return in the blend "
                                 "(e.g. 0.7*bottom_up + 0.3*direct instead of 50/50).",
        })
    elif model_cmp.get("better_model") == "direct":
        suggestions.append({
            "issue": f"Direct HSI model outperforming bottom-up model "
                     f"({model_cmp['direct_model_directional_accuracy']} vs "
                     f"{model_cmp['bottom_up_model_directional_accuracy']})",
            "suggested_action": "In predict.py, reduce weight on bottom_up_pred_return in the blend, "
                                 "or expand stock_universe.py coverage (more constituents) to improve bottom-up signal quality.",
        })

    # --- Stale data materially hurting performance ---
    if data_quality.get("stale_data_significantly_worse"):
        suggestions.append({
            "issue": f"Predictions made on stale data perform much worse "
                     f"(stale acc={data_quality['accuracy_when_stale']} vs fresh acc={data_quality['accuracy_when_fresh']})",
            "suggested_action": "Add a stronger fallback data source (e.g. paid API) or skip/flag predictions "
                                 "made on stale data rather than treating them as normal signals.",
        })

    if not suggestions:
        suggestions.append({
            "issue": "No material issues detected",
            "suggested_action": "Current parameters appear reasonably calibrated. Continue monitoring.",
        })

    return suggestions


# ---------------------------------------------------------------------------
# 8. Charts
# ---------------------------------------------------------------------------

def save_charts(df: pd.DataFrame, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Predicted vs Actual close
    axes[0].plot(df["predict_date"], df["actual_close"], label="Actual Close", marker="o")
    axes[0].plot(df["predict_date"], df["pred_price_mid"], label="Predicted Mid", marker="x")
    axes[0].fill_between(df["predict_date"], df["pred_price_low"], df["pred_price_high"],
                          alpha=0.2, label="Q10-Q90 Band")
    axes[0].set_title("Predicted vs Actual HSI Close")
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    # Rolling directional accuracy
    rolling_acc = df["directional_hit"].rolling(window=5, min_periods=1).mean()
    axes[1].plot(df["predict_date"], rolling_acc, label="Rolling 5-day Directional Accuracy", color="green")
    axes[1].axhline(y=DIRECTIONAL_ACC_MIN, color="red", linestyle="--", label=f"Min threshold ({DIRECTIONAL_ACC_MIN})")
    axes[1].set_title("Rolling Directional Accuracy")
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    chart_path = out_dir / f"performance_chart_{datetime.date.today()}.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path


# ---------------------------------------------------------------------------
# 9. Markdown report assembly
# ---------------------------------------------------------------------------

def build_markdown_report(core, calib, conf, model_cmp, data_quality, suggestions, chart_path) -> str:
    today = datetime.date.today().isoformat()
    lines = [
        f"# HSI Model Performance Report — {today}",
        "",
        "## Core Accuracy Metrics",
        f"- Predictions evaluated: {core['n_predictions_evaluated']}",
        f"- RMSE: {core['rmse']}",
        f"- MAE: {core['mae']}",
        f"- MAPE: {core['mape_pct']}%",
        f"- Directional accuracy (overall): {core['directional_accuracy_overall']}",
        f"- Directional accuracy (last 5): {core['directional_accuracy_last5']}",
        f"- Directional accuracy (last 20): {core['directional_accuracy_last20']}",
        "",
        "## Quantile Band Calibration (Target: 80% coverage)",
        f"- Actual coverage: {calib['actual_coverage_pct']}%",
        f"- Deviation: {calib['deviation_pct_points']} percentage points",
        f"- Band too narrow: {calib.get('band_too_narrow')}",
        f"- Band too wide: {calib.get('band_too_wide')}",
        "",
        "## Meta-Labeling Confidence Calibration",
        json.dumps(conf, indent=2),
        "",
        "## Direct Model vs Bottom-Up Model",
        json.dumps(model_cmp, indent=2),
        "",
        "## Data Quality Impact (Stale vs Fresh)",
        json.dumps(data_quality, indent=2),
        "",
        "## Parameter Tuning Suggestions",
    ]
    for i, s in enumerate(suggestions, 1):
        lines.append(f"{i}. **Issue:** {s['issue']}")
        lines.append(f"   **Suggested action:** {s['suggested_action']}")
        lines.append("")

    lines.append(f"## Chart")
    lines.append(f"![performance chart]({chart_path.name})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_completed_log()

    core = compute_core_metrics(df)
    calib = compute_quantile_calibration(df)
    conf = compute_confidence_calibration(df)
    model_cmp = compute_model_comparison(df)
    data_quality = compute_data_quality_impact(df)
    suggestions = generate_tuning_suggestions(core, calib, conf, model_cmp, data_quality)

    chart_path = save_charts(df, PRED_DIR)
    report_md = build_markdown_report(core, calib, conf, model_cmp, data_quality, suggestions, chart_path)

    report_path = PRED_DIR / f"performance_report_{datetime.date.today()}.md"
    report_path.write_text(report_md, encoding="utf-8")

    print(report_md)
    print(f"\nReport saved to: {report_path}")
    print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    main()
