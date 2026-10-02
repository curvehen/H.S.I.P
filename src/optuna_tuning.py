"""
Optuna hyperparameter search for the LightGBM quantile model.
Uses Purged K-Fold (embargo = barrier holding period) to avoid leakage
during the search, consistent with the main training validation scheme.
"""

import json
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.model_selection import KFold

from config import (BARRIER_HOLDING_DAYS, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT_SEC,
                     OPTUNA_BEST_PARAMS_PATH)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def purged_kfold_indices(n_samples, n_splits=5, embargo=BARRIER_HOLDING_DAYS):
    kf = KFold(n_splits=n_splits, shuffle=False)
    for train_idx, test_idx in kf.split(np.arange(n_samples)):
        test_start, test_end = test_idx.min(), test_idx.max()
        purge_mask = ~((train_idx >= test_start - embargo) & (train_idx <= test_end + embargo))
        yield train_idx[purge_mask], test_idx


def make_objective(X: pd.DataFrame, y: pd.Series):
    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 80),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        }

        rmses = []
        for train_idx, test_idx in purged_kfold_indices(len(X)):
            model = lgb.LGBMRegressor(objective="quantile", alpha=0.5, verbosity=-1, **params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            rmse = float(np.sqrt(np.mean((preds - y.iloc[test_idx]) ** 2)))
            rmses.append(rmse)

        return float(np.mean(rmses))

    return objective


def run_optuna_search(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Runs Optuna study, returns best hyperparameters.
    Time-capped (OPTUNA_TIMEOUT_SEC) to avoid runaway search time in Colab sessions.
    """
    study = optuna.create_study(direction="minimize")
    objective = make_objective(X, y)
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT_SEC)

    best_params = study.best_params
    best_params["best_rmse"] = study.best_value

    with open(OPTUNA_BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"Optuna search complete. Best RMSE: {study.best_value:.5f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")

    return {k: v for k, v in best_params.items() if k != "best_rmse"}


def load_best_params() -> dict:
    """Loads previously found best params, or returns sensible defaults if none exist."""
    if OPTUNA_BEST_PARAMS_PATH.exists():
        with open(OPTUNA_BEST_PARAMS_PATH) as f:
            params = json.load(f)
        return {k: v for k, v in params.items() if k != "best_rmse"}
    return {
        "num_leaves": 31, "learning_rate": 0.03, "max_depth": 5,
        "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 0.1, "n_estimators": 500,
    }
