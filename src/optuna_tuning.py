"""
Optuna hyperparameter search for the LightGBM next-day close model.
Uses Purged K-Fold (embargo matching config.PURGE_EMBARGO_DAYS).
"""

import json
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb

from config import PURGE_EMBARGO_DAYS, MODEL_DIR

optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_N_TRIALS = 30
OPTUNA_TIMEOUT_SEC = 1200
OPTUNA_BEST_PARAMS_PATH = MODEL_DIR / "optuna_best_params.json"


def purged_kfold_indices(n_samples, n_splits=5, embargo=PURGE_EMBARGO_DAYS):
    from sklearn.model_selection import KFold
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
            "verbosity": -1,
        }
        rmses = []
        for train_idx, test_idx in purged_kfold_indices(len(X)):
            model = lgb.LGBMRegressor(objective="regression", **params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            rmses.append(float(np.sqrt(np.mean((preds - y.iloc[test_idx]) ** 2))))
        return float(np.mean(rmses))
    return objective


def run_optuna_search(X: pd.DataFrame, y: pd.Series) -> dict:
    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(X, y), n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT_SEC)
    best_params = study.best_params
    best_params["verbosity"] = -1

    with open(OPTUNA_BEST_PARAMS_PATH, "w") as f:
        json.dump({**best_params, "best_rmse": study.best_value}, f, indent=2)

    print(f"Optuna best RMSE: {study.best_value:.5f}")
    print(json.dumps(best_params, indent=2))
    return best_params


def load_best_params() -> dict:
    if OPTUNA_BEST_PARAMS_PATH.exists():
        with open(OPTUNA_BEST_PARAMS_PATH) as f:
            params = json.load(f)
        return {k: v for k, v in params.items() if k != "best_rmse"}
    from config import LGB_PARAMS
    return LGB_PARAMS
