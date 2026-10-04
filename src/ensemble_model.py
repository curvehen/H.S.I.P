"""
Ensemble model combining LightGBM, Random Forest, and Ridge regression.
Final prediction = weighted average, weights derived from inverse validation RMSE.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import joblib

from config import ENSEMBLE_RF_PATH, ENSEMBLE_RIDGE_PATH, ENSEMBLE_WEIGHTS_PATH, MODEL_DIR, LGB_PARAMS


class HSIEnsembleModel:
    def __init__(self, lgb_params: dict = None):
        self.lgb_params = lgb_params or LGB_PARAMS
        self.models = {}
        self.weights = {}
        self.lag_cols = []

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        lgb_model = lgb.LGBMRegressor(objective="regression", **self.lgb_params)
        lgb_model.fit(X, y)
        lgb_val_pred = lgb_model.predict(X_val)
        lgb_rmse = float(np.sqrt(np.mean((lgb_val_pred - y_val) ** 2)))

        rf_model = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42)
        rf_model.fit(X, y)
        rf_val_pred = rf_model.predict(X_val)
        rf_rmse = float(np.sqrt(np.mean((rf_val_pred - y_val) ** 2)))

        lag_cols = [c for c in X.columns if "return_lag" in c or "MA" in c]
        ridge_model = Ridge(alpha=1.0)
        if lag_cols:
            ridge_model.fit(X[lag_cols], y)
            ridge_val_pred = ridge_model.predict(X_val[lag_cols])
        else:
            ridge_val_pred = np.full(len(y_val), y.mean())
        ridge_rmse = float(np.sqrt(np.mean((ridge_val_pred - y_val) ** 2)))

        inv_rmses = {
            "lgb": 1.0 / (lgb_rmse + 1e-8),
            "rf": 1.0 / (rf_rmse + 1e-8),
            "ridge": 1.0 / (ridge_rmse + 1e-8),
        }
        total = sum(inv_rmses.values())
        self.weights = {k: v / total for k, v in inv_rmses.items()}
        self.models = {"lgb": lgb_model, "rf": rf_model, "ridge": ridge_model}
        self.lag_cols = lag_cols

        return {"lgb_rmse": lgb_rmse, "rf_rmse": rf_rmse, "ridge_rmse": ridge_rmse, "weights": self.weights}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = {
            "lgb": self.models["lgb"].predict(X),
            "rf": self.models["rf"].predict(X),
            "ridge": self.models["ridge"].predict(X[self.lag_cols]) if self.lag_cols else np.zeros(len(X)),
        }
        return sum(self.weights[k] * preds[k] for k in preds)

    def save(self, prefix: str = "hsi"):
        self.models["lgb"].booster_.save_model(str(MODEL_DIR / f"{prefix}_ensemble_lgb.txt"))
        joblib.dump(self.models["rf"], ENSEMBLE_RF_PATH)
        joblib.dump(self.models["ridge"], ENSEMBLE_RIDGE_PATH)
        meta = {"weights": self.weights, "lag_cols": self.lag_cols}
        with open(ENSEMBLE_WEIGHTS_PATH, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, prefix: str = "hsi"):
        instance = cls()
        lgb_booster = lgb.Booster(model_file=str(MODEL_DIR / f"{prefix}_ensemble_lgb.txt"))
        instance.models = {
            "lgb": lgb_booster,
            "rf": joblib.load(ENSEMBLE_RF_PATH),
            "ridge": joblib.load(ENSEMBLE_RIDGE_PATH),
        }
        with open(ENSEMBLE_WEIGHTS_PATH) as f:
            meta = json.load(f)
        instance.weights = meta["weights"]
        instance.lag_cols = meta["lag_cols"]
        return instance

    def predict_loaded(self, X: pd.DataFrame) -> np.ndarray:
        """Use this predict variant when models were loaded via .load() (lgb is a Booster, not sklearn estimator)."""
        preds = {
            "lgb": self.models["lgb"].predict(X),
            "rf": self.models["rf"].predict(X),
            "ridge": self.models["ridge"].predict(X[self.lag_cols]) if self.lag_cols else np.zeros(len(X)),
        }
        return sum(self.weights[k] * preds[k] for k in preds)
