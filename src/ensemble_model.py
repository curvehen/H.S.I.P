"""
Ensemble model combining:
1. LightGBM Quantile Regressor (primary, captures non-linear patterns)
2. Random Forest Regressor (different bias/variance tradeoff, robust to outliers)
3. Ridge Regression on lag features (captures simple linear momentum/mean-reversion)

Final prediction = weighted average, with weights learned via validation
performance (inverse-RMSE weighting) rather than fixed 1/3 each.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import joblib

from config import ENSEMBLE_META_PATH, MODEL_DIR


class HSIEnsembleModel:
    def __init__(self, lgb_params: dict = None):
        self.lgb_params = lgb_params or {}
        self.models = {}
        self.weights = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        # --- Model 1: LightGBM ---
        lgb_model = lgb.LGBMRegressor(
            objective="quantile", alpha=0.5, verbosity=-1, **self.lgb_params
        )
        lgb_model.fit(X, y)
        lgb_val_pred = lgb_model.predict(X_val)
        lgb_rmse = float(np.sqrt(np.mean((lgb_val_pred - y_val) ** 2)))

        # --- Model 2: Random Forest ---
        rf_model = RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42
        )
        rf_model.fit(X, y)
        rf_val_pred = rf_model.predict(X_val)
        rf_rmse = float(np.sqrt(np.mean((rf_val_pred - y_val) ** 2)))

        # --- Model 3: Ridge on lag/momentum features only ---
        lag_cols = [c for c in X.columns if "return_lag" in c or "MA" in c]
        ridge_model = Ridge(alpha=1.0)
        if lag_cols:
            ridge_model.fit(X[lag_cols], y)
            ridge_val_pred = ridge_model.predict(X_val[lag_cols])
        else:
            ridge_val_pred = np.full(len(y_val), y.mean())
        ridge_rmse = float(np.sqrt(np.mean((ridge_val_pred - y_val) ** 2)))

        # --- Inverse-RMSE weighting (lower error = higher weight) ---
        inv_rmses = {
            "lgb": 1.0 / (lgb_rmse + 1e-8),
            "rf": 1.0 / (rf_rmse + 1e-8),
            "ridge": 1.0 / (ridge_rmse + 1e-8),
        }
        total = sum(inv_rmses.values())
        self.weights = {k: v / total for k, v in inv_rmses.items()}

        self.models = {"lgb": lgb_model, "rf": rf_model, "ridge": ridge_model}
        self.lag_cols = lag_cols

        return {
            "lgb_rmse": lgb_rmse, "rf_rmse": rf_rmse, "ridge_rmse": ridge_rmse,
            "weights": self.weights,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = {
            "lgb": self.models["lgb"].predict(X),
            "rf": self.models["rf"].predict(X),
            "ridge": self.models["ridge"].predict(X[self.lag_cols]) if self.lag_cols else np.zeros(len(X)),
        }
        blended = sum(self.weights[k] * preds[k] for k in preds)
        return blended

    def save(self, prefix: str = "hsi"):
        joblib.dump(self.models["lgb"], MODEL_DIR / f"{prefix}_ensemble_lgb.pkl")
        joblib.dump(self.models["rf"], MODEL_DIR / f"{prefix}_ensemble_rf.pkl")
        joblib.dump(self.models["ridge"], MODEL_DIR / f"{prefix}_ensemble_ridge.pkl")
        meta = {"weights": self.weights, "lag_cols": self.lag_cols}
        with open(ENSEMBLE_META_PATH, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, prefix: str = "hsi"):
        instance = cls()
        instance.models = {
            "lgb": joblib.load(MODEL_DIR / f"{prefix}_ensemble_lgb.pkl"),
            "rf": joblib.load(MODEL_DIR / f"{prefix}_ensemble_rf.pkl"),
            "ridge": joblib.load(MODEL_DIR / f"{prefix}_ensemble_ridge.pkl"),
        }
        with open(ENSEMBLE_META_PATH) as f:
            meta = json.load(f)
        instance.weights = meta["weights"]
        instance.lag_cols = meta["lag_cols"]
        return instance
