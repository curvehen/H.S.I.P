"""
Probability classifier: predicts P(next_close_return > 0) directly,
separate from the regression models. Used for the P(升)/P(跌) email fields
and signal strength classification.
"""

import lightgbm as lgb
import joblib
import pandas as pd
from pathlib import Path
from config import LGB_PARAMS


def train_probability_model(X: pd.DataFrame, y_return: pd.Series, save_path: Path):
    y_binary = (y_return > 0).astype(int)
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=5,
        num_leaves=31, min_child_samples=20, verbosity=-1
    )
    clf.fit(X, y_binary)
    joblib.dump(clf, save_path)
    return clf


def load_probability_model(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


def predict_probability_up(clf, X_latest: pd.DataFrame) -> float:
    """Returns P(up) in [0,1]. Falls back to 0.5 (neutral) if model missing."""
    if clf is None:
        return 0.5
    proba = clf.predict_proba(X_latest)[0]
    classes = list(clf.classes_)
    if 1 in classes:
        return float(proba[classes.index(1)])
    return 0.5


def classify_signal_strength(p_up: float) -> dict:
    """
    Signal strength based on how far P(up) deviates from 50/50.
    margin: percentage points away from neutral (e.g. P(up)=57.7% -> margin=7.7)
    """
    margin = abs(p_up - 0.5) * 100
    if margin < 5:
        label = "弱"
    elif margin < 15:
        label = "中"
    else:
        label = "強"
    return {"label": label, "margin_pct": round(margin, 1)}
