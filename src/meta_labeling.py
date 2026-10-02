"""
Meta-labeling: second-layer classifier that judges whether the primary
model's signal is trustworthy enough to act on. Reduces false positives
from the quantile regression by learning when it historically failed.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from config import META_MODEL_PATH


def build_meta_labels(primary_pred_return: pd.Series, actual_return: pd.Series,
                       threshold: float = 0.0) -> pd.Series:
    """
    Meta-label = 1 if primary model's directional call was correct, else 0.
    Only samples where primary model gave a non-trivial signal are used.
    """
    pred_direction = np.sign(primary_pred_return)
    actual_direction = np.sign(actual_return)
    meta_label = (pred_direction == actual_direction).astype(int)
    return meta_label


def train_meta_model(X: pd.DataFrame, meta_labels: pd.Series) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42
    )
    clf.fit(X, meta_labels)
    joblib.dump(clf, META_MODEL_PATH)
    return clf


def load_meta_model():
    if META_MODEL_PATH.exists():
        return joblib.load(META_MODEL_PATH)
    return None


def get_signal_confidence(meta_model, X_latest: pd.DataFrame) -> float:
    """Returns probability [0,1] that the primary signal is trustworthy."""
    if meta_model is None:
        return 1.0  # no meta-model yet -> full trust (fallback)
    proba = meta_model.predict_proba(X_latest)[0]
    # class 1 = "correct" ; handle case where only one class was seen in training
    if len(meta_model.classes_) == 2:
        return float(proba[list(meta_model.classes_).index(1)])
    return float(proba[0])
