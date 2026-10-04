"""
Meta-confidence model: RandomForest classifier predicting whether the
primary model's directional call (sign of predicted next_close_return)
is likely correct, given the same feature set. Output is a 0-1 confidence
score used for the "worth trading" verdict.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from config import META_MODEL_PATH


def build_meta_labels(primary_pred_return: pd.Series, actual_return: pd.Series) -> pd.Series:
    pred_direction = np.sign(primary_pred_return)
    actual_direction = np.sign(actual_return)
    return (pred_direction == actual_direction).astype(int)


def train_meta_model(X: pd.DataFrame, meta_labels: pd.Series):
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
    if meta_model is None:
        return 0.5  # neutral default when no meta-model exists yet
    proba = meta_model.predict_proba(X_latest)[0]
    classes = list(meta_model.classes_)
    if 1 in classes:
        return float(proba[classes.index(1)])
    return 0.5
