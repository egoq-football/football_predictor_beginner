from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


class FootballPredictor:
    """Small ensemble: logistic regression + random forest."""

    def __init__(self) -> None:
        self.logistic = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        self.forest = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            class_weight="balanced_subsample",
        )
        self.classes_ = np.array([0, 1, 2])
        self.metrics_: dict[str, float] = {}

    def fit(self, table: pd.DataFrame) -> "FootballPredictor":
        X = table[FEATURE_COLUMNS]
        y = table["target"].astype(int)
        self.logistic.fit(X, y)
        self.forest.fit(X, y)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        p1 = self._aligned_proba(self.logistic, features)
        p2 = self._aligned_proba(self.forest, features)
        return (p1 * 0.55 + p2 * 0.45)

    def _aligned_proba(self, estimator, X: pd.DataFrame) -> np.ndarray:
        raw = estimator.predict_proba(X[FEATURE_COLUMNS])
        result = np.zeros((len(X), 3), dtype=float)
        for idx, cls in enumerate(estimator.classes_):
            result[:, int(cls)] = raw[:, idx]
        return result


def train_with_chronological_test(table: pd.DataFrame) -> FootballPredictor:
    table = table.sort_values("date").reset_index(drop=True)
    split = int(len(table) * 0.8)
    train = table.iloc[:split]
    test = table.iloc[split:]

    predictor = FootballPredictor().fit(train)
    probs = predictor.predict_proba(test)
    pred = probs.argmax(axis=1)
    y_test = test["target"].astype(int).to_numpy()

    predictor.metrics_ = {
        "matches_total": float(len(table)),
        "train_matches": float(len(train)),
        "test_matches": float(len(test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, probs, labels=[0, 1, 2])),
    }

    # Refit on all data for future predictions.
    predictor.fit(table)
    return predictor


def save_model(model: FootballPredictor, path: str | Path = "models/football_predictor.joblib") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path = "models/football_predictor.joblib") -> FootballPredictor:
    return joblib.load(path)
