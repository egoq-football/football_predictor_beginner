from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS

MODEL_VERSION = "0.3.2-2010-opponent-strength"


class FootballPredictor:
    """Small ensemble: logistic regression + random forest + extra trees."""

    def __init__(self) -> None:
        self.feature_columns_ = list(FEATURE_COLUMNS)
        self.model_version_ = MODEL_VERSION
        self.logistic = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=800, class_weight="balanced", C=0.8)),
            ]
        )
        self.forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=18,
            random_state=42,
            n_jobs=1,
            class_weight="balanced_subsample",
        )
        self.boosting = ExtraTreesClassifier(
            n_estimators=90,
            max_depth=10,
            min_samples_leaf=10,
            random_state=43,
            n_jobs=1,
            class_weight="balanced",
        )
        self.classes_ = np.array([0, 1, 2])
        self.metrics_: dict[str, float] = {}
        self.confusion_matrix_: list[list[int]] = []

    def fit(self, table: pd.DataFrame) -> "FootballPredictor":
        X = table[self.feature_columns_]
        y = table["target"].astype(int)

        # The full archive starts in 2010, but recent matches should influence
        # the fitted models more strongly. A six-year half-life keeps old data
        # useful while reducing its weight. The separate last-five-match layer
        # in predict.py remains outside this 50% mathematical-model block.
        if "date" in table.columns:
            dates = pd.to_datetime(table["date"], errors="coerce")
            latest = dates.max()
            age_years = (latest - dates).dt.days.fillna(0).to_numpy(dtype=float) / 365.25
            sample_weight = 0.25 + 0.75 * np.power(0.5, age_years / 6.0)
        else:
            sample_weight = np.ones(len(table), dtype=float)

        self.logistic.fit(X, y, model__sample_weight=sample_weight)
        self.forest.fit(X, y, sample_weight=sample_weight)
        self.boosting.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        X = features[self.feature_columns_]
        p1 = self._aligned_proba(self.logistic, X)
        p2 = self._aligned_proba(self.forest, X)
        p3 = self._aligned_proba(self.boosting, X)
        probs = p1 * 0.35 + p2 * 0.40 + p3 * 0.25
        return probs / probs.sum(axis=1, keepdims=True)

    def _aligned_proba(self, estimator, X: pd.DataFrame) -> np.ndarray:
        raw = estimator.predict_proba(X)
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

    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    baseline_home = np.full_like(y_test, 2)

    predictor.metrics_ = {
        "matches_total": float(len(table)),
        "train_matches": float(len(train)),
        "test_matches": float(len(test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "baseline_home_accuracy": float(accuracy_score(y_test, baseline_home)),
        "log_loss": float(log_loss(y_test, probs, labels=[0, 1, 2])),
        "away_win_accuracy": float(per_class[0]),
        "draw_accuracy": float(per_class[1]),
        "home_win_accuracy": float(per_class[2]),
    }
    predictor.confusion_matrix_ = cm.astype(int).tolist()

    # Refit on all data for future predictions.
    predictor.fit(table)
    return predictor


def is_model_compatible(model: FootballPredictor) -> bool:
    return (
        getattr(model, "model_version_", None) == MODEL_VERSION
        and list(getattr(model, "feature_columns_", [])) == list(FEATURE_COLUMNS)
    )


def save_model(model: FootballPredictor, path: str | Path = "models/football_predictor.joblib") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path = "models/football_predictor.joblib") -> FootballPredictor:
    return joblib.load(path)
