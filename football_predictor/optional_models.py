from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from math import factorial
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


@dataclass
class OptionalPrediction:
    available: bool
    home_mean: float | None = None
    away_mean: float | None = None
    reason: str = ""


class PairPoissonModel:
    def __init__(self, home_target: str, away_target: str, min_rows: int = 300) -> None:
        self.home_target = home_target
        self.away_target = away_target
        self.min_rows = int(min_rows)
        self.home_model = Pipeline([
            ("scale", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.35, max_iter=500)),
        ])
        self.away_model = Pipeline([
            ("scale", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.35, max_iter=500)),
        ])
        self.available_ = False
        self.rows_ = 0

    def fit(self, table: pd.DataFrame) -> "PairPoissonModel":
        needed = FEATURE_COLUMNS + [self.home_target, self.away_target]
        if any(col not in table.columns for col in needed):
            return self
        sub = table.dropna(subset=[self.home_target, self.away_target]).copy()
        self.rows_ = len(sub)
        if len(sub) < self.min_rows:
            return self
        X = sub[FEATURE_COLUMNS]
        self.home_model.fit(X, np.clip(sub[self.home_target].astype(float), 0.0, None))
        self.away_model.fit(X, np.clip(sub[self.away_target].astype(float), 0.0, None))
        self.available_ = True
        return self

    def predict(self, features: pd.DataFrame) -> OptionalPrediction:
        if not self.available_:
            return OptionalPrediction(False, reason=f"Недостаточно строк: {self.rows_}/{self.min_rows}")
        home = max(float(self.home_model.predict(features[FEATURE_COLUMNS])[0]), 0.01)
        away = max(float(self.away_model.predict(features[FEATURE_COLUMNS])[0]), 0.01)
        return OptionalPrediction(True, home, away)


class OptionalMarketModels:
    def __init__(self) -> None:
        self.halftime = PairPoissonModel("home_ht_score", "away_ht_score", min_rows=250)
        self.second_half = PairPoissonModel("home_second_half_score", "away_second_half_score", min_rows=250)
        self.corners = PairPoissonModel("home_corners", "away_corners", min_rows=350)
        self.cards = PairPoissonModel("home_yellow_cards", "away_yellow_cards", min_rows=350)

    def fit(self, table: pd.DataFrame) -> "OptionalMarketModels":
        self.halftime.fit(table)
        self.second_half.fit(table)
        self.corners.fit(table)
        self.cards.fit(table)
        return self


def poisson_total_probability(mean_total: float, threshold: float, over: bool = True, max_value: int = 30) -> float:
    probs = []
    for k in range(max_value + 1):
        probs.append(np.exp(-mean_total) * mean_total**k / factorial(k))
    probs = np.array(probs, dtype=float)
    if over:
        return float(probs[np.arange(len(probs)) > threshold].sum())
    return float(probs[np.arange(len(probs)) < threshold].sum())
