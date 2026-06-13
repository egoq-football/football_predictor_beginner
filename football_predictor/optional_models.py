from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import factorial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


EPS = 1e-8


def _sample_weights(dates: pd.Series, half_life_days: float = 730.0) -> np.ndarray:
    values = pd.to_datetime(dates, errors="coerce")
    latest = values.max()
    if pd.isna(latest):
        return np.ones(len(values), dtype=float)
    ages = (latest - values).dt.days.fillna(0).to_numpy(dtype=float)
    return 0.20 + 0.80 * np.power(0.5, ages / half_life_days)


def _poisson_probs(mean_total: float, max_value: int = 40) -> np.ndarray:
    mean_total = max(float(mean_total), EPS)
    values = np.arange(max_value + 1, dtype=int)
    probs = np.array([np.exp(-mean_total) * mean_total**int(k) / factorial(int(k)) for k in values], dtype=float)
    total = probs.sum()
    return probs / total if total > 0 else probs


def poisson_total_probability(mean_total: float, threshold: float, over: bool = True, max_value: int = 40) -> float:
    probs = _poisson_probs(mean_total, max_value=max_value)
    values = np.arange(len(probs), dtype=float)
    if over:
        return float(probs[values > threshold].sum())
    return float(probs[values < threshold].sum())


@dataclass
class OptionalModelStatus:
    name: str
    active: bool = False
    rows: int = 0
    validation_rows: int = 0
    data_from: str = ""
    data_to: str = ""
    selected_algorithm: str = ""
    mae: float | None = None
    baseline_mae: float | None = None
    poisson_deviance: float | None = None
    baseline_deviance: float | None = None
    improvement: float | None = None
    calibrated_markets: int = 0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptionalPrediction:
    available: bool
    home_mean: float | None = None
    away_mean: float | None = None
    reason: str = ""
    status: dict[str, Any] = field(default_factory=dict)


class PairCountModel:
    """Two count regressors with chronological validation and market calibration.

    The model activates only when it has a usable chronological validation block and
    is not materially worse than a simple historical-mean baseline. This prevents a
    nominal row threshold from being mistaken for model quality.
    """

    def __init__(
        self,
        name: str,
        home_target: str,
        away_target: str,
        min_rows: int,
        market_lines: tuple[float, ...],
    ) -> None:
        self.name = name
        self.home_target = home_target
        self.away_target = away_target
        self.min_rows = int(min_rows)
        self.market_lines = tuple(float(x) for x in market_lines)
        self.home_model: Any | None = None
        self.away_model: Any | None = None
        self.available_ = False
        self.status_ = OptionalModelStatus(name=name)
        self.total_calibrators_: dict[float, IsotonicRegression] = {}

    @staticmethod
    def _poisson_pipeline() -> Pipeline:
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.30, max_iter=700)),
        ])

    @staticmethod
    def _boosting() -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.055,
            max_iter=220,
            max_leaf_nodes=15,
            min_samples_leaf=18,
            l2_regularization=0.8,
            random_state=42,
        )

    def _fit_candidate(self, algorithm: str, X: pd.DataFrame, y: pd.Series, weights: np.ndarray):
        model = self._boosting() if algorithm == "Градиентный бустинг" else self._poisson_pipeline()
        if algorithm == "Градиентный бустинг":
            model.fit(X, y, sample_weight=weights)
        else:
            model.fit(X, y, model__sample_weight=weights)
        return model

    @staticmethod
    def _safe_predict(model, X: pd.DataFrame) -> np.ndarray:
        return np.clip(np.asarray(model.predict(X), dtype=float), 0.01, 30.0)

    def fit(self, table: pd.DataFrame) -> "PairCountModel":
        needed = FEATURE_COLUMNS + ["date", self.home_target, self.away_target]
        if any(col not in table.columns for col in needed):
            self.status_.reason = "В обучающей таблице отсутствуют необходимые поля."
            return self

        sub = table.dropna(subset=[self.home_target, self.away_target]).copy()
        sub[self.home_target] = pd.to_numeric(sub[self.home_target], errors="coerce")
        sub[self.away_target] = pd.to_numeric(sub[self.away_target], errors="coerce")
        sub = sub.dropna(subset=[self.home_target, self.away_target]).sort_values("date").reset_index(drop=True)
        sub = sub[(sub[self.home_target] >= 0) & (sub[self.away_target] >= 0)]
        rows = len(sub)
        self.status_.rows = rows
        if rows:
            self.status_.data_from = pd.Timestamp(sub["date"].min()).date().isoformat()
            self.status_.data_to = pd.Timestamp(sub["date"].max()).date().isoformat()

        validation_rows = max(20, int(rows * 0.20)) if rows else 0
        self.status_.validation_rows = min(validation_rows, rows)
        if rows < self.min_rows or rows - validation_rows < 50:
            self.status_.reason = (
                "Модель ещё не обучена: автоматический сбор данных продолжается. "
                f"Сейчас пригодных матчей: {rows}."
            )
            return self

        train = sub.iloc[:-validation_rows]
        valid = sub.iloc[-validation_rows:]
        X_train = train[FEATURE_COLUMNS]
        X_valid = valid[FEATURE_COLUMNS]
        weights = _sample_weights(train["date"])
        y_home_train = train[self.home_target].astype(float)
        y_away_train = train[self.away_target].astype(float)
        y_home_valid = valid[self.home_target].astype(float).to_numpy()
        y_away_valid = valid[self.away_target].astype(float).to_numpy()

        baseline_home = np.full(validation_rows, max(float(y_home_train.mean()), 0.01))
        baseline_away = np.full(validation_rows, max(float(y_away_train.mean()), 0.01))
        baseline_mae = 0.5 * (
            mean_absolute_error(y_home_valid, baseline_home) + mean_absolute_error(y_away_valid, baseline_away)
        )
        baseline_dev = 0.5 * (
            mean_poisson_deviance(y_home_valid, baseline_home) + mean_poisson_deviance(y_away_valid, baseline_away)
        )

        candidates: list[tuple[float, float, str, Any, Any, np.ndarray, np.ndarray]] = []
        for algorithm in ("Пуассоновская регрессия", "Градиентный бустинг"):
            try:
                hm = self._fit_candidate(algorithm, X_train, y_home_train, weights)
                am = self._fit_candidate(algorithm, X_train, y_away_train, weights)
                hp = self._safe_predict(hm, X_valid)
                ap = self._safe_predict(am, X_valid)
                mae = 0.5 * (mean_absolute_error(y_home_valid, hp) + mean_absolute_error(y_away_valid, ap))
                dev = 0.5 * (
                    mean_poisson_deviance(y_home_valid, hp) + mean_poisson_deviance(y_away_valid, ap)
                )
                candidates.append((dev, mae, algorithm, hm, am, hp, ap))
            except Exception:
                continue

        if not candidates:
            self.status_.reason = "Ни один алгоритм не смог обучиться на доступных данных."
            return self

        dev, mae, algorithm, _, _, valid_home_pred, valid_away_pred = min(candidates, key=lambda item: (item[0], item[1]))
        improvement = (baseline_dev - dev) / max(baseline_dev, EPS)
        self.status_.selected_algorithm = algorithm
        self.status_.mae = float(mae)
        self.status_.baseline_mae = float(baseline_mae)
        self.status_.poisson_deviance = float(dev)
        self.status_.baseline_deviance = float(baseline_dev)
        self.status_.improvement = float(improvement)

        # We reject clearly harmful models but permit small validation noise when the
        # candidate is essentially tied with the baseline and the dataset is sizable.
        quality_ok = dev <= baseline_dev * 1.03 and mae <= baseline_mae * 1.05
        if not quality_ok:
            self.status_.reason = (
                "Данных достаточно, но модель пока не прошла хронологическую проверку "
                "лучше простого среднего. Она не используется в прогнозе."
            )
            return self

        full_X = sub[FEATURE_COLUMNS]
        full_weights = _sample_weights(sub["date"])
        self.home_model = self._fit_candidate(algorithm, full_X, sub[self.home_target].astype(float), full_weights)
        self.away_model = self._fit_candidate(algorithm, full_X, sub[self.away_target].astype(float), full_weights)

        valid_total_pred = valid_home_pred + valid_away_pred
        valid_total_actual = y_home_valid + y_away_valid
        for line in self.market_lines:
            raw = np.array([poisson_total_probability(mean, line, over=True) for mean in valid_total_pred])
            target = (valid_total_actual > line).astype(float)
            if len(np.unique(target)) >= 2 and len(target) >= 35:
                try:
                    calibrator = IsotonicRegression(out_of_bounds="clip")
                    calibrator.fit(raw, target)
                    self.total_calibrators_[line] = calibrator
                except Exception:
                    pass

        self.status_.calibrated_markets = len(self.total_calibrators_)
        self.available_ = True
        self.status_.active = True
        self.status_.reason = "Модель активна и прошла хронологическую проверку."
        return self

    def predict(self, features: pd.DataFrame) -> OptionalPrediction:
        if not self.available_ or self.home_model is None or self.away_model is None:
            return OptionalPrediction(False, reason=self.status_.reason, status=self.status_.as_dict())
        home = max(float(self.home_model.predict(features[FEATURE_COLUMNS])[0]), 0.01)
        away = max(float(self.away_model.predict(features[FEATURE_COLUMNS])[0]), 0.01)
        return OptionalPrediction(True, home, away, status=self.status_.as_dict())

    def total_probability(self, mean_total: float, line: float, over: bool) -> float:
        raw_over = poisson_total_probability(mean_total, line, over=True)
        calibrator = self.total_calibrators_.get(float(line))
        if calibrator is not None:
            raw_over = float(np.clip(calibrator.predict([raw_over])[0], 0.01, 0.99))
        return raw_over if over else 1.0 - raw_over


class OptionalMarketModels:
    def __init__(self) -> None:
        self.halftime = PairCountModel(
            "Первый тайм", "home_ht_score", "away_ht_score", min_rows=90, market_lines=(0.5, 1.5, 2.5)
        )
        self.second_half = PairCountModel(
            "Второй тайм", "home_second_half_score", "away_second_half_score", min_rows=90, market_lines=(0.5, 1.5, 2.5)
        )
        self.corners = PairCountModel(
            "Угловые", "home_corners", "away_corners", min_rows=120, market_lines=(7.5, 8.5, 9.5, 10.5)
        )
        self.cards = PairCountModel(
            "Жёлтые карточки", "home_yellow_cards", "away_yellow_cards", min_rows=120, market_lines=(2.5, 3.5, 4.5, 5.5)
        )

    def fit(self, table: pd.DataFrame) -> "OptionalMarketModels":
        self.halftime.fit(table)
        self.second_half.fit(table)
        self.corners.fit(table)
        self.cards.fit(table)
        return self

    def status_rows(self) -> list[dict[str, Any]]:
        return [
            self.halftime.status_.as_dict(),
            self.second_half.status_.as_dict(),
            self.corners.status_.as_dict(),
            self.cards.status_.as_dict(),
        ]
