from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = [
    "probability",
    "agreement",
    "data_quality",
    "abs_elo_diff",
    "abs_fifa_diff",
    "expected_total",
    "line",
]
CATEGORICAL_COLUMNS = ["category", "direction"]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


@dataclass
class SelectorMetrics:
    active: bool = False
    training_rows: int = 0
    calibration_rows: int = 0
    validation_rows: int = 0
    brier: float | None = None
    log_loss: float | None = None
    auc: float | None = None
    reason: str = "Селектор ещё не обучен."

    def as_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "training_rows": self.training_rows,
            "calibration_rows": self.calibration_rows,
            "validation_rows": self.validation_rows,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "auc": self.auc,
            "reason": self.reason,
        }


@dataclass
class OutcomeSelectorModel:
    """Learns how reliable a candidate market is from historical predictions.

    The model is intentionally separate from the match-result ensemble. It does
    not invent bookmaker prices. It estimates the chance that a candidate
    outcome is correct, using historical out-of-sample candidates. Real market
    prices, when available, are used only as an eligibility filter later.
    """

    estimator: Pipeline | None = None
    calibrator: IsotonicRegression | None = None
    metrics: SelectorMetrics = field(default_factory=SelectorMetrics)

    @staticmethod
    def _pipeline() -> Pipeline:
        transform = ColumnTransformer(
            transformers=[
                (
                    "category",
                    OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                    CATEGORICAL_COLUMNS,
                ),
                ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ],
            remainder="drop",
        )
        return Pipeline(
            [
                ("features", transform),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1200,
                        C=0.55,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def fit(self, frame: pd.DataFrame) -> "OutcomeSelectorModel":
        if frame is None or frame.empty or "target" not in frame:
            self.metrics.reason = "Нет исторических кандидатов для обучения селектора."
            return self

        data = frame.copy()
        for col in FEATURE_COLUMNS:
            if col not in data:
                data[col] = "unknown" if col in CATEGORICAL_COLUMNS else 0.0
        data["date"] = pd.to_datetime(data.get("date"), errors="coerce")
        data = data.dropna(subset=["target"]).sort_values(["date", "match_key", "candidate_order"], na_position="first")
        data["target"] = pd.to_numeric(data["target"], errors="coerce")
        data = data.dropna(subset=["target"])
        data["target"] = data["target"].astype(int)
        for col in NUMERIC_COLUMNS:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
        for col in CATEGORICAL_COLUMNS:
            data[col] = data[col].fillna("unknown").astype(str)

        # Split by match rather than candidate row so outcomes from one match do
        # not leak across train/calibration/validation partitions.
        match_keys = data["match_key"].drop_duplicates().tolist()
        if len(match_keys) < 120:
            self.metrics.reason = f"Недостаточно исторических матчей для селектора: {len(match_keys)}."
            return self

        train_end = max(1, int(len(match_keys) * 0.60))
        cal_end = max(train_end + 1, int(len(match_keys) * 0.80))
        train_keys = set(match_keys[:train_end])
        cal_keys = set(match_keys[train_end:cal_end])
        valid_keys = set(match_keys[cal_end:])

        train = data[data["match_key"].isin(train_keys)]
        calibration = data[data["match_key"].isin(cal_keys)]
        validation = data[data["match_key"].isin(valid_keys)]
        self.metrics.training_rows = len(train)
        self.metrics.calibration_rows = len(calibration)
        self.metrics.validation_rows = len(validation)

        if min(len(train), len(calibration), len(validation)) < 100 or train["target"].nunique() < 2:
            self.metrics.reason = "Хронологические части селектора слишком малы или содержат один класс."
            return self

        estimator = self._pipeline()
        estimator.fit(train[FEATURE_COLUMNS], train["target"])

        raw_cal = estimator.predict_proba(calibration[FEATURE_COLUMNS])[:, 1]
        calibrator: IsotonicRegression | None = None
        if calibration["target"].nunique() >= 2:
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
            calibrator.fit(raw_cal, calibration["target"].to_numpy(dtype=float))

        raw_valid = estimator.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
        valid_prob = calibrator.predict(raw_valid) if calibrator is not None else raw_valid
        valid_prob = np.clip(valid_prob, 0.01, 0.99)
        y_valid = validation["target"].to_numpy(dtype=int)

        self.estimator = estimator
        self.calibrator = calibrator
        self.metrics.active = True
        self.metrics.brier = float(brier_score_loss(y_valid, valid_prob))
        self.metrics.log_loss = float(log_loss(y_valid, valid_prob, labels=[0, 1]))
        try:
            self.metrics.auc = float(roc_auc_score(y_valid, valid_prob))
        except Exception:
            self.metrics.auc = None
        self.metrics.reason = "Селектор обучен на хронологических вневыборочных кандидатах и откалиброван."
        return self

    def predict_success_probability(self, rows: pd.DataFrame) -> np.ndarray:
        if self.estimator is None or rows is None or rows.empty:
            return np.asarray([], dtype=float)
        frame = rows.copy()
        for col in FEATURE_COLUMNS:
            if col not in frame:
                frame[col] = "unknown" if col in CATEGORICAL_COLUMNS else 0.0
        for col in NUMERIC_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        for col in CATEGORICAL_COLUMNS:
            frame[col] = frame[col].fillna("unknown").astype(str)
        raw = self.estimator.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
        if self.calibrator is not None:
            raw = self.calibrator.predict(raw)
        return np.clip(np.asarray(raw, dtype=float), 0.01, 0.99)

    def status(self) -> dict[str, Any]:
        return self.metrics.as_dict()


def _agreement(values: list[float]) -> float:
    clean = [float(v) for v in values if np.isfinite(v)]
    if not clean:
        return 0.45
    return float(np.clip(1.0 - np.std(clean) / 0.22, 0.20, 1.0))


def _history_data_quality(row: pd.Series) -> float:
    parts = [
        1.0 if float(row.get("fifa_available", 0.0)) >= 0.5 else 0.45,
        min(1.0, min(float(row.get("home_matches", 0.0)), float(row.get("away_matches", 0.0))) / 25.0),
        1.0 if float(row.get("context_lineups_known", 0.0)) >= 0.5 else 0.72,
    ]
    return float(np.clip(np.mean(parts), 0.25, 1.0))


def build_historical_selector_frame(
    table: pd.DataFrame,
    final_probs: np.ndarray,
    component_probabilities: dict[str, np.ndarray],
    dc_model: Any,
) -> pd.DataFrame:
    """Create historical candidate outcomes from genuinely out-of-sample forecasts."""
    from .dixon_coles import score_markets

    rows: list[dict[str, Any]] = []
    final_probs = np.asarray(final_probs, dtype=float)
    for position, (_, match) in enumerate(table.reset_index(drop=True).iterrows()):
        home_goals = int(match["home_score"])
        away_goals = int(match["away_score"])
        total_goals = home_goals + away_goals
        expected = dc_model.predict(str(match["home_team"]), str(match["away_team"]), bool(match["neutral"]))
        markets = score_markets(expected)
        match_key = f"{pd.Timestamp(match['date']).date()}|{match['home_team']}|{match['away_team']}|{position}"
        quality = _history_data_quality(match)
        common = {
            "date": pd.Timestamp(match["date"]),
            "match_key": match_key,
            "data_quality": quality,
            "abs_elo_diff": abs(float(match.get("elo_diff", 0.0))),
            "abs_fifa_diff": abs(float(match.get("fifa_points_diff", 0.0))),
            "expected_total": float(expected.home_lambda + expected.away_lambda),
        }
        order = 0

        def add(category: str, direction: str, probability: float, agreement: float, line: float, target: bool) -> None:
            nonlocal order
            rows.append({
                **common,
                "candidate_order": order,
                "category": category,
                "direction": direction,
                "probability": float(np.clip(probability, 0.001, 0.999)),
                "agreement": float(np.clip(agreement, 0.0, 1.0)),
                "line": float(line),
                "target": int(bool(target)),
            })
            order += 1

        # 1X2: final ensemble plus agreement of all base layers.
        for idx, direction, target_idx in ((2, "home", 2), (1, "draw", 1), (0, "away", 0)):
            values = [float(values[position, idx]) for values in component_probabilities.values()]
            add(
                "match_result",
                direction,
                float(final_probs[position, idx]),
                _agreement(values),
                0.0,
                int(match["target"]) == target_idx,
            )

        # Goal markets. A second independent estimate based on recent scoring
        # averages gives the selector a model-agreement signal.
        h5 = 0.5 * (float(match.get("home_gf_5", 1.2)) + float(match.get("away_ga_5", 1.2)))
        a5 = 0.5 * (float(match.get("away_gf_5", 1.2)) + float(match.get("home_ga_5", 1.2)))
        h10 = 0.5 * (float(match.get("home_gf_10", 1.2)) + float(match.get("away_ga_10", 1.2)))
        a10 = 0.5 * (float(match.get("away_gf_10", 1.2)) + float(match.get("home_ga_10", 1.2)))
        alt_home = float(np.clip(0.72 * h5 + 0.28 * h10, 0.05, 5.5))
        alt_away = float(np.clip(0.72 * a5 + 0.28 * a10, 0.05, 5.5))
        alt_total = alt_home + alt_away

        # Local import avoids coupling the selector estimator to UI code.
        from .optional_models import poisson_total_probability

        for line in (1.5, 2.5, 3.5, 4.5):
            suffix = str(line).replace(".", "_")
            p_over = float(markets[f"over_{suffix}"])
            p_under = float(markets[f"under_{suffix}"])
            alt_over = poisson_total_probability(alt_total, line, True)
            add(f"total_{line}", "over", p_over, _agreement([p_over, alt_over]), line, total_goals > line)
            add(f"total_{line}", "under", p_under, _agreement([p_under, 1.0 - alt_over]), line, total_goals < line)

        btts_yes = float(markets["btts_yes"])
        alt_btts = float((1.0 - np.exp(-alt_home)) * (1.0 - np.exp(-alt_away)))
        add("btts", "yes", btts_yes, _agreement([btts_yes, alt_btts]), 0.0, home_goals > 0 and away_goals > 0)
        add("btts", "no", 1.0 - btts_yes, _agreement([1.0 - btts_yes, 1.0 - alt_btts]), 0.0, home_goals == 0 or away_goals == 0)

        for team_key, actual, alt_mean in (("home", home_goals, alt_home), ("away", away_goals, alt_away)):
            for line in (1.5, 2.5):
                suffix = str(line).replace(".", "_")
                p_over = float(markets[f"{team_key}_over_{suffix}"])
                alt_over = poisson_total_probability(alt_mean, line, True)
                add("team_total", f"{team_key}_over", p_over, _agreement([p_over, alt_over]), line, actual > line)
                add("team_total", f"{team_key}_under", 1.0 - p_over, _agreement([1.0 - p_over, 1.0 - alt_over]), line, actual < line)

    return pd.DataFrame(rows)
