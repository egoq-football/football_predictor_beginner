from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_BUNDLE_PATH, MODEL_VERSION
from .dixon_coles import DixonColesModel
from .elo import three_way_probabilities
from .features import FEATURE_COLUMNS, fifa_probabilities, recent_form_probabilities
from .optional_models import OptionalMarketModels
from .outcome_selector import OutcomeSelectorModel, build_historical_selector_frame


EPS = 1e-9
META_CONTEXT_COLUMNS = [
    "abs_elo_diff", "abs_fifa_points_diff", "fifa_available",
    "context_group_stage", "context_knockout", "context_group_round",
    "context_points_diff", "context_goal_difference_diff",
    "context_must_win_diff", "context_draw_enough_diff",
    "context_rotation_diff", "context_rest_diff", "context_lineups_known",
    "optional_stats_available", "lineup_data_available",
]


def _aligned_proba(estimator, X: pd.DataFrame) -> np.ndarray:
    raw = estimator.predict_proba(X)
    out = np.zeros((len(X), 3), dtype=float)
    for idx, cls in enumerate(estimator.classes_):
        out[:, int(cls)] = raw[:, idx]
    out = np.clip(out, EPS, None)
    return out / out.sum(axis=1, keepdims=True)


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    target = np.eye(3)[y_true.astype(int)]
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))


class TemperatureScaler:
    def __init__(self) -> None:
        self.temperature_ = 1.0

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        probs = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
        y = np.asarray(y, dtype=int)

        def objective(temp: float) -> float:
            scaled = self.transform(probs, temperature=temp)
            return float(log_loss(y, scaled, labels=[0, 1, 2]))

        result = minimize_scalar(objective, bounds=(0.35, 4.0), method="bounded")
        self.temperature_ = float(result.x if result.success else 1.0)
        return self

    def transform(self, probs: np.ndarray, temperature: float | None = None) -> np.ndarray:
        temp = float(temperature if temperature is not None else self.temperature_)
        logits = np.log(np.clip(probs, EPS, 1.0)) / max(temp, 1e-3)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class ModelMetrics:
    model: str
    matches: int
    accuracy: float
    log_loss: float
    brier: float


@dataclass
class WorldCupModelBundle:
    version: str = MODEL_VERSION
    ml_model: object | None = None
    meta_model: object | None = None
    calibrator: TemperatureScaler = field(default_factory=TemperatureScaler)
    dixon_coles: DixonColesModel = field(default_factory=DixonColesModel)
    optional_models: OptionalMarketModels = field(default_factory=OptionalMarketModels)
    outcome_selector: OutcomeSelectorModel = field(default_factory=OutcomeSelectorModel)
    metrics: list[dict] = field(default_factory=list)
    train_end_date: str = ""
    training_matches: int = 0
    data_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def _base_vector(self, row: pd.Series, home: str, away: str, neutral: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        ml = _aligned_proba(self.ml_model, frame[FEATURE_COLUMNS])[0]
        dc = self.dixon_coles.predict(home, away, neutral).probabilities_1x2
        elo_home, elo_draw, elo_away = three_way_probabilities(float(row["home_elo"]), float(row["away_elo"]), neutral)
        elo = np.array([elo_away, elo_draw, elo_home], dtype=float)
        fifa = fifa_probabilities(row)
        recent = recent_form_probabilities(row)
        components = {"ml": ml, "dixon_coles": dc, "elo": elo, "fifa": fifa, "recent_form": recent}
        vector = np.concatenate([ml, dc, elo, fifa, recent, row[META_CONTEXT_COLUMNS].to_numpy(dtype=float)])
        return vector, components

    def predict(self, features: pd.DataFrame, home: str, away: str, neutral: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        row = features.iloc[0]
        vector, components = self._base_vector(row, home, away, neutral)
        raw = _aligned_proba(self.meta_model, pd.DataFrame([vector]))
        calibrated = self.calibrator.transform(raw)[0]
        return calibrated, components


def _new_ml_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=500, C=0.55, class_weight="balanced", solver="lbfgs"
        )),
    ])


def _fit_ml(model: Pipeline, table: pd.DataFrame) -> Pipeline:
    model.fit(
        table[FEATURE_COLUMNS],
        table["target"].astype(int),
        model__sample_weight=_sample_weights(table["date"]),
    )
    return model


def _meta_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1200, C=0.65, class_weight="balanced")),
    ])


def _sample_weights(dates: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(dates)
    latest = dates.max()
    age_years = (latest - dates).dt.days.to_numpy(dtype=float) / 365.25
    return 0.18 + 0.82 * np.power(0.5, age_years / 5.0)


def _base_predictions(
    table: pd.DataFrame,
    ml_model,
    dc_model: DixonColesModel,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ml = _aligned_proba(ml_model, table[FEATURE_COLUMNS])
    dc_rows: list[np.ndarray] = []
    elo_rows: list[np.ndarray] = []
    fifa_rows: list[np.ndarray] = []
    recent_rows: list[np.ndarray] = []
    for _, row in table.iterrows():
        dc_rows.append(dc_model.predict(str(row["home_team"]), str(row["away_team"]), bool(row["neutral"])).probabilities_1x2)
        eh, ed, ea = three_way_probabilities(float(row["home_elo"]), float(row["away_elo"]), bool(row["neutral"]))
        elo_rows.append(np.array([ea, ed, eh], dtype=float))
        fifa_rows.append(fifa_probabilities(row))
        recent_rows.append(recent_form_probabilities(row))
    parts = {
        "ml": ml,
        "dixon_coles": np.vstack(dc_rows),
        "elo": np.vstack(elo_rows),
        "fifa": np.vstack(fifa_rows),
        "recent_form": np.vstack(recent_rows),
    }
    meta_x = np.hstack([
        parts["ml"], parts["dixon_coles"], parts["elo"], parts["fifa"], parts["recent_form"],
        table[META_CONTEXT_COLUMNS].to_numpy(dtype=float),
    ])
    return meta_x, parts


def _metric(name: str, y: np.ndarray, probs: np.ndarray) -> dict:
    probs = np.clip(probs, EPS, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return {
        "model": name,
        "matches": int(len(y)),
        "accuracy": float(accuracy_score(y, probs.argmax(axis=1))),
        "log_loss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "brier": multiclass_brier(y, probs),
    }


def train_bundle(table: pd.DataFrame) -> WorldCupModelBundle:
    table = table.sort_values("date").reset_index(drop=True)
    n = len(table)
    if n < 800:
        raise ValueError("Для обучения требуется минимум 800 матчей.")

    base_end = int(n * 0.70)
    meta_end = int(n * 0.82)
    calibration_end = int(n * 0.88)

    base_train = table.iloc[:base_end]
    meta_train = table.iloc[base_end:meta_end]
    calibration = table.iloc[meta_end:calibration_end]
    test = table.iloc[calibration_end:]

    ml_base = _new_ml_model()
    _fit_ml(ml_base, base_train)
    dc_base = DixonColesModel().fit(base_train)

    meta_x, _ = _base_predictions(meta_train, ml_base, dc_base)
    meta = _meta_model()
    meta.fit(meta_x, meta_train["target"].astype(int))

    cal_x, _ = _base_predictions(calibration, ml_base, dc_base)
    cal_raw = _aligned_proba(meta, pd.DataFrame(cal_x))
    calibrator = TemperatureScaler().fit(cal_raw, calibration["target"].astype(int).to_numpy())

    # Honest final holdout: update base models with all data available before it.
    pretest = table.iloc[:calibration_end]
    ml_eval = _new_ml_model()
    _fit_ml(ml_eval, pretest)
    dc_eval = DixonColesModel().fit(pretest)
    test_x, test_parts = _base_predictions(test, ml_eval, dc_eval)
    raw_final = _aligned_proba(meta, pd.DataFrame(test_x))
    final_probs = calibrator.transform(raw_final)
    y_test = test["target"].astype(int).to_numpy()

    metrics = [_metric("Итоговый ансамбль", y_test, final_probs)]
    display_names = {
        "ml": "Машинное обучение",
        "dixon_coles": "Dixon–Coles",
        "elo": "Elo",
        "fifa": "FIFA",
        "recent_form": "Последние матчи",
    }
    for key, probs in test_parts.items():
        metrics.append(_metric(display_names[key], y_test, probs))
    home_baseline = np.tile(np.array([0.18, 0.25, 0.57]), (len(test), 1))
    metrics.append(_metric("Всегда первая команда", y_test, home_baseline))

    # Train the non-obvious-outcome selector only on genuinely out-of-sample
    # candidate predictions from the final chronological holdout. Its internal
    # train/calibration/validation split is chronological by match.
    selector_frame = build_historical_selector_frame(test, final_probs, test_parts, dc_eval)
    outcome_selector = OutcomeSelectorModel().fit(selector_frame)

    # Refit deployable base models on all completed data.
    ml_final = _new_ml_model()
    _fit_ml(ml_final, table)
    dc_final = DixonColesModel().fit(table)
    optional_models = OptionalMarketModels().fit(table)

    return WorldCupModelBundle(
        version=MODEL_VERSION,
        ml_model=ml_final,
        meta_model=meta,
        calibrator=calibrator,
        dixon_coles=dc_final,
        optional_models=optional_models,
        outcome_selector=outcome_selector,
        metrics=metrics,
        train_end_date=pd.Timestamp(table["date"].max()).date().isoformat(),
        training_matches=len(table),
    )


def save_bundle(bundle: WorldCupModelBundle, path: str | Path = MODEL_BUNDLE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: str | Path = MODEL_BUNDLE_PATH) -> WorldCupModelBundle:
    bundle = joblib.load(path)
    version = str(getattr(bundle, "version", ""))
    compatible_versions = {MODEL_VERSION, "4.2.1-world-cup-2026"}
    required = ("ml_model", "meta_model", "calibrator", "dixon_coles", "optional_models", "outcome_selector")
    if version not in compatible_versions or any(not hasattr(bundle, name) for name in required):
        raise ValueError("Сохранённая модель несовместима с текущей версией.")
    return bundle
