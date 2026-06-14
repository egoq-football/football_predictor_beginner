from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
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
BASE_COMPONENT_ORDER = ["ml", "dixon_coles", "elo", "fifa", "recent_form"]
META_DERIVED_COLUMNS = [
    "consensus_away", "consensus_draw", "consensus_home",
    "disagreement_away", "disagreement_draw", "disagreement_home",
    "leader_agreement", "consensus_entropy", "consensus_margin",
]


def _aligned_proba(estimator, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    raw = estimator.predict_proba(X)
    out = np.zeros((len(X), 3), dtype=float)
    for idx, cls in enumerate(estimator.classes_):
        out[:, int(cls)] = raw[:, idx]
    out = np.clip(out, EPS, None)
    return out / out.sum(axis=1, keepdims=True)


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    target = np.eye(3)[y_true.astype(int)]
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))


def class_brier_scores(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    target = np.eye(3)[y_true.astype(int)]
    return np.mean((probs - target) ** 2, axis=0)


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    """Macro one-vs-rest ECE for away/draw/home probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    class_values: list[float] = []
    for cls in range(3):
        actual = (y_true == cls).astype(float)
        confidence = probs[:, cls]
        score = 0.0
        for idx in range(bins):
            left, right = edges[idx], edges[idx + 1]
            mask = (confidence >= left) & (confidence < right if idx < bins - 1 else confidence <= right)
            if not np.any(mask):
                continue
            score += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(actual[mask].mean()))
        class_values.append(score)
    return float(np.mean(class_values))


class VectorScaler:
    """Class-specific probability calibration.

    Unlike a single temperature, this can correct a systematic draw bias without
    forcing the same correction onto home and away wins. Only six regularised
    parameters are learned, which keeps it stable on a modest calibration block.
    """

    def __init__(self) -> None:
        self.scale_ = np.ones(3, dtype=float)
        self.bias_ = np.zeros(3, dtype=float)
        self.success_ = False

    @property
    def temperature_(self) -> float:
        return float(np.mean(1.0 / np.clip(self.scale_, 1e-6, None)))

    def fit(
        self,
        probs: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "VectorScaler":
        probs = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
        y = np.asarray(y, dtype=int)
        weights = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        weights = weights / max(float(weights.mean()), EPS)
        logp = np.log(probs)

        def objective(params: np.ndarray) -> float:
            scale = params[:3]
            bias = params[3:]
            logits = logp * scale + bias
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            calibrated = exp / exp.sum(axis=1, keepdims=True)
            nll = -np.average(np.log(np.clip(calibrated[np.arange(len(y)), y], EPS, 1.0)), weights=weights)
            penalty = 0.012 * float(np.sum((scale - 1.0) ** 2)) + 0.018 * float(np.sum(bias**2))
            return float(nll + penalty)

        result = minimize(
            objective,
            np.concatenate([self.scale_, self.bias_]),
            method="L-BFGS-B",
            bounds=[(0.25, 3.5)] * 3 + [(-2.0, 2.0)] * 3,
        )
        if result.success:
            self.scale_ = np.asarray(result.x[:3], dtype=float)
            self.bias_ = np.asarray(result.x[3:], dtype=float)
            self.success_ = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        probs = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
        logits = np.log(probs) * self.scale_ + self.bias_
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


class ConsensusBlender:
    """Learned safeguard against unstable stacking extrapolation."""

    def __init__(self) -> None:
        self.meta_weight_ = 1.0

    def fit(
        self,
        meta_probs: np.ndarray,
        consensus_probs: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "ConsensusBlender":
        meta_probs = np.asarray(meta_probs, dtype=float)
        consensus_probs = np.asarray(consensus_probs, dtype=float)
        y = np.asarray(y, dtype=int)
        weights = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)

        def objective(alpha: float) -> float:
            blended = alpha * meta_probs + (1.0 - alpha) * consensus_probs
            selected = np.clip(blended[np.arange(len(y)), y], EPS, 1.0)
            return float(-np.average(np.log(selected), weights=weights))

        result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
        self.meta_weight_ = float(result.x if result.success else 1.0)
        return self

    def transform(self, meta_probs: np.ndarray, consensus_probs: np.ndarray) -> np.ndarray:
        output = self.meta_weight_ * np.asarray(meta_probs, dtype=float) + (1.0 - self.meta_weight_) * np.asarray(consensus_probs, dtype=float)
        output = np.clip(output, EPS, None)
        return output / output.sum(axis=1, keepdims=True)


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
    calibrator: VectorScaler = field(default_factory=VectorScaler)
    blender: ConsensusBlender = field(default_factory=ConsensusBlender)
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
        vector = _meta_matrix_from_parts(
            {name: values.reshape(1, -1) for name, values in components.items()},
            pd.DataFrame([row[META_CONTEXT_COLUMNS].to_dict()]),
        )[0]
        return vector, components

    def predict(self, features: pd.DataFrame, home: str, away: str, neutral: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        row = features.iloc[0]
        vector, components = self._base_vector(row, home, away, neutral)
        raw = _aligned_proba(self.meta_model, pd.DataFrame([vector]))
        calibrated = self.calibrator.transform(raw)
        consensus = _consensus_from_parts({name: values.reshape(1, -1) for name, values in components.items()})
        final = self.blender.transform(calibrated, consensus)[0]
        return final, components


def _new_ml_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=700, C=0.62, class_weight=None, solver="lbfgs"
        )),
    ])


def _training_weights(table: pd.DataFrame) -> np.ndarray:
    dates = pd.to_datetime(table["date"])
    latest = dates.max()
    age_years = (latest - dates).dt.days.to_numpy(dtype=float) / 365.25
    recency = 0.22 + 0.78 * np.power(0.5, age_years / 4.5)

    importance = pd.to_numeric(table.get("importance", 0.65), errors="coerce").fillna(0.65).to_numpy(dtype=float)
    competition = 0.58 + 0.72 * np.clip(importance, 0.35, 1.0)
    world_cup = np.where(pd.to_numeric(table.get("is_world_cup", 0), errors="coerce").fillna(0).to_numpy() > 0, 1.12, 1.0)
    friendly = np.where(pd.to_numeric(table.get("is_friendly", 0), errors="coerce").fillna(0).to_numpy() > 0, 0.78, 1.0)

    weights = recency * competition * world_cup * friendly
    weights = np.clip(weights, 0.12, 2.25)
    return weights / max(float(weights.mean()), EPS)


def _fit_ml(model: Pipeline, table: pd.DataFrame) -> Pipeline:
    model.fit(
        table[FEATURE_COLUMNS],
        table["target"].astype(int),
        model__sample_weight=_training_weights(table),
    )
    return model


def _meta_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1500, C=0.80, class_weight=None, solver="lbfgs")),
    ])


def _component_matrix(parts: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([parts[name] for name in BASE_COMPONENT_ORDER], axis=1)


def _consensus_from_parts(parts: dict[str, np.ndarray]) -> np.ndarray:
    # Recent-form probabilities are deliberately excluded from the safeguard:
    # they are useful as a feature but materially weaker as a standalone model.
    selected = np.stack([parts[name] for name in ("ml", "dixon_coles", "elo", "fifa")], axis=1)
    consensus = selected.mean(axis=1)
    consensus = np.clip(consensus, EPS, None)
    return consensus / consensus.sum(axis=1, keepdims=True)


def _derived_meta_features(parts: dict[str, np.ndarray]) -> np.ndarray:
    matrix = _component_matrix(parts)
    consensus = matrix.mean(axis=1)
    disagreement = matrix.std(axis=1)
    leaders = matrix.argmax(axis=2)
    agreement = np.array([
        np.bincount(row, minlength=3).max() / len(BASE_COMPONENT_ORDER)
        for row in leaders
    ], dtype=float).reshape(-1, 1)
    entropy = (-np.sum(consensus * np.log(np.clip(consensus, EPS, 1.0)), axis=1) / np.log(3.0)).reshape(-1, 1)
    sorted_probs = np.sort(consensus, axis=1)
    margin = (sorted_probs[:, -1] - sorted_probs[:, -2]).reshape(-1, 1)
    return np.hstack([consensus, disagreement, agreement, entropy, margin])


def _meta_matrix_from_parts(parts: dict[str, np.ndarray], context: pd.DataFrame) -> np.ndarray:
    return np.hstack([
        *[parts[name] for name in BASE_COMPONENT_ORDER],
        context[META_CONTEXT_COLUMNS].to_numpy(dtype=float),
        _derived_meta_features(parts),
    ])


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
    return _meta_matrix_from_parts(parts, table), parts


def _metric(name: str, y: np.ndarray, probs: np.ndarray) -> dict:
    probs = np.clip(probs, EPS, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    class_brier = class_brier_scores(y, probs)
    actual_share = np.bincount(y.astype(int), minlength=3) / max(len(y), 1)
    predicted_share = probs.mean(axis=0)
    return {
        "model": name,
        "matches": int(len(y)),
        "accuracy": float(accuracy_score(y, probs.argmax(axis=1))),
        "log_loss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "brier": multiclass_brier(y, probs),
        "ece": expected_calibration_error(y, probs),
        "away_brier": float(class_brier[0]),
        "draw_brier": float(class_brier[1]),
        "home_brier": float(class_brier[2]),
        "actual_draw_share": float(actual_share[1]),
        "predicted_draw_share": float(predicted_share[1]),
    }


def _expanding_oof_meta(table: pd.DataFrame, start: int, end: int, folds: int = 4) -> tuple[np.ndarray, pd.DataFrame]:
    boundaries = np.linspace(start, end, folds + 1, dtype=int)
    matrices: list[np.ndarray] = []
    rows: list[pd.DataFrame] = []
    for idx in range(folds):
        train_end = int(boundaries[idx])
        valid_end = int(boundaries[idx + 1])
        train = table.iloc[:train_end]
        valid = table.iloc[train_end:valid_end]
        if len(train) < 800 or valid.empty:
            continue
        ml = _fit_ml(_new_ml_model(), train)
        dc = DixonColesModel().fit(train)
        meta_x, _ = _base_predictions(valid, ml, dc)
        matrices.append(meta_x)
        rows.append(valid)
    if not matrices:
        raise ValueError("Не удалось построить хронологические OOF-прогнозы для метамодели.")
    return np.vstack(matrices), pd.concat(rows, ignore_index=True)


def train_bundle(table: pd.DataFrame) -> WorldCupModelBundle:
    table = table.sort_values("date").reset_index(drop=True)
    n = len(table)
    if n < 800:
        raise ValueError("Для обучения требуется минимум 800 матчей.")

    meta_start = int(n * 0.50)
    meta_end = int(n * 0.82)
    calibration_end = int(n * 0.86)
    blend_end = int(n * 0.90)

    calibration = table.iloc[meta_end:calibration_end]
    blend = table.iloc[calibration_end:blend_end]
    test = table.iloc[blend_end:]

    # Expanding-window out-of-fold predictions prevent the stacker from learning
    # on in-sample base-model probabilities and reduce deployment distribution shift.
    meta_x, meta_rows = _expanding_oof_meta(table, meta_start, meta_end, folds=4)
    meta = _meta_model()
    meta.fit(meta_x, meta_rows["target"].astype(int), model__sample_weight=_training_weights(meta_rows))

    # Calibrate on a strictly later block.
    pre_cal = table.iloc[:meta_end]
    ml_cal = _fit_ml(_new_ml_model(), pre_cal)
    dc_cal = DixonColesModel().fit(pre_cal)
    cal_x, _ = _base_predictions(calibration, ml_cal, dc_cal)
    cal_raw = _aligned_proba(meta, pd.DataFrame(cal_x))
    calibrator = VectorScaler().fit(
        cal_raw,
        calibration["target"].astype(int).to_numpy(),
        sample_weight=_training_weights(calibration),
    )

    # Learn how much to trust the calibrated stacker versus the robust consensus
    # on another later block, without any hand-written ensemble weight.
    pre_blend = table.iloc[:calibration_end]
    ml_blend = _fit_ml(_new_ml_model(), pre_blend)
    dc_blend = DixonColesModel().fit(pre_blend)
    blend_x, blend_parts = _base_predictions(blend, ml_blend, dc_blend)
    blend_raw = _aligned_proba(meta, pd.DataFrame(blend_x))
    blend_calibrated = calibrator.transform(blend_raw)
    blender = ConsensusBlender().fit(
        blend_calibrated,
        _consensus_from_parts(blend_parts),
        blend["target"].astype(int).to_numpy(),
        sample_weight=_training_weights(blend),
    )

    # Honest final holdout: every learned component is frozen before this block.
    pretest = table.iloc[:blend_end]
    ml_eval = _fit_ml(_new_ml_model(), pretest)
    dc_eval = DixonColesModel().fit(pretest)
    test_x, test_parts = _base_predictions(test, ml_eval, dc_eval)
    raw_final = _aligned_proba(meta, pd.DataFrame(test_x))
    calibrated_final = calibrator.transform(raw_final)
    final_probs = blender.transform(calibrated_final, _consensus_from_parts(test_parts))
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

    selector_frame = build_historical_selector_frame(test, final_probs, test_parts, dc_eval)
    outcome_selector = OutcomeSelectorModel().fit(selector_frame)

    ml_final = _fit_ml(_new_ml_model(), table)
    dc_final = DixonColesModel().fit(table)
    optional_models = OptionalMarketModels().fit(table)

    return WorldCupModelBundle(
        version=MODEL_VERSION,
        ml_model=ml_final,
        meta_model=meta,
        calibrator=calibrator,
        blender=blender,
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
    required = ("ml_model", "meta_model", "calibrator", "blender", "dixon_coles", "optional_models", "outcome_selector")
    if version != MODEL_VERSION or any(not hasattr(bundle, name) for name in required):
        raise ValueError("Сохранённая модель несовместима с текущей версией.")
    return bundle
