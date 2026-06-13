from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

import numpy as np
import pandas as pd

from .odds_provider import MarketSnapshot
from .optional_models import poisson_total_probability


MAX_NON_OBVIOUS_PROBABILITY = 1.0 / 1.40
MIN_SELECTION_PROBABILITY = 0.52
MIN_MARKET_EDGE = 0.03
MIN_AGREEMENT = 0.52
MIN_DATA_QUALITY = 0.55


@dataclass
class OutcomeCandidate:
    category: str
    label: str
    probability: float
    model: str
    market_key: str
    agreement: float
    data_quality: float
    score: float = 0.0
    market_edge: float | None = None
    market_checked: bool = False
    reason: str = ""

    def public_dict(self) -> dict[str, Any]:
        return {
            "Категория": self.category,
            "Исход": self.label,
            "Вероятность": self.probability,
            "Уверенность": confidence_label(self.score),
            "Основание": self.reason,
            "Модель": self.model,
            "Проверка": "рыночная" if self.market_checked else "математическая",
        }


def confidence_label(score: float) -> str:
    if score >= 0.72:
        return "высокая"
    if score >= 0.61:
        return "выше средней"
    if score >= 0.53:
        return "средняя"
    return "ограниченная"


def _prob_from_components(components: dict[str, dict[str, float]], side: str) -> tuple[float, float]:
    values = [float(v.get(side, np.nan)) for v in components.values() if pd.notna(v.get(side, np.nan))]
    if not values:
        return 0.5, 0.45
    mean = float(np.mean(values))
    agreement = float(np.clip(1.0 - np.std(values) / 0.22, 0.20, 1.0))
    return mean, agreement


def _alt_goal_means(features: dict[str, float]) -> tuple[float, float]:
    h5 = 0.5 * (float(features.get("home_gf_5", 1.2)) + float(features.get("away_ga_5", 1.2)))
    a5 = 0.5 * (float(features.get("away_gf_5", 1.2)) + float(features.get("home_ga_5", 1.2)))
    h10 = 0.5 * (float(features.get("home_gf_10", 1.2)) + float(features.get("away_ga_10", 1.2)))
    a10 = 0.5 * (float(features.get("away_gf_10", 1.2)) + float(features.get("home_ga_10", 1.2)))
    return float(np.clip(0.72 * h5 + 0.28 * h10, 0.05, 5.5)), float(np.clip(0.72 * a5 + 0.28 * a10, 0.05, 5.5))


def _binary_agreement(primary: float, secondary: float | None) -> float:
    if secondary is None:
        return 0.58
    return float(np.clip(1.0 - abs(float(primary) - float(secondary)) / 0.30, 0.25, 1.0))


def _data_quality(features: dict[str, float], lineups_known: bool, optional_required: bool = False) -> float:
    parts = [
        1.0 if float(features.get("fifa_available", 0.0)) >= 0.5 else 0.45,
        min(1.0, min(float(features.get("home_matches", 0.0)), float(features.get("away_matches", 0.0))) / 25.0),
        1.0 if lineups_known else 0.72,
    ]
    if optional_required:
        parts.append(1.0 if float(features.get("optional_stats_available", 0.0)) >= 0.5 else 0.45)
    return float(np.clip(np.mean(parts), 0.25, 1.0))


def _candidate(
    category: str,
    label: str,
    probability: float,
    model: str,
    market_key: str,
    agreement: float,
    quality: float,
    reason: str,
) -> OutcomeCandidate:
    return OutcomeCandidate(category, label, float(probability), model, market_key, float(agreement), float(quality), reason=reason)


def _add_total_candidates(
    rows: list[OutcomeCandidate],
    category_prefix: str,
    name_prefix: str,
    mean_total: float,
    model_name: str,
    quality: float,
    lines: tuple[float, ...],
    probability_fn=None,
    alt_mean: float | None = None,
    key_prefix: str = "",
) -> None:
    for line in lines:
        if probability_fn is None:
            over = poisson_total_probability(mean_total, line, True)
        else:
            over = float(probability_fn(mean_total, line, True))
        under = 1.0 - over
        alt_over = poisson_total_probability(alt_mean, line, True) if alt_mean is not None else None
        suffix = str(line).replace(".", "_")
        rows.append(_candidate(
            f"{category_prefix} {str(line).replace('.', ',')}",
            f"{name_prefix} больше {str(line).replace('.', ',')}",
            over,
            model_name,
            f"{key_prefix}over_{suffix}",
            _binary_agreement(over, alt_over),
            quality,
            "Вероятность рассчитана отдельной моделью количества событий и проверена на хронологической выборке.",
        ))
        rows.append(_candidate(
            f"{category_prefix} {str(line).replace('.', ',')}",
            f"{name_prefix} меньше {str(line).replace('.', ',')}",
            under,
            model_name,
            f"{key_prefix}under_{suffix}",
            _binary_agreement(under, None if alt_over is None else 1.0 - alt_over),
            quality,
            "Вероятность рассчитана отдельной моделью количества событий и проверена на хронологической выборке.",
        ))


def build_candidates(
    home: str,
    away: str,
    final_probs: np.ndarray,
    components: dict[str, dict[str, float]],
    markets: dict[str, Any],
    halftime: dict[str, Any],
    second_half: dict[str, Any],
    corners: dict[str, Any],
    cards: dict[str, Any],
    features: dict[str, float],
    lineups_known: bool,
) -> list[OutcomeCandidate]:
    away_p, draw_p, home_p = map(float, final_probs)
    base_quality = _data_quality(features, lineups_known)
    optional_quality = _data_quality(features, lineups_known, optional_required=True)
    rows: list[OutcomeCandidate] = []

    for side, label, probability, market_key in (
        ("home", f"Победа {home}", home_p, "home_win"),
        ("draw", "Ничья", draw_p, "draw"),
        ("away", f"Победа {away}", away_p, "away_win"),
    ):
        _, agreement = _prob_from_components(components, side)
        rows.append(_candidate(
            "Исход матча", label, probability, "Калиброванный ансамбль", market_key,
            agreement, base_quality,
            "Результат подтверждается метамоделью, рейтингами, формой и моделью голов.",
        ))

    alt_home, alt_away = _alt_goal_means(features)
    alt_total = alt_home + alt_away
    for line in (1.5, 2.5, 3.5, 4.5):
        suffix = str(line).replace(".", "_")
        over = float(markets[f"over_{suffix}"])
        under = float(markets[f"under_{suffix}"])
        alt_over = poisson_total_probability(alt_total, line, True)
        rows.append(_candidate(
            f"Тотал {str(line).replace('.', ',')}", f"Тотал матча больше {str(line).replace('.', ',')}", over,
            "Dixon–Coles + форма результативности", f"over_{suffix}", _binary_agreement(over, alt_over), base_quality,
            "Dixon–Coles сопоставлен со средними забитыми и пропущенными голами обеих команд.",
        ))
        rows.append(_candidate(
            f"Тотал {str(line).replace('.', ',')}", f"Тотал матча меньше {str(line).replace('.', ',')}", under,
            "Dixon–Coles + форма результативности", f"under_{suffix}", _binary_agreement(under, 1.0 - alt_over), base_quality,
            "Dixon–Coles сопоставлен со средними забитыми и пропущенными голами обеих команд.",
        ))

    btts = float(markets["btts_yes"])
    alt_btts = (1.0 - np.exp(-alt_home)) * (1.0 - np.exp(-alt_away))
    rows.append(_candidate("Обе забьют", "Обе команды забьют — да", btts, "Dixon–Coles", "btts_yes", _binary_agreement(btts, alt_btts), base_quality, "Учитываются атакующая и оборонительная сила обеих команд."))
    rows.append(_candidate("Обе забьют", "Обе команды забьют — нет", 1.0 - btts, "Dixon–Coles", "btts_no", _binary_agreement(1.0 - btts, 1.0 - alt_btts), base_quality, "Учитываются атакующая и оборонительная сила обеих команд."))

    for team_key, team_name, alt_mean in (("home", home, alt_home), ("away", away, alt_away)):
        for line in (1.5, 2.5):
            suffix = str(line).replace(".", "_")
            over = float(markets[f"{team_key}_over_{suffix}"])
            alt_over = poisson_total_probability(alt_mean, line, True)
            rows.append(_candidate(
                "Индивидуальный тотал", f"{team_name}: тотал больше {str(line).replace('.', ',')}", over,
                "Dixon–Coles + форма атаки", f"{team_key}_over_{suffix}", _binary_agreement(over, alt_over), base_quality,
                "Сопоставлены результативность команды и оборона соперника.",
            ))
            rows.append(_candidate(
                "Индивидуальный тотал", f"{team_name}: тотал меньше {str(line).replace('.', ',')}", 1.0 - over,
                "Dixon–Coles + форма атаки", f"{team_key}_under_{suffix}", _binary_agreement(1.0 - over, 1.0 - alt_over), base_quality,
                "Сопоставлены результативность команды и оборона соперника.",
            ))

    if halftime.get("available"):
        for label, key in ((f"Победа {home} в первом тайме", "home_win"), ("Ничья в первом тайме", "draw"), (f"Победа {away} в первом тайме", "away_win")):
            rows.append(_candidate("Первый тайм", label, float(halftime[key]), "Отдельная калиброванная модель первого тайма", f"ht_{key}", 0.70, optional_quality, "Модель обучена только на счётах первого тайма."))
        _add_total_candidates(rows, "Тотал первого тайма", "Тотал первого тайма", float(halftime["home_mean"] + halftime["away_mean"]), "Отдельная модель первого тайма", optional_quality, (0.5, 1.5, 2.5), halftime.get("probability_fn"), key_prefix="ht_")

    if second_half.get("available"):
        for label, key in ((f"Победа {home} во втором тайме", "home_win"), ("Ничья во втором тайме", "draw"), (f"Победа {away} во втором тайме", "away_win")):
            rows.append(_candidate("Второй тайм", label, float(second_half[key]), "Отдельная калиброванная модель второго тайма", f"sh_{key}", 0.68, optional_quality, "Модель обучена только на событиях после перерыва."))
        _add_total_candidates(rows, "Тотал второго тайма", "Тотал второго тайма", float(second_half["home_mean"] + second_half["away_mean"]), "Отдельная модель второго тайма", optional_quality, (0.5, 1.5, 2.5), second_half.get("probability_fn"), key_prefix="sh_")

    if corners.get("available"):
        _add_total_candidates(rows, "Угловые", "Тотал угловых", float(corners["total_mean"]), "Отдельная калиброванная модель угловых", optional_quality, (7.5, 8.5, 9.5, 10.5), corners.get("probability_fn"), key_prefix="corners_")

    if cards.get("available"):
        _add_total_candidates(rows, "Жёлтые карточки", "Тотал жёлтых карточек", float(cards["total_mean"]), "Отдельная калиброванная модель карточек", optional_quality, (2.5, 3.5, 4.5, 5.5), cards.get("probability_fn"), key_prefix="cards_")

    return rows


def _is_structurally_obvious(candidate: OutcomeCandidate, features: dict[str, float]) -> bool:
    # Approximate lower price boundary 1.40 when no market line is available.
    if candidate.probability > MAX_NON_OBVIOUS_PROBABILITY:
        return True

    if candidate.category == "Исход матча":
        fifa_gap = abs(float(features.get("fifa_points_diff", 0.0)))
        elo_gap = abs(float(features.get("elo_diff", 0.0)))
        # Do not call a clear favourite's straight win the best "non-obvious" idea.
        if candidate.probability >= 0.60 or fifa_gap >= 85 or elo_gap >= 125:
            return True

    # Extremely safe unders and team unders are structurally obvious even when
    # their raw probability is just below the general threshold.
    if candidate.market_key in {"under_4_5", "home_under_2_5", "away_under_2_5"} and candidate.probability >= 0.66:
        return True
    return False


def _canonical_selector_features(candidate: OutcomeCandidate, features: dict[str, float]) -> dict[str, Any]:
    key = candidate.market_key
    line_match = re.search(r"(\d+_\d+)$", key)
    line = float(line_match.group(1).replace("_", ".")) if line_match else 0.0
    if key in {"home_win", "draw", "away_win"}:
        category = "match_result"
        direction = {"home_win": "home", "draw": "draw", "away_win": "away"}[key]
    elif key.startswith("btts_"):
        category = "btts"
        direction = key.removeprefix("btts_")
    elif key.startswith(("home_over_", "home_under_", "away_over_", "away_under_")):
        category = "team_total"
        direction = key.rsplit("_", 2)[0]
    elif key.startswith(("over_", "under_")):
        category = f"total_{line}"
        direction = "over" if key.startswith("over_") else "under"
    elif key.startswith("ht_"):
        category = "halftime"
        direction = key.removeprefix("ht_").rsplit("_", 2)[0]
    elif key.startswith("sh_"):
        category = "second_half"
        direction = key.removeprefix("sh_").rsplit("_", 2)[0]
    elif key.startswith("corners_"):
        category = "corners"
        direction = "over" if "over" in key else "under"
    elif key.startswith("cards_"):
        category = "cards"
        direction = "over" if "over" in key else "under"
    else:
        category = "other"
        direction = "unknown"

    alt_home, alt_away = _alt_goal_means(features)
    return {
        "category": category,
        "direction": direction,
        "probability": candidate.probability,
        "agreement": candidate.agreement,
        "data_quality": candidate.data_quality,
        "abs_elo_diff": abs(float(features.get("elo_diff", 0.0))),
        "abs_fifa_diff": abs(float(features.get("fifa_points_diff", 0.0))),
        "expected_total": alt_home + alt_away,
        "line": line,
    }


def select_non_obvious_outcomes(
    candidates: list[OutcomeCandidate],
    market_snapshot: MarketSnapshot | None,
    features: dict[str, float],
    selector_model: Any | None = None,
) -> dict[str, Any]:
    market_snapshot = market_snapshot or MarketSnapshot()
    eligible: list[OutcomeCandidate] = []
    feature_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        if (
            candidate.probability < MIN_SELECTION_PROBABILITY
            or candidate.agreement < MIN_AGREEMENT
            or candidate.data_quality < MIN_DATA_QUALITY
            or _is_structurally_obvious(candidate, features)
        ):
            continue

        quote = market_snapshot.quote(candidate.market_key) if market_snapshot.available else None
        if quote is not None:
            candidate.market_checked = True
            candidate.market_edge = candidate.probability - quote.fair_probability
            if quote.decimal_odds < 1.40 or candidate.market_edge < MIN_MARKET_EDGE:
                continue
            candidate.reason += " Рыночный консенсус использован только как скрытая проверка преимущества модели."
        else:
            # Внутренний фильтр сохраняется, но техническое пояснение не выводится пользователю.
            pass

        eligible.append(candidate)
        feature_rows.append(_canonical_selector_features(candidate, features))

    selector_active = bool(selector_model is not None and getattr(selector_model, "estimator", None) is not None)
    if eligible:
        if selector_active:
            learned = selector_model.predict_success_probability(pd.DataFrame(feature_rows))
            for candidate, score in zip(eligible, learned):
                candidate.score = float(score)
        else:
            # Compatibility fallback for an old bundle. New v4.3 bundles include
            # a trained selector; this branch avoids crashing during redeploy.
            for candidate in eligible:
                candidate.score = float(candidate.probability)

    eligible.sort(
        key=lambda row: (row.score, row.market_edge if row.market_edge is not None else -1.0, row.probability),
        reverse=True,
    )
    selected = eligible[0] if eligible else None
    alternatives: list[OutcomeCandidate] = []
    if selected is not None:
        seen_categories = {selected.category}
        for candidate in eligible[1:]:
            # Prefer alternatives from different market families instead of
            # showing three near-identical totals.
            if candidate.category in seen_categories and len(alternatives) < 2:
                continue
            alternatives.append(candidate)
            seen_categories.add(candidate.category)
            if len(alternatives) >= 3:
                break
    selector_status = selector_model.status() if selector_model is not None and hasattr(selector_model, "status") else {}
    return {
        "found": selected is not None,
        "best": selected.public_dict() if selected else None,
        "alternatives": [row.public_dict() for row in alternatives],
        "market_check_available": bool(market_snapshot.available),
        "market_message": market_snapshot.message,
        "eligible_count": len(eligible),
        "selector_status": selector_status,
        "message": (
            "Выбран исход с наивысшей исторически обученной оценкой надёжности среди неочевидных кандидатов."
            if selected else
            "Подходящий неочевидный исход не найден: система не будет принудительно предлагать слабый вариант."
        ),
    }

