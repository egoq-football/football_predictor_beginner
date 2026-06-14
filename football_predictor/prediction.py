from __future__ import annotations

from dataclasses import asdict
from uuid import uuid4

import numpy as np
import pandas as pd

from .context import MatchContext, infer_group_motivation
from .data_loader import load_match_lineups, load_player_pool
from .dixon_coles import score_markets
from .features import SequentialFeatureBuilder, current_feature_frame
from .fifa_rankings import FifaRankingHistory
from .lineups import assess_squad
from .market_selector import build_candidates, select_non_obvious_outcomes
from .model import WorldCupModelBundle
from .odds_provider import MarketSnapshot


def _pair_poisson_markets(home_mean: float, away_mean: float, max_goals: int = 10) -> dict[str, float]:
    from math import factorial

    matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i, j] = (
                np.exp(-home_mean) * home_mean**i / factorial(i)
                * np.exp(-away_mean) * away_mean**j / factorial(j)
            )
    total = matrix.sum()
    matrix = matrix / total if total > 0 else matrix
    return {
        "away_win": float(np.triu(matrix, 1).sum()),
        "draw": float(np.trace(matrix)),
        "home_win": float(np.tril(matrix, -1).sum()),
    }


def _market_summary_rows(
    home: str,
    away: str,
    final_probs: np.ndarray,
    markets: dict,
    halftime: dict,
    second_half: dict,
    corners: dict,
    cards: dict,
) -> list[dict]:
    away_p, draw_p, home_p = map(float, final_probs)
    rows: list[dict] = []

    label, probability = max(
        [(f"Победа {home}", home_p), ("Ничья", draw_p), (f"Победа {away}", away_p)],
        key=lambda item: item[1],
    )
    rows.append({"Категория": "Исход матча", "Наиболее вероятный исход": label, "Вероятность": probability, "Модель": "Калиброванный ансамбль"})

    for line in (1.5, 2.5, 3.5, 4.5):
        suffix = str(line).replace(".", "_")
        over = float(markets[f"over_{suffix}"])
        under = float(markets[f"under_{suffix}"])
        label, probability = (
            (f"Тотал больше {str(line).replace('.', ',')}", over)
            if over >= under else
            (f"Тотал меньше {str(line).replace('.', ',')}", under)
        )
        rows.append({"Категория": f"Тотал {str(line).replace('.', ',')}", "Наиболее вероятный исход": label, "Вероятность": probability, "Модель": "Dixon–Coles"})

    btts_yes = float(markets["btts_yes"])
    rows.append({
        "Категория": "Обе забьют",
        "Наиболее вероятный исход": "Обе забьют — да" if btts_yes >= 0.5 else "Обе забьют — нет",
        "Вероятность": max(btts_yes, 1.0 - btts_yes),
        "Модель": "Dixon–Coles",
    })

    top_score, top_probability = markets["top_scorelines"][0]
    rows.append({"Категория": "Точный счёт", "Наиболее вероятный исход": top_score, "Вероятность": float(top_probability), "Модель": "Dixon–Coles"})

    for category, section, labels in (
        ("Исход первого тайма", halftime, (f"Победа {home} в первом тайме", "Ничья в первом тайме", f"Победа {away} в первом тайме")),
        ("Исход второго тайма", second_half, (f"Победа {home} во втором тайме", "Ничья во втором тайме", f"Победа {away} во втором тайме")),
    ):
        if section.get("available"):
            values = [(labels[0], section["home_win"]), (labels[1], section["draw"]), (labels[2], section["away_win"])]
            label, probability = max(values, key=lambda item: item[1])
            rows.append({"Категория": category, "Наиболее вероятный исход": label, "Вероятность": float(probability), "Модель": section.get("model_name", "Отдельная модель")})

    if corners.get("available"):
        values = [("Тотал угловых больше 8,5", corners["over_8_5"]), ("Тотал угловых меньше 8,5", corners["under_8_5"])]
        label, probability = max(values, key=lambda item: item[1])
        rows.append({"Категория": "Угловые 8,5", "Наиболее вероятный исход": label, "Вероятность": float(probability), "Модель": corners.get("model_name", "Модель угловых")})

    if cards.get("available"):
        values = [("Тотал жёлтых карточек больше 3,5", cards["over_3_5"]), ("Тотал жёлтых карточек меньше 3,5", cards["under_3_5"])]
        label, probability = max(values, key=lambda item: item[1])
        rows.append({"Категория": "Жёлтые карточки 3,5", "Наиболее вероятный исход": label, "Вероятность": float(probability), "Модель": cards.get("model_name", "Модель карточек")})
    return rows


def _optional_section(model, prediction, lines: tuple[float, ...], section_kind: str) -> dict:
    if not prediction.available or prediction.home_mean is None or prediction.away_mean is None:
        return {
            "available": False,
            "reason": prediction.reason,
            "status": prediction.status,
            "model_name": prediction.status.get("selected_algorithm", "") if prediction.status else "",
        }

    section = {
        "available": True,
        **_pair_poisson_markets(prediction.home_mean, prediction.away_mean),
        "home_mean": float(prediction.home_mean),
        "away_mean": float(prediction.away_mean),
        "total_mean": float(prediction.home_mean + prediction.away_mean),
        "status": prediction.status,
        "model_name": prediction.status.get("selected_algorithm", "Отдельная модель") if prediction.status else "Отдельная модель",
        "probability_fn": model.total_probability,
    }
    for line in lines:
        suffix = str(line).replace(".", "_")
        section[f"over_{suffix}"] = model.total_probability(section["total_mean"], line, True)
        section[f"under_{suffix}"] = model.total_probability(section["total_mean"], line, False)
    return section


def _effect_label(value: float, weak: float, strong: float, positive_name: str, negative_name: str) -> tuple[str, str]:
    magnitude = abs(value)
    if magnitude < weak:
        return "Нейтрально", "слабое"
    direction = positive_name if value > 0 else negative_name
    return direction, "сильное" if magnitude >= strong else "умеренное"


def predict_world_cup_match(
    bundle: WorldCupModelBundle,
    builder: SequentialFeatureBuilder,
    fifa: FifaRankingHistory,
    home: str,
    away: str,
    match_date: str | pd.Timestamp,
    neutral: bool,
    context: MatchContext,
    selected_home_players: list[str] | None = None,
    selected_away_players: list[str] | None = None,
    player_pool: pd.DataFrame | None = None,
    lineup_history: pd.DataFrame | None = None,
    data_source_notes: list[str] | None = None,
    market_snapshot: MarketSnapshot | None = None,
) -> dict:
    context = infer_group_motivation(context)
    player_pool = player_pool if player_pool is not None else load_player_pool()
    lineup_history = lineup_history if lineup_history is not None else load_match_lineups()

    home_squad = assess_squad(player_pool, home, selected_home_players, lineup_history=lineup_history, match_date=match_date)
    away_squad = assess_squad(player_pool, away, selected_away_players, lineup_history=lineup_history, match_date=match_date)
    lineup_home = home_squad.relative_strength if home_squad.available else None
    lineup_away = away_squad.relative_strength if away_squad.available else None

    features = current_feature_frame(
        builder, home, away, neutral, match_date, context,
        lineup_strength_home=lineup_home,
        lineup_strength_away=lineup_away,
    )
    final_probs, raw_components = bundle.predict(features, home, away, neutral)
    components = {
        name: {"away": float(values[0]), "draw": float(values[1]), "home": float(values[2])}
        for name, values in raw_components.items()
    }

    component_matrix = np.vstack([np.asarray(values, dtype=float) for values in raw_components.values()])
    component_leaders = component_matrix.argmax(axis=1)
    leader_counts = np.bincount(component_leaders, minlength=3)
    leader_agreement = float(leader_counts.max() / max(len(component_leaders), 1))
    component_consensus = component_matrix.mean(axis=0)
    disagreement = float(component_matrix.std(axis=0).mean())
    ensemble_consensus_gap = float(np.max(np.abs(final_probs - component_consensus)))
    if leader_agreement >= 0.80 and disagreement < 0.075 and ensemble_consensus_gap < 0.14:
        stability_label = "высокая"
    elif leader_agreement >= 0.60 and disagreement < 0.115 and ensemble_consensus_gap < 0.20:
        stability_label = "средняя"
    else:
        stability_label = "низкая"
    stability = {
        "label": stability_label,
        "leader_agreement": leader_agreement,
        "mean_disagreement": disagreement,
        "ensemble_consensus_gap": ensemble_consensus_gap,
        "consensus_away": float(component_consensus[0]),
        "consensus_draw": float(component_consensus[1]),
        "consensus_home": float(component_consensus[2]),
    }

    dc_prediction = bundle.dixon_coles.predict(home, away, neutral)
    markets = score_markets(dc_prediction)

    halftime_prediction = bundle.optional_models.halftime.predict(features)
    second_prediction = bundle.optional_models.second_half.predict(features)
    corners_prediction = bundle.optional_models.corners.predict(features)
    cards_prediction = bundle.optional_models.cards.predict(features)

    halftime = _optional_section(bundle.optional_models.halftime, halftime_prediction, (0.5, 1.5, 2.5), "halftime")
    second_half = _optional_section(bundle.optional_models.second_half, second_prediction, (0.5, 1.5, 2.5), "second_half")
    corners = _optional_section(bundle.optional_models.corners, corners_prediction, (7.5, 8.5, 9.5, 10.5), "corners")
    cards = _optional_section(bundle.optional_models.cards, cards_prediction, (2.5, 3.5, 4.5, 5.5), "cards")

    row = features.iloc[0]
    feature_dict = row.to_dict()
    candidates = build_candidates(
        home, away, final_probs, components, markets, halftime, second_half, corners, cards,
        feature_dict, context.lineups_known,
    )
    non_obvious = select_non_obvious_outcomes(
        candidates, market_snapshot, feature_dict, selector_model=bundle.outcome_selector
    )
    # Bound callables are only needed during selection and must not leak into the
    # Streamlit session/journal payload.
    for section in (halftime, second_half, corners, cards):
        section.pop("probability_fn", None)

    current_home_fifa = fifa.current_lookup(home)
    current_away_fifa = fifa.current_lookup(away)
    explanation_rows: list[dict[str, str]] = []
    headline_parts: list[str] = []

    if current_home_fifa and current_away_fifa:
        gap = float(current_home_fifa.points - current_away_fifa.points)
        direction, strength = _effect_label(gap, 25, 120, home, away)
        explanation_rows.append({
            "Фактор": "Рейтинг FIFA",
            "Направление": direction,
            "Сила влияния": strength,
            "Что увидела модель": (
                f"{home}: {current_home_fifa.rank or '—'} место и {current_home_fifa.points:.2f} очка; "
                f"{away}: {current_away_fifa.rank or '—'} место и {current_away_fifa.points:.2f} очка. "
                f"Разница {abs(gap):.1f} очка."
            ),
        })
        if abs(gap) >= 75:
            headline_parts.append(f"рейтинг FIFA в пользу {direction}")

    opponent_gap = float(row.get("opponent_elo_diff_5", 0.0))
    direction, strength = _effect_label(opponent_gap, 15, 70, home, away)
    explanation_rows.append({
        "Фактор": "Уровень последних соперников",
        "Направление": direction,
        "Сила влияния": strength,
        "Что увидела модель": (
            f"Разница среднего Elo соперников за последние пять матчей — {abs(opponent_gap):.1f}. "
            "Результаты против сильных соперников ценятся выше, чем такие же результаты против слабых."
        ),
    })

    form_gap = float(row.get("adjusted_form_diff_5", 0.0))
    direction, strength = _effect_label(form_gap, 0.08, 0.30, home, away)
    explanation_rows.append({
        "Фактор": "Форма последних пяти матчей",
        "Направление": direction,
        "Сила влияния": strength,
        "Что увидела модель": (
            f"Разница скорректированной формы — {abs(form_gap):.2f} условного очка за матч. "
            "Учтены место проведения, сила соперника и результат выше либо ниже ожидания."
        ),
    })
    if abs(form_gap) >= 0.18:
        headline_parts.append(f"скорректированная форма лучше у {direction}")

    goal_edge = float(dc_prediction.home_lambda - dc_prediction.away_lambda)
    direction, strength = _effect_label(goal_edge, 0.12, 0.65, home, away)
    explanation_rows.append({
        "Фактор": "Dixon–Coles и ожидаемые голы",
        "Направление": direction,
        "Сила влияния": strength,
        "Что увидела модель": (
            f"Ожидаемые голы: {home} {dc_prediction.home_lambda:.2f} — {dc_prediction.away_lambda:.2f} {away}. "
            "Модель учитывает атаку и оборону обеих команд и корректирует низкие счета."
        ),
    })

    context_notes: list[str] = []
    if context.stage == "group":
        context_notes.append(
            f"Перед матчем: {home} — {context.home_points} очков, разница {context.home_goal_difference:+d}; "
            f"{away} — {context.away_points} очков, разница {context.away_goal_difference:+d}."
        )
        if context.home_must_win:
            context_notes.append(f"Для {home} победа особенно важна.")
        if context.away_must_win:
            context_notes.append(f"Для {away} победа особенно важна.")
        if context.home_draw_enough:
            context_notes.append(f"{home} может устраивать ничья.")
        if context.away_draw_enough:
            context_notes.append(f"{away} может устраивать ничья.")
    else:
        context_notes.append("Плей-офф: отдельно рассчитаны 90 минут, дополнительное время и итоговый проход.")
    explanation_rows.append({
        "Фактор": "Турнирная ситуация",
        "Направление": "Учтена автоматически",
        "Сила влияния": "зависит от таблицы",
        "Что увидела модель": " ".join(context_notes),
    })

    explanation_rows.append({
        "Фактор": "Стартовые составы",
        "Направление": "Учтены" if context.lineups_known else "Без поправки",
        "Сила влияния": "умеренное" if context.lineups_known else "нет данных",
        "Что увидела модель": f"{home}: {home_squad.explanation} {away}: {away_squad.explanation}",
    })

    notes = data_source_notes or []
    explanation_rows.append({
        "Фактор": "Качество и актуальность данных",
        "Направление": "Проверено автоматически",
        "Сила влияния": "ограничивает уверенность",
        "Что увидела модель": " ".join(notes) if notes else "Использованы последние доступные результаты и рейтинги.",
    })

    leader_index = int(np.argmax(final_probs))
    leader = [away, "ничья", home][leader_index]
    summary = (
        "Ничья является наиболее вероятным отдельным исходом по итоговому ансамблю."
        if leader == "ничья" else
        f"Итоговый ансамбль отдаёт преимущество команде {leader}."
    )
    if headline_parts:
        summary += " Наиболее заметные факторы: " + "; ".join(headline_parts) + "."
    if non_obvious.get("found"):
        summary += f" Отдельно найден неочевидный исход: {non_obvious['best']['Исход']}."
    if stability_label == "низкая":
        summary += " Базовые модели заметно расходятся, поэтому прогноз следует считать нестабильным."

    progression = None
    if context.stage == "knockout":
        draw_90 = float(final_probs[1])
        strength_edge = float(row["elo_diff"] + row["fifa_points_diff"] * 0.35)
        home_conditional = 1.0 / (1.0 + np.exp(-strength_edge / 170.0))
        home_advance = float(final_probs[2] + draw_90 * home_conditional)
        progression = {
            "home_advance": home_advance,
            "away_advance": 1.0 - home_advance,
            "extra_time_probability": draw_90,
        }

    optional_status = bundle.optional_models.status_rows()
    outcomes = _market_summary_rows(home, away, final_probs, markets, halftime, second_half, corners, cards)

    return {
        "prediction_id": str(uuid4()),
        "home": home,
        "away": away,
        "match_date": pd.Timestamp(match_date).date().isoformat(),
        "neutral": neutral,
        "context": context.as_dict(),
        "prob_away_win": float(final_probs[0]),
        "prob_draw": float(final_probs[1]),
        "prob_home_win": float(final_probs[2]),
        "expected_goals_home": float(dc_prediction.home_lambda),
        "expected_goals_away": float(dc_prediction.away_lambda),
        "markets": markets,
        "halftime": halftime,
        "second_half": second_half,
        "corners": corners,
        "cards": cards,
        "optional_model_status": optional_status,
        "non_obvious_selection": non_obvious,
        "progression": progression,
        "components": components,
        "stability": stability,
        "outcomes": outcomes,
        "features": feature_dict,
        "explanations": [summary] + [row["Что увидела модель"] for row in explanation_rows],
        "explanation_rows": explanation_rows,
        "summary": summary,
        "home_squad": asdict(home_squad),
        "away_squad": asdict(away_squad),
        "model_version": bundle.version,
    }
