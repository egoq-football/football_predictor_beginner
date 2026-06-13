from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import MODEL_BUNDLE_PATH
from .context import MatchContext, infer_group_motivation
from .data_loader import load_match_lineups, load_player_pool
from .dixon_coles import score_markets
from .features import SequentialFeatureBuilder, current_feature_frame
from .fifa_rankings import FifaRankingHistory
from .lineups import SquadAssessment, assess_squad
from .model import WorldCupModelBundle, load_bundle
from .optional_models import poisson_total_probability


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-9, None)
    return probs / probs.sum()


def _most_likely_pair(label_a: str, p_a: float, label_b: str, p_b: float) -> tuple[str, float]:
    return (label_a, float(p_a)) if p_a >= p_b else (label_b, float(p_b))


def _outcomes_table(
    home: str,
    away: str,
    final_probs: np.ndarray,
    markets: dict,
    halftime: dict | None,
    second_half: dict | None,
    corners: dict | None,
    cards: dict | None,
) -> list[dict]:
    away_p, draw_p, home_p = map(float, final_probs)
    rows: list[dict] = []

    match_candidates = [(f"Победа {home}", home_p), ("Ничья", draw_p), (f"Победа {away}", away_p)]
    match_label, match_prob = max(match_candidates, key=lambda x: x[1])
    rows.append({"Категория": "Исход матча", "Наиболее вероятный исход": match_label, "Вероятность": match_prob, "Модель": "Калиброванный ансамбль"})

    double_candidates = [
        (f"{home} или ничья", home_p + draw_p),
        (f"{away} или ничья", away_p + draw_p),
        ("Без ничьей", home_p + away_p),
    ]
    label, prob = max(double_candidates, key=lambda x: x[1])
    rows.append({"Категория": "Двойной шанс", "Наиболее вероятный исход": label, "Вероятность": prob, "Модель": "Калиброванный ансамбль"})

    for threshold in [1.5, 2.5, 3.5]:
        over = float(markets[f"over_{str(threshold).replace('.', '_')}"])
        under = float(markets[f"under_{str(threshold).replace('.', '_')}"])
        label, prob = _most_likely_pair(f"Тотал больше {str(threshold).replace('.', ',')}", over, f"Тотал меньше {str(threshold).replace('.', ',')}", under)
        rows.append({"Категория": f"Тотал {str(threshold).replace('.', ',')}", "Наиболее вероятный исход": label, "Вероятность": prob, "Модель": "Dixon–Coles"})

    label, prob = _most_likely_pair("Обе забьют — да", float(markets["btts_yes"]), "Обе забьют — нет", float(markets["btts_no"]))
    rows.append({"Категория": "Обе забьют", "Наиболее вероятный исход": label, "Вероятность": prob, "Модель": "Dixon–Coles"})

    top_score, top_score_p = markets["top_scorelines"][0]
    rows.append({"Категория": "Точный счёт", "Наиболее вероятный исход": top_score, "Вероятность": float(top_score_p), "Модель": "Dixon–Coles"})

    if halftime and halftime.get("available"):
        candidates = [(f"Победа {home} в 1-м тайме", halftime["home_win"]), ("Ничья в 1-м тайме", halftime["draw"]), (f"Победа {away} в 1-м тайме", halftime["away_win"])]
        label, prob = max(candidates, key=lambda x: x[1])
        rows.append({"Категория": "Исход первого тайма", "Наиболее вероятный исход": label, "Вероятность": float(prob), "Модель": "Отдельная модель 1-го тайма"})
    if second_half and second_half.get("available"):
        candidates = [(f"Победа {home} во 2-м тайме", second_half["home_win"]), ("Ничья во 2-м тайме", second_half["draw"]), (f"Победа {away} во 2-м тайме", second_half["away_win"])]
        label, prob = max(candidates, key=lambda x: x[1])
        rows.append({"Категория": "Исход второго тайма", "Наиболее вероятный исход": label, "Вероятность": float(prob), "Модель": "Отдельная модель 2-го тайма"})
    if corners and corners.get("available"):
        label, prob = _most_likely_pair("Тотал угловых больше 8,5", corners["over_8_5"], "Тотал угловых меньше 8,5", corners["under_8_5"])
        rows.append({"Категория": "Угловые 8,5", "Наиболее вероятный исход": label, "Вероятность": prob, "Модель": "Отдельная модель угловых"})
    if cards and cards.get("available"):
        label, prob = _most_likely_pair("Тотал жёлтых карточек больше 3,5", cards["over_3_5"], "Тотал жёлтых карточек меньше 3,5", cards["under_3_5"])
        rows.append({"Категория": "Жёлтые карточки 3,5", "Наиболее вероятный исход": label, "Вероятность": prob, "Модель": "Отдельная модель карточек"})
    return rows


def _pair_poisson_markets(home_mean: float, away_mean: float, max_goals: int = 8) -> dict[str, float]:
    from math import factorial
    matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i, j] = np.exp(-home_mean) * home_mean**i / factorial(i) * np.exp(-away_mean) * away_mean**j / factorial(j)
    matrix /= matrix.sum()
    return {
        "away_win": float(np.triu(matrix, 1).sum()),
        "draw": float(np.trace(matrix)),
        "home_win": float(np.tril(matrix, -1).sum()),
    }


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
    manual_home_strength: float | None = None,
    manual_away_strength: float | None = None,
) -> dict:
    context = infer_group_motivation(context)
    player_pool = player_pool if player_pool is not None else load_player_pool()
    home_squad = assess_squad(player_pool, home, selected_home_players)
    away_squad = assess_squad(player_pool, away, selected_away_players)
    lineup_home = home_squad.relative_strength if home_squad.available else manual_home_strength
    lineup_away = away_squad.relative_strength if away_squad.available else manual_away_strength
    if not home_squad.available and manual_home_strength is not None:
        home_squad.relative_strength = float(manual_home_strength)
        home_squad.explanation = f"Ручная оценка силы состава: {manual_home_strength * 100:.1f}% от оптимального."
    if not away_squad.available and manual_away_strength is not None:
        away_squad.relative_strength = float(manual_away_strength)
        away_squad.explanation = f"Ручная оценка силы состава: {manual_away_strength * 100:.1f}% от оптимального."

    features = current_feature_frame(
        builder, home, away, neutral, match_date, context,
        lineup_strength_home=lineup_home,
        lineup_strength_away=lineup_away,
    )
    final_probs, components = bundle.predict(features, home, away, neutral)
    dc_prediction = bundle.dixon_coles.predict(home, away, neutral)
    markets = score_markets(dc_prediction)

    ht_pred = bundle.optional_models.halftime.predict(features)
    second_pred = bundle.optional_models.second_half.predict(features)
    corners_pred = bundle.optional_models.corners.predict(features)
    cards_pred = bundle.optional_models.cards.predict(features)

    halftime = {"available": False, "reason": ht_pred.reason}
    if ht_pred.available and ht_pred.home_mean is not None and ht_pred.away_mean is not None:
        halftime = {"available": True, **_pair_poisson_markets(ht_pred.home_mean, ht_pred.away_mean), "home_mean": ht_pred.home_mean, "away_mean": ht_pred.away_mean}

    second_half = {"available": False, "reason": second_pred.reason}
    if second_pred.available and second_pred.home_mean is not None and second_pred.away_mean is not None:
        second_half = {"available": True, **_pair_poisson_markets(second_pred.home_mean, second_pred.away_mean), "home_mean": second_pred.home_mean, "away_mean": second_pred.away_mean}

    corners = {"available": False, "reason": corners_pred.reason}
    if corners_pred.available and corners_pred.home_mean is not None and corners_pred.away_mean is not None:
        mean_total = corners_pred.home_mean + corners_pred.away_mean
        corners = {
            "available": True,
            "home_mean": corners_pred.home_mean,
            "away_mean": corners_pred.away_mean,
            "total_mean": mean_total,
            "over_8_5": poisson_total_probability(mean_total, 8.5, True),
            "under_8_5": poisson_total_probability(mean_total, 8.5, False),
        }

    cards = {"available": False, "reason": cards_pred.reason}
    if cards_pred.available and cards_pred.home_mean is not None and cards_pred.away_mean is not None:
        mean_total = cards_pred.home_mean + cards_pred.away_mean
        cards = {
            "available": True,
            "home_mean": cards_pred.home_mean,
            "away_mean": cards_pred.away_mean,
            "total_mean": mean_total,
            "over_3_5": poisson_total_probability(mean_total, 3.5, True),
            "under_3_5": poisson_total_probability(mean_total, 3.5, False),
        }

    current_home_fifa = fifa.current_lookup(home)
    current_away_fifa = fifa.current_lookup(away)
    explanations = []
    if current_home_fifa and current_away_fifa:
        gap = current_home_fifa.points - current_away_fifa.points
        leader = home if gap > 0 else away
        explanations.append(
            f"FIFA: {home} — {current_home_fifa.rank or '—'} место ({current_home_fifa.points:.2f}), "
            f"{away} — {current_away_fifa.rank or '—'} место ({current_away_fifa.points:.2f}); "
            f"разница {abs(gap):.2f} очка в пользу {leader}."
        )
    row = features.iloc[0]
    opp_leader = home if row["opponent_elo_diff_5"] > 0 else away
    explanations.append(
        f"Сила соперников в последних пяти матчах: преимущество у {opp_leader}; "
        f"разница среднего Elo соперников {abs(row['opponent_elo_diff_5']):.1f}."
    )
    form_leader = home if row["adjusted_form_diff_5"] > 0 else away
    explanations.append(
        f"Форма с поправкой на силу соперников лучше у {form_leader} "
        f"({abs(row['adjusted_form_diff_5']):.2f} условного очка за матч разницы)."
    )
    if context.home_must_win or context.away_must_win:
        must = home if context.home_must_win else away
        explanations.append(f"Турнирная ситуация: для {must} победа отмечена как необходимая.")
    if context.home_draw_enough or context.away_draw_enough:
        enough = home if context.home_draw_enough else away
        explanations.append(f"Турнирная ситуация: {enough} может устраивать ничья.")
    if home_squad.available or away_squad.available:
        explanations.append(f"Состав {home}: {home_squad.explanation}")
        explanations.append(f"Состав {away}: {away_squad.explanation}")

    progression = None
    if context.stage == "knockout":
        draw_90 = float(final_probs[1])
        strength_edge = float(row["elo_diff"] + row["fifa_points_diff"] * 0.35)
        home_conditional = 1.0 / (1.0 + np.exp(-strength_edge / 170.0))
        home_advance = float(final_probs[2] + draw_90 * home_conditional)
        away_advance = 1.0 - home_advance
        progression = {
            "home_advance": home_advance,
            "away_advance": away_advance,
            "extra_time_probability": draw_90,
        }

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
        "expected_goals_home": dc_prediction.home_lambda,
        "expected_goals_away": dc_prediction.away_lambda,
        "markets": markets,
        "halftime": halftime,
        "second_half": second_half,
        "corners": corners,
        "cards": cards,
        "progression": progression,
        "components": {name: {"away": float(p[0]), "draw": float(p[1]), "home": float(p[2])} for name, p in components.items()},
        "outcomes": _outcomes_table(home, away, final_probs, markets, halftime, second_half, corners, cards),
        "features": row.to_dict(),
        "explanations": explanations,
        "home_squad": asdict(home_squad),
        "away_squad": asdict(away_squad),
        "model_version": bundle.version,
    }
