from __future__ import annotations

from math import exp, factorial

import numpy as np
import pandas as pd


def _blend(value: float, default: float, reliability: float) -> float:
    return value * reliability + default * (1.0 - reliability)


def estimate_expected_goals(
    features: pd.DataFrame,
    fifa_points_diff: float = 0.0,
) -> tuple[float, float]:
    """Estimate scoreline lambdas from both teams' scoring/conceding averages.

    The most likely scores are intentionally separated from FIFA and Elo. For
    each side the goal expectation is the arithmetic mean of: (1) that team's
    recent goals scored and (2) the opponent's recent goals conceded. The last
    five matches carry 75% of each average and the last ten carry 25%.
    """
    row = features.iloc[0]
    neutral = bool(row["neutral"])
    base_goal_rate = 1.27

    home_reliability = min(float(row["home_matches"]) / 5.0, 1.0)
    away_reliability = min(float(row["away_matches"]) / 5.0, 1.0)

    home_gf_recent = float(row["home_gf_5"]) * 0.75 + float(row["home_gf_10"]) * 0.25
    home_ga_recent = float(row["home_ga_5"]) * 0.75 + float(row["home_ga_10"]) * 0.25
    away_gf_recent = float(row["away_gf_5"]) * 0.75 + float(row["away_gf_10"]) * 0.25
    away_ga_recent = float(row["away_ga_5"]) * 0.75 + float(row["away_ga_10"]) * 0.25

    home_gf = _blend(home_gf_recent, base_goal_rate, home_reliability)
    home_ga = _blend(home_ga_recent, base_goal_rate, home_reliability)
    away_gf = _blend(away_gf_recent, base_goal_rate, away_reliability)
    away_ga = _blend(away_ga_recent, base_goal_rate, away_reliability)

    home_xg = (home_gf + away_ga) / 2.0
    away_xg = (away_gf + home_ga) / 2.0

    # A small venue correction is applied only when the match is not neutral.
    if not neutral:
        home_xg *= 1.06
        away_xg *= 0.96

    home_xg = float(np.clip(home_xg, 0.16, 4.4))
    away_xg = float(np.clip(away_xg, 0.16, 4.4))
    return home_xg, away_xg




def estimate_scoreline_goal_means(features: pd.DataFrame) -> tuple[float, float]:
    """Goal means for the *scoreline* table only.

    They are based on both teams' average scored and conceded goals with a
    stronger weight on the last five matches.
    """
    row = features.iloc[0]
    neutral = bool(row["neutral"])

    home_scored = float(row["home_gf_5"]) * 0.65 + float(row["home_gf_10"]) * 0.35
    home_conceded = float(row["home_ga_5"]) * 0.65 + float(row["home_ga_10"]) * 0.35
    away_scored = float(row["away_gf_5"]) * 0.65 + float(row["away_gf_10"]) * 0.35
    away_conceded = float(row["away_ga_5"]) * 0.65 + float(row["away_ga_10"]) * 0.35

    home_lambda = (home_scored + away_conceded) / 2.0
    away_lambda = (away_scored + home_conceded) / 2.0
    if not neutral:
        home_lambda *= 1.06
        away_lambda *= 0.96
    return float(np.clip(home_lambda, 0.16, 4.4)), float(np.clip(away_lambda, 0.16, 4.4))

def poisson_probability(lmbda: float, goals: int) -> float:
    return exp(-lmbda) * (lmbda ** goals) / factorial(goals)


def score_matrix(home_xg: float, away_xg: float, max_goals: int = 8) -> np.ndarray:
    matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            matrix[h, a] = poisson_probability(home_xg, h) * poisson_probability(away_xg, a)
    return matrix / matrix.sum()


def scoreline_probabilities(home_xg: float, away_xg: float, max_goals: int = 8) -> list[tuple[str, float]]:
    matrix = score_matrix(home_xg, away_xg, max_goals)
    scores: list[tuple[str, float]] = []
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            scores.append((f"{h}:{a}", float(matrix[h, a])))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _matrix_markets(matrix: np.ndarray) -> dict[str, float]:
    home_win = away_win = draw = 0.0
    totals = {0.5: 0.0, 1.5: 0.0, 2.5: 0.0, 3.5: 0.0, 4.5: 0.0}
    btts_yes = 0.0
    home_clean_sheet = away_clean_sheet = 0.0
    home_scores = away_scores = 0.0
    home_over_15 = away_over_15 = 0.0
    home_win_to_nil = away_win_to_nil = 0.0

    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            p = float(matrix[h, a])
            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p
            total = h + a
            for line in totals:
                if total > line:
                    totals[line] += p
            if h > 0 and a > 0:
                btts_yes += p
            if a == 0:
                home_clean_sheet += p
            if h == 0:
                away_clean_sheet += p
            if h > 0:
                home_scores += p
            if a > 0:
                away_scores += p
            if h >= 2:
                home_over_15 += p
            if a >= 2:
                away_over_15 += p
            if h > a and a == 0:
                home_win_to_nil += p
            if a > h and h == 0:
                away_win_to_nil += p

    out = {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "btts_yes": btts_yes,
        "btts_no": 1.0 - btts_yes,
        "home_clean_sheet": home_clean_sheet,
        "away_clean_sheet": away_clean_sheet,
        "home_scores": home_scores,
        "home_not_score": 1.0 - home_scores,
        "away_scores": away_scores,
        "away_not_score": 1.0 - away_scores,
        "home_team_over_1_5": home_over_15,
        "home_team_under_1_5": 1.0 - home_over_15,
        "away_team_over_1_5": away_over_15,
        "away_team_under_1_5": 1.0 - away_over_15,
        "home_win_to_nil": home_win_to_nil,
        "away_win_to_nil": away_win_to_nil,
    }
    for line, value in totals.items():
        suffix = str(line).replace(".", "_")
        out[f"over_{suffix}"] = value
        out[f"under_{suffix}"] = 1.0 - value
    return out


def market_probabilities(home_xg: float, away_xg: float, max_goals: int = 8) -> dict[str, float]:
    matrix = score_matrix(home_xg, away_xg, max_goals)
    out = _matrix_markets(matrix)
    out.update({
        "poisson_home_win": out["home_win"],
        "poisson_draw": out["draw"],
        "poisson_away_win": out["away_win"],
        "double_chance_home_or_draw": out["home_win"] + out["draw"],
        "double_chance_away_or_draw": out["away_win"] + out["draw"],
        "double_chance_no_draw": out["home_win"] + out["away_win"],
    })

    first = _matrix_markets(score_matrix(home_xg * 0.44, away_xg * 0.44, max_goals=5))
    second = _matrix_markets(score_matrix(home_xg * 0.56, away_xg * 0.56, max_goals=5))
    for key, value in first.items():
        out[f"first_half_{key}"] = value
    for key, value in second.items():
        out[f"second_half_{key}"] = value
    return out


def _choose(category: str, options: list[tuple[str, float]], model: str) -> dict[str, object]:
    outcome, probability = max(options, key=lambda item: item[1])
    return {
        "Категория": category,
        "Наиболее вероятный исход": outcome,
        "Вероятность": float(probability),
        "Модель": model,
    }


def most_likely_outcomes(
    home: str,
    away: str,
    final_probs: np.ndarray,
    markets: dict[str, float],
    top_scoreline: tuple[str, float],
) -> list[dict[str, object]]:
    away_win, draw, home_win = [float(x) for x in final_probs]
    rows = [
        _choose("Исход матча", [(f"Победа {home}", home_win), ("Ничья", draw), (f"Победа {away}", away_win)], "ансамбль"),
        _choose("Двойной шанс", [
            (f"{home} или ничья", home_win + draw),
            (f"{away} или ничья", away_win + draw),
            ("Без ничьей", home_win + away_win),
        ], "ансамбль"),
        _choose("Тотал 1,5", [("Больше 1,5", markets["over_1_5"]), ("Меньше 1,5", markets["under_1_5"])], "Пуассон"),
        _choose("Тотал 2,5", [("Больше 2,5", markets["over_2_5"]), ("Меньше 2,5", markets["under_2_5"])], "Пуассон"),
        _choose("Тотал 3,5", [("Больше 3,5", markets["over_3_5"]), ("Меньше 3,5", markets["under_3_5"])], "Пуассон"),
        _choose("Обе забьют", [("Да", markets["btts_yes"]), ("Нет", markets["btts_no"])], "Пуассон"),
        _choose(f"Индивидуальный тотал {home} 1,5", [
            (f"{home} больше 1,5", markets["home_team_over_1_5"]),
            (f"{home} меньше 1,5", markets["home_team_under_1_5"]),
        ], "Пуассон"),
        _choose(f"Индивидуальный тотал {away} 1,5", [
            (f"{away} больше 1,5", markets["away_team_over_1_5"]),
            (f"{away} меньше 1,5", markets["away_team_under_1_5"]),
        ], "Пуассон"),
        _choose("Победа всухую", [
            (f"{home} победит всухую", markets["home_win_to_nil"]),
            (f"{away} победит всухую", markets["away_win_to_nil"]),
            ("Победы всухую не будет", 1.0 - markets["home_win_to_nil"] - markets["away_win_to_nil"]),
        ], "Пуассон"),
        _choose(f"Гол {home}", [(f"{home} забьёт", markets["home_scores"]), (f"{home} не забьёт", markets["home_not_score"])], "Пуассон"),
        _choose(f"Гол {away}", [(f"{away} забьёт", markets["away_scores"]), (f"{away} не забьёт", markets["away_not_score"])], "Пуассон"),
        _choose("Исход первого тайма", [
            (f"Победа {home} в 1-м тайме", markets["first_half_home_win"]),
            ("Ничья в 1-м тайме", markets["first_half_draw"]),
            (f"Победа {away} в 1-м тайме", markets["first_half_away_win"]),
        ], "Пуассон по таймам"),
        _choose("Тотал первого тайма 0,5", [
            ("Больше 0,5 в 1-м тайме", markets["first_half_over_0_5"]),
            ("Меньше 0,5 в 1-м тайме", markets["first_half_under_0_5"]),
        ], "Пуассон по таймам"),
        _choose("Тотал первого тайма 1,5", [
            ("Больше 1,5 в 1-м тайме", markets["first_half_over_1_5"]),
            ("Меньше 1,5 в 1-м тайме", markets["first_half_under_1_5"]),
        ], "Пуассон по таймам"),
        _choose("Исход второго тайма", [
            (f"Победа {home} во 2-м тайме", markets["second_half_home_win"]),
            ("Ничья во 2-м тайме", markets["second_half_draw"]),
            (f"Победа {away} во 2-м тайме", markets["second_half_away_win"]),
        ], "Пуассон по таймам"),
        _choose("Тотал второго тайма 0,5", [
            ("Больше 0,5 во 2-м тайме", markets["second_half_over_0_5"]),
            ("Меньше 0,5 во 2-м тайме", markets["second_half_under_0_5"]),
        ], "Пуассон по таймам"),
        _choose("Тотал второго тайма 1,5", [
            ("Больше 1,5 во 2-м тайме", markets["second_half_over_1_5"]),
            ("Меньше 1,5 во 2-м тайме", markets["second_half_under_1_5"]),
        ], "Пуассон по таймам"),
        {
            "Категория": "Точный счёт",
            "Наиболее вероятный исход": top_scoreline[0],
            "Вероятность": float(top_scoreline[1]),
            "Модель": "Средние забитые/пропущенные + Пуассон",
        },
    ]
    return sorted(rows, key=lambda row: float(row["Вероятность"]), reverse=True)
