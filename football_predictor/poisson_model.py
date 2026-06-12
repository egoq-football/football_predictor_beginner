from __future__ import annotations

from math import exp, factorial

import numpy as np
import pandas as pd


def _blend(value: float, default: float, reliability: float) -> float:
    return value * reliability + default * (1.0 - reliability)


def estimate_expected_goals(features: pd.DataFrame) -> tuple[float, float]:
    row = features.iloc[0]
    neutral = bool(row["neutral"])
    base_goal_rate = 1.28

    home_reliability = min(float(row["home_matches"]) / 18.0, 1.0)
    away_reliability = min(float(row["away_matches"]) / 18.0, 1.0)

    home_gf = _blend(float(row["home_gf_10"]), base_goal_rate, home_reliability)
    home_ga = _blend(float(row["home_ga_10"]), base_goal_rate, home_reliability)
    away_gf = _blend(float(row["away_gf_10"]), base_goal_rate, away_reliability)
    away_ga = _blend(float(row["away_ga_10"]), base_goal_rate, away_reliability)

    home_adv = 0.14 if not neutral else 0.0
    away_penalty = -0.06 if not neutral else 0.0
    elo_adj = float(np.clip(row["elo_diff_raw"], -350, 350)) / 900.0

    home_xg = (home_gf * 0.52 + away_ga * 0.48) + home_adv + elo_adj
    away_xg = (away_gf * 0.52 + home_ga * 0.48) + away_penalty - elo_adj * 0.78

    home_xg = float(np.clip(home_xg, 0.18, 4.2))
    away_xg = float(np.clip(away_xg, 0.18, 4.2))
    return home_xg, away_xg


def poisson_probability(lmbda: float, goals: int) -> float:
    return exp(-lmbda) * (lmbda ** goals) / factorial(goals)


def score_matrix(home_xg: float, away_xg: float, max_goals: int = 8) -> np.ndarray:
    matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            matrix[h, a] = poisson_probability(home_xg, h) * poisson_probability(away_xg, a)
    matrix = matrix / matrix.sum()
    return matrix


def scoreline_probabilities(home_xg: float, away_xg: float, max_goals: int = 8) -> list[tuple[str, float]]:
    matrix = score_matrix(home_xg, away_xg, max_goals)
    scores: list[tuple[str, float]] = []
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            scores.append((f"{h}:{a}", float(matrix[h, a])))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def market_probabilities(home_xg: float, away_xg: float, max_goals: int = 8) -> dict[str, float]:
    matrix = score_matrix(home_xg, away_xg, max_goals)
    home_win = away_win = draw = 0.0
    over_15 = over_25 = over_35 = 0.0
    btts_yes = 0.0
    home_clean_sheet = 0.0
    away_clean_sheet = 0.0

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
            if total >= 2:
                over_15 += p
            if total >= 3:
                over_25 += p
            if total >= 4:
                over_35 += p
            if h > 0 and a > 0:
                btts_yes += p
            if a == 0:
                home_clean_sheet += p
            if h == 0:
                away_clean_sheet += p

    return {
        "poisson_home_win": home_win,
        "poisson_draw": draw,
        "poisson_away_win": away_win,
        "over_1_5": over_15,
        "under_1_5": 1.0 - over_15,
        "over_2_5": over_25,
        "under_2_5": 1.0 - over_25,
        "over_3_5": over_35,
        "under_3_5": 1.0 - over_35,
        "btts_yes": btts_yes,
        "btts_no": 1.0 - btts_yes,
        "home_clean_sheet": home_clean_sheet,
        "away_clean_sheet": away_clean_sheet,
        "double_chance_home_or_draw": home_win + draw,
        "double_chance_away_or_draw": away_win + draw,
        "double_chance_no_draw": home_win + away_win,
    }
