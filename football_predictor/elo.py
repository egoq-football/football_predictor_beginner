from __future__ import annotations

import math

BASE_ELO = 1500.0
HOME_ADVANTAGE = 55.0
BASE_K = 24.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def result_score(goals_a: int, goals_b: int) -> float:
    if goals_a > goals_b:
        return 1.0
    if goals_a == goals_b:
        return 0.5
    return 0.0


def goal_multiplier(goals_a: int, goals_b: int) -> float:
    diff = abs(int(goals_a) - int(goals_b))
    if diff <= 1:
        return 1.0
    return 1.0 + math.log(diff) * 0.55


def update_elo(
    home_rating: float,
    away_rating: float,
    home_score: int,
    away_score: int,
    neutral: bool,
    importance: float = 1.0,
) -> tuple[float, float]:
    advantage = 0.0 if neutral else HOME_ADVANTAGE
    expected_home = expected_score(home_rating + advantage, away_rating)
    actual_home = result_score(home_score, away_score)
    change = BASE_K * float(importance) * goal_multiplier(home_score, away_score) * (actual_home - expected_home)
    return home_rating + change, away_rating - change


def three_way_probabilities(home_rating: float, away_rating: float, neutral: bool) -> tuple[float, float, float]:
    advantage = 0.0 if neutral else HOME_ADVANTAGE
    edge = (home_rating + advantage - away_rating) / 400.0
    non_draw_home = 1.0 / (1.0 + 10.0 ** (-edge))
    draw = max(0.16, min(0.31, 0.27 - abs(edge) * 0.06))
    home = (1.0 - draw) * non_draw_home
    away = (1.0 - draw) * (1.0 - non_draw_home)
    return float(home), float(draw), float(away)
