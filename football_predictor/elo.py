from __future__ import annotations

import math

BASE_ELO = 1500.0
HOME_ADVANTAGE = 60.0
K_FACTOR = 28.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for team A against team B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def match_result_points(goals_a: int, goals_b: int) -> float:
    if goals_a > goals_b:
        return 1.0
    if goals_a == goals_b:
        return 0.5
    return 0.0


def goal_difference_multiplier(goals_a: int, goals_b: int) -> float:
    diff = abs(goals_a - goals_b)
    if diff <= 1:
        return 1.0
    return math.log(diff + 1.0) * 1.15


def update_elo(
    home_elo: float,
    away_elo: float,
    home_score: int,
    away_score: int,
    neutral: bool,
    k_factor: float = K_FACTOR,
) -> tuple[float, float]:
    home_adv = 0.0 if neutral else HOME_ADVANTAGE
    expected_home = expected_score(home_elo + home_adv, away_elo)
    actual_home = match_result_points(home_score, away_score)
    mult = goal_difference_multiplier(home_score, away_score)
    change = k_factor * mult * (actual_home - expected_home)
    return home_elo + change, away_elo - change
