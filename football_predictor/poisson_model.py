from __future__ import annotations

from math import exp, factorial

import numpy as np
import pandas as pd


def estimate_expected_goals(features: pd.DataFrame) -> tuple[float, float]:
    row = features.iloc[0]
    home_gf = float(row["home_gf_5"])
    home_ga = float(row["home_ga_5"])
    away_gf = float(row["away_gf_5"])
    away_ga = float(row["away_ga_5"])
    neutral = bool(row["neutral"])

    home_adv = 0.12 if not neutral else 0.0
    away_penalty = -0.06 if not neutral else 0.0

    home_xg = (home_gf * 0.55 + away_ga * 0.45) + home_adv
    away_xg = (away_gf * 0.55 + home_ga * 0.45) + away_penalty

    home_xg = float(np.clip(home_xg, 0.25, 3.2))
    away_xg = float(np.clip(away_xg, 0.25, 3.2))
    return home_xg, away_xg


def poisson_probability(lmbda: float, goals: int) -> float:
    return exp(-lmbda) * (lmbda ** goals) / factorial(goals)


def scoreline_probabilities(home_xg: float, away_xg: float, max_goals: int = 6) -> list[tuple[str, float]]:
    scores: list[tuple[str, float]] = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = poisson_probability(home_xg, h) * poisson_probability(away_xg, a)
            scores.append((f"{h}:{a}", float(prob)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
