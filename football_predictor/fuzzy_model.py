from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def fuzzy_probabilities(features: pd.DataFrame) -> dict[str, float]:
    """A simple explainable fuzzy-like model.

    It is not a copy of the academic papers, but it follows the same idea:
    linguistic rules are converted into numeric degrees.
    """
    row = features.iloc[0]
    elo_adv = float(row["elo_diff"])
    form_adv = float(row["form_points_diff_5"])
    gd_adv = float(row["goal_diff_form_diff_5"])
    h2h_adv = float(row["h2h_goal_diff"])

    home_strength = _sigmoid(elo_adv / 180.0)
    away_strength = _sigmoid(-elo_adv / 180.0)
    home_form = _sigmoid(form_adv / 0.8)
    away_form = _sigmoid(-form_adv / 0.8)
    home_goals = _sigmoid(gd_adv / 1.2)
    away_goals = _sigmoid(-gd_adv / 1.2)
    home_h2h = _sigmoid(h2h_adv / 1.5)
    away_h2h = _sigmoid(-h2h_adv / 1.5)

    # IF team is stronger AND in better form AND h2h is not bad THEN win is likely.
    home_rule = min(home_strength, home_form, max(home_goals, home_h2h))
    away_rule = min(away_strength, away_form, max(away_goals, away_h2h))

    # Draw is more likely when teams are close by Elo and recent form.
    closeness = np.exp(-abs(elo_adv) / 220.0) * np.exp(-abs(form_adv) / 1.2)
    draw_rule = float(0.65 * closeness)

    values = np.array([away_rule, draw_rule, home_rule], dtype=float)
    values = values + 0.05
    values = values / values.sum()
    return {
        "away_win": float(values[0]),
        "draw": float(values[1]),
        "home_win": float(values[2]),
    }


def explain_features(features: pd.DataFrame, home: str, away: str) -> list[str]:
    row = features.iloc[0]
    explanations: list[str] = []

    elo_diff = float(row["elo_diff"])
    form_diff = float(row["form_points_diff_5"])
    gd_diff = float(row["goal_diff_form_diff_5"])
    h2h = float(row["h2h_goal_diff"])

    if abs(elo_diff) < 40:
        explanations.append("По рейтингу силы команды близки друг к другу.")
    elif elo_diff > 0:
        explanations.append(f"По расчетному Elo преимущество у {home}.")
    else:
        explanations.append(f"По расчетному Elo преимущество у {away}.")

    if abs(form_diff) < 0.35:
        explanations.append("Форма последних 5 матчей примерно равная.")
    elif form_diff > 0:
        explanations.append(f"По очкам в последних 5 матчах лучше выглядит {home}.")
    else:
        explanations.append(f"По очкам в последних 5 матчах лучше выглядит {away}.")

    if gd_diff > 0.35:
        explanations.append(f"Разница голов в последних матчах лучше у {home}.")
    elif gd_diff < -0.35:
        explanations.append(f"Разница голов в последних матчах лучше у {away}.")
    else:
        explanations.append("По разнице голов в последних матчах явного преимущества нет.")

    if h2h > 0.3:
        explanations.append(f"Личные встречи в среднем немного в пользу {home}.")
    elif h2h < -0.3:
        explanations.append(f"Личные встречи в среднем немного в пользу {away}.")
    else:
        explanations.append("Личные встречи не дают сильного сигнала.")

    return explanations
