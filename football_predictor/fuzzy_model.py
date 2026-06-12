from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def fuzzy_probabilities(features: pd.DataFrame) -> dict[str, float]:
    """Explainable fuzzy-like model based on linguistic IF-THEN logic."""
    row = features.iloc[0]
    elo_adv = float(row["elo_diff"])
    form_adv = float(row["form_points_diff_5"])
    gd_adv = float(row["goal_diff_form_diff_5"])
    h2h_adv = float(row["h2h_goal_diff"])
    attack_adv = float(row["attack_diff_10"])
    defense_adv = float(row["defense_diff_10"])

    home_strength = _sigmoid(elo_adv / 180.0)
    away_strength = _sigmoid(-elo_adv / 180.0)
    home_form = _sigmoid(form_adv / 0.8)
    away_form = _sigmoid(-form_adv / 0.8)
    home_goals = _sigmoid((gd_adv + attack_adv + defense_adv) / 1.8)
    away_goals = _sigmoid(-(gd_adv + attack_adv + defense_adv) / 1.8)
    home_h2h = _sigmoid(h2h_adv / 1.5)
    away_h2h = _sigmoid(-h2h_adv / 1.5)

    home_rule = max(
        min(home_strength, home_form, max(home_goals, home_h2h)),
        min(home_strength, home_goals),
    )
    away_rule = max(
        min(away_strength, away_form, max(away_goals, away_h2h)),
        min(away_strength, away_goals),
    )

    closeness = np.exp(-abs(elo_adv) / 220.0) * np.exp(-abs(form_adv) / 1.2)
    draw_tendency = float(row.get("draw_rate_sum_5", 0.5))
    draw_rule = float((0.48 + 0.22 * draw_tendency) * closeness)

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
    attack = float(row["attack_diff_10"])
    defense = float(row["defense_diff_10"])

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

    if attack > 0.25:
        explanations.append(f"По средней результативности за 10 матчей атака сильнее у {home}.")
    elif attack < -0.25:
        explanations.append(f"По средней результативности за 10 матчей атака сильнее у {away}.")

    if defense > 0.25:
        explanations.append(f"По пропущенным голам за 10 матчей оборона надежнее у {home}.")
    elif defense < -0.25:
        explanations.append(f"По пропущенным голам за 10 матчей оборона надежнее у {away}.")

    if h2h > 0.3:
        explanations.append(f"Очные встречи в среднем в пользу {home}.")
    elif h2h < -0.3:
        explanations.append(f"Очные встречи в среднем в пользу {away}.")
    else:
        explanations.append("Очные встречи не дают сильного преимущества одной стороне.")

    return explanations
