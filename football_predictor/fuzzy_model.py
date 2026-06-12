from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def fuzzy_probabilities(features: pd.DataFrame) -> dict[str, float]:
    """Explainable fuzzy-like model based on linguistic IF-THEN logic."""
    row = features.iloc[0]
    elo_adv = float(row["elo_diff"])
    raw_form_adv = float(row["form_points_diff_5"])
    strength_form_adv = float(row["strength_form_points_diff_5"])
    performance_adv = float(row["performance_diff_5"])
    raw_gd_adv = float(row["goal_diff_form_diff_5"])
    strength_gd_adv = float(row["strength_goal_diff_diff_5"])
    h2h_adv = float(row["h2h_goal_diff"])
    attack_adv = float(row["attack_diff_10"])
    defense_adv = float(row["defense_diff_10"])

    # The recent-form rule explicitly rewards results achieved against stronger
    # opponents and performance above pre-match Elo expectations.
    form_adv = raw_form_adv * 0.45 + strength_form_adv * 0.40 + performance_adv * 1.15
    gd_adv = raw_gd_adv * 0.55 + strength_gd_adv * 0.45

    home_strength = _sigmoid(elo_adv / 180.0)
    away_strength = _sigmoid(-elo_adv / 180.0)
    home_form = _sigmoid(form_adv / 0.85)
    away_form = _sigmoid(-form_adv / 0.85)
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
    home_opp = float(row["home_opponent_elo_5"])
    away_opp = float(row["away_opponent_elo_5"])
    strength_form = float(row["strength_form_points_diff_5"])
    performance = float(row["performance_diff_5"])

    if abs(elo_diff) < 40:
        explanations.append("По внутреннему рейтингу Elo команды близки друг к другу.")
    elif elo_diff > 0:
        explanations.append(f"По внутреннему рейтингу Elo преимущество у {home}.")
    else:
        explanations.append(f"По внутреннему рейтингу Elo преимущество у {away}.")

    if abs(form_diff) < 0.35:
        explanations.append("По обычным очкам форма последних 5 матчей примерно равная.")
    elif form_diff > 0:
        explanations.append(f"По очкам в последних 5 матчах лучше выглядит {home}.")
    else:
        explanations.append(f"По очкам в последних 5 матчах лучше выглядит {away}.")

    opp_gap = home_opp - away_opp
    if abs(opp_gap) < 35:
        explanations.append(
            f"Средняя сила соперников в последних 5 матчах была близкой: {home} — Elo {home_opp:.0f}, "
            f"{away} — Elo {away_opp:.0f}."
        )
    elif opp_gap > 0:
        explanations.append(
            f"Последние 5 матчей {home} были против более сильных соперников в среднем "
            f"(Elo {home_opp:.0f} против {away_opp:.0f}), поэтому эта форма получает дополнительный вес."
        )
    else:
        explanations.append(
            f"Последние 5 матчей {away} были против более сильных соперников в среднем "
            f"(Elo {away_opp:.0f} против {home_opp:.0f}), поэтому эта форма получает дополнительный вес."
        )

    if abs(strength_form) >= 0.25 or abs(performance) >= 0.08:
        if strength_form + performance * 1.5 > 0:
            explanations.append(f"После поправки на силу соперников недавние результаты лучше у {home}.")
        else:
            explanations.append(f"После поправки на силу соперников недавние результаты лучше у {away}.")

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
        explanations.append(f"По пропущенным голам за 10 матчей оборона надёжнее у {home}.")
    elif defense < -0.25:
        explanations.append(f"По пропущенным голам за 10 матчей оборона надёжнее у {away}.")

    if h2h > 0.3:
        explanations.append(f"Очные встречи в среднем в пользу {home}.")
    elif h2h < -0.3:
        explanations.append(f"Очные встречи в среднем в пользу {away}.")
    else:
        explanations.append("Очные встречи не дают сильного преимущества одной стороне.")

    return explanations
