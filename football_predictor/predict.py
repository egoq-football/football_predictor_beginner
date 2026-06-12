from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import list_teams, load_results
from .features import build_current_states, build_training_table, make_match_features
from .fifa_ranking import FifaRankingInfo, fifa_probabilities
from .fuzzy_model import explain_features, fuzzy_probabilities
from .model import is_model_compatible, load_model, save_model, train_with_chronological_test
from .poisson_model import (
    estimate_expected_goals,
    market_probabilities,
    most_likely_outcomes,
    scoreline_probabilities,
)
from .stats import h2h_matches, team_recent_matches, team_summary


LABELS = {0: "Победа второй команды", 1: "Ничья", 2: "Победа первой команды"}


def prepare_model(
    results_path: str = "data/results.csv",
    model_path: str = "models/football_predictor.joblib",
    force_retrain: bool = False,
):
    df = load_results(results_path)
    model_file = Path(model_path)

    model = None
    if model_file.exists() and not force_retrain:
        try:
            loaded = load_model(model_file)
            if is_model_compatible(loaded):
                model = loaded
        except Exception:
            model = None

    if model is None:
        table = build_training_table(df, min_year=2010)
        if len(table) < 80:
            raise ValueError(
                "Для обучения найдено слишком мало завершённых матчей начиная с 2010 года. "
                "Обнови данные в боковой панели."
            )
        model = train_with_chronological_test(table)
        save_model(model, model_file)

    states, h2h = build_current_states(df)
    teams = list_teams(df)
    return df, model, states, h2h, teams


def recent_form_probabilities(features: pd.DataFrame) -> dict[str, float]:
    """Three-way probabilities with opponent-adjusted last-five form."""
    row = features.iloc[0]
    edge = (
        float(row["form_points_diff_5"]) * 0.58
        + float(row["strength_form_points_diff_5"]) * 0.62
        + float(row["performance_diff_5"]) * 1.35
        + float(row["goal_diff_form_diff_5"]) * 0.36
        + float(row["strength_goal_diff_diff_5"]) * 0.34
        + float(row["win_rate_diff_5"]) * 0.34
        + (float(row["home_gf_5"]) - float(row["away_gf_5"])) * 0.12
        + (float(row["away_ga_5"]) - float(row["home_ga_5"])) * 0.12
    )
    # The 10-match window is deliberately only a small stabilizer.
    edge += float(row["form_points_diff_10"]) * 0.10
    edge += float(row["goal_diff_form_diff_10"]) * 0.06

    non_draw_home = 1.0 / (1.0 + np.exp(-edge / 1.30))
    draw_signal = float(row["draw_rate_sum_5"])
    draw = 0.21 + min(draw_signal * 0.10, 0.10) - min(abs(edge) * 0.025, 0.07)
    draw = float(np.clip(draw, 0.16, 0.34))
    home_win = float((1.0 - draw) * non_draw_home)
    away_win = float((1.0 - draw) * (1.0 - non_draw_home))
    return {"away_win": away_win, "draw": draw, "home_win": home_win, "edge": float(edge)}


def _fifa_gap_description(home: str, away: str, home_fifa, away_fifa) -> str:
    points_gap = abs(float(home_fifa.points) - float(away_fifa.points))
    rank_gap = abs(int(home_fifa.rank) - int(away_fifa.rank))
    leader = home if home_fifa.points > away_fifa.points else away

    if points_gap < 30 and rank_gap <= 5:
        level = "небольшое"
    elif points_gap < 80 and rank_gap <= 15:
        level = "заметное"
    elif points_gap < 170 and rank_gap <= 35:
        level = "значительное"
    else:
        level = "очень большое"

    return (
        f"Разница составляет {points_gap:.2f} очка и {rank_gap} позиций — это {level} "
        f"преимущество рейтинга в пользу {leader}."
    )


def predict_match(
    home: str,
    away: str,
    neutral: bool,
    model,
    states,
    h2h,
    fifa_lookup: dict[str, FifaRankingInfo] | None = None,
    df: pd.DataFrame | None = None,
) -> dict:
    if home == away:
        raise ValueError("Нужно выбрать две разные команды.")
    if home not in states:
        raise ValueError(f"У команды нет завершённых матчей с 2010 года: {home}")
    if away not in states:
        raise ValueError(f"У команды нет завершённых матчей с 2010 года: {away}")

    fifa_lookup = fifa_lookup or {}
    features = make_match_features(home, away, neutral, states, h2h)

    ml_probs = model.predict_proba(features)[0]
    fuzzy = fuzzy_probabilities(features)
    fuzzy_probs = np.array([fuzzy["away_win"], fuzzy["draw"], fuzzy["home_win"]], dtype=float)

    recent = recent_form_probabilities(features)
    recent_probs = np.array([recent["away_win"], recent["draw"], recent["home_win"]], dtype=float)

    fifa = fifa_probabilities(home, away, neutral, fifa_lookup)
    fifa_probs = np.array([fifa["away_win"], fifa["draw"], fifa["home_win"]], dtype=float)

    # Goal and score markets are based directly on both teams' recent scoring
    # and conceding averages. Elo/FIFA do not alter the scoreline lambdas.
    home_xg, away_xg = estimate_expected_goals(features, fifa_points_diff=fifa["points_diff"])
    markets = market_probabilities(home_xg, away_xg, max_goals=8)
    poisson_probs = np.array(
        [markets["poisson_away_win"], markets["poisson_draw"], markets["poisson_home_win"]],
        dtype=float,
    )

    if fifa["available"]:
        weights = {"recent5": 0.30, "fifa": 0.20, "ml": 0.25, "poisson": 0.20, "fuzzy": 0.05}
    else:
        weights = {"recent5": 0.50, "fifa": 0.00, "ml": 0.25, "poisson": 0.20, "fuzzy": 0.05}

    final_probs = (
        recent_probs * weights["recent5"]
        + fifa_probs * weights["fifa"]
        + ml_probs * weights["ml"]
        + poisson_probs * weights["poisson"]
        + fuzzy_probs * weights["fuzzy"]
    )
    final_probs = final_probs / final_probs.sum()

    scores = scoreline_probabilities(home_xg, away_xg, max_goals=8)[:12]
    likely = most_likely_outcomes(home, away, final_probs, markets, scores[0])

    home_fifa = fifa_lookup.get(home)
    away_fifa = fifa_lookup.get(away)
    explanations = explain_features(features, home, away)
    if home_fifa and away_fifa:
        explanations.insert(
            0,
            f"Рейтинг FIFA: {home} — {home_fifa.rank}-е место ({home_fifa.points:.2f}), "
            f"{away} — {away_fifa.rank}-е место ({away_fifa.points:.2f}). "
            + _fifa_gap_description(home, away, home_fifa, away_fifa),
        )
    else:
        explanations.insert(0, "Для одной из команд рейтинг FIFA не загрузился; его вес перераспределён на последние матчи.")

    result = {
        "home": home,
        "away": away,
        "neutral": neutral,
        "prob_away_win": float(final_probs[0]),
        "prob_draw": float(final_probs[1]),
        "prob_home_win": float(final_probs[2]),
        "ml_probs": {"away_win": float(ml_probs[0]), "draw": float(ml_probs[1]), "home_win": float(ml_probs[2])},
        "recent_probs": recent,
        "fifa_probs": fifa,
        "fuzzy_probs": fuzzy,
        "poisson_probs": {"away_win": float(poisson_probs[0]), "draw": float(poisson_probs[1]), "home_win": float(poisson_probs[2])},
        "model_weights": weights,
        "expected_goals_home": home_xg,
        "expected_goals_away": away_xg,
        "scoreline_home_mean": home_xg,
        "scoreline_away_mean": away_xg,
        "markets": markets,
        "most_likely_outcomes": likely,
        "top_scorelines": scores,
        "features": features.iloc[0].to_dict(),
        "explanations": explanations,
        "team_summary": [
            team_summary(home, states[home], home_fifa),
            team_summary(away, states[away], away_fifa),
        ],
    }

    if df is not None:
        result["recent_home"] = team_recent_matches(df, home, 10)
        result["recent_away"] = team_recent_matches(df, away, 10)
        result["h2h_table"] = h2h_matches(df, home, away, 10)

    return result
