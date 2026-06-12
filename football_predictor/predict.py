from __future__ import annotations

from pathlib import Path

import numpy as np

from .data_loader import list_teams, load_results
from .features import build_current_states, build_training_table, make_match_features
from .fuzzy_model import explain_features, fuzzy_probabilities
from .model import load_model, save_model, train_with_chronological_test
from .poisson_model import estimate_expected_goals, scoreline_probabilities


LABELS = {0: "Победа второй команды", 1: "Ничья", 2: "Победа первой команды"}


def prepare_model(results_path: str = "data/results.csv", model_path: str = "models/football_predictor.joblib", force_retrain: bool = False):
    df = load_results(results_path)
    model_file = Path(model_path)

    if model_file.exists() and not force_retrain:
        model = load_model(model_file)
    else:
        table = build_training_table(df)
        model = train_with_chronological_test(table)
        save_model(model, model_file)

    states, h2h = build_current_states(df)
    teams = list_teams(df)
    return df, model, states, h2h, teams


def predict_match(home: str, away: str, neutral: bool, model, states, h2h) -> dict:
    if home == away:
        raise ValueError("Нужно выбрать две разные команды.")
    if home not in states:
        raise ValueError(f"Команда не найдена в базе: {home}")
    if away not in states:
        raise ValueError(f"Команда не найдена в базе: {away}")

    features = make_match_features(home, away, neutral, states, h2h)
    ml_probs = model.predict_proba(features)[0]
    fuzzy = fuzzy_probabilities(features)
    fuzzy_probs = np.array([fuzzy["away_win"], fuzzy["draw"], fuzzy["home_win"]])

    final_probs = ml_probs * 0.82 + fuzzy_probs * 0.18
    final_probs = final_probs / final_probs.sum()

    home_xg, away_xg = estimate_expected_goals(features)
    scores = scoreline_probabilities(home_xg, away_xg, max_goals=6)[:8]

    return {
        "home": home,
        "away": away,
        "neutral": neutral,
        "prob_away_win": float(final_probs[0]),
        "prob_draw": float(final_probs[1]),
        "prob_home_win": float(final_probs[2]),
        "expected_goals_home": home_xg,
        "expected_goals_away": away_xg,
        "top_scorelines": scores,
        "features": features.iloc[0].to_dict(),
        "explanations": explain_features(features, home, away),
    }
