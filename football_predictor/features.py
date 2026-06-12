from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .elo import BASE_ELO, HOME_ADVANTAGE, update_elo


@dataclass
class TeamState:
    elo: float = BASE_ELO
    matches: int = 0
    recent_points: deque = field(default_factory=lambda: deque(maxlen=15))
    recent_goal_diff: deque = field(default_factory=lambda: deque(maxlen=15))
    recent_goals_for: deque = field(default_factory=lambda: deque(maxlen=15))
    recent_goals_against: deque = field(default_factory=lambda: deque(maxlen=15))


def _avg(values, default: float = 0.0) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float(default)


def _avg_last(values, n: int, default: float = 0.0) -> float:
    values = list(values)[-n:]
    return float(np.mean(values)) if values else float(default)


def _rate_last(values, n: int, predicate, default: float = 0.0) -> float:
    values = list(values)[-n:]
    return float(np.mean([1.0 if predicate(v) else 0.0 for v in values])) if values else float(default)


def _points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _result_class(home_score: int, away_score: int) -> int:
    """0 = away win, 1 = draw, 2 = home win."""
    if home_score > away_score:
        return 2
    if home_score == away_score:
        return 1
    return 0


FEATURE_COLUMNS = [
    "elo_diff",
    "elo_diff_raw",
    "abs_elo_diff",
    "home_elo",
    "away_elo",
    "home_matches",
    "away_matches",
    "experience_diff",
    "form_points_diff_3",
    "form_points_diff_5",
    "form_points_diff_10",
    "form_points_diff_15",
    "win_rate_diff_5",
    "win_rate_diff_10",
    "draw_rate_sum_5",
    "goal_diff_form_diff_3",
    "goal_diff_form_diff_5",
    "goal_diff_form_diff_10",
    "goal_diff_form_diff_15",
    "home_gf_5",
    "home_ga_5",
    "away_gf_5",
    "away_ga_5",
    "home_gf_10",
    "home_ga_10",
    "away_gf_10",
    "away_ga_10",
    "attack_diff_10",
    "defense_diff_10",
    "total_goals_diff_10",
    "h2h_goal_diff",
    "h2h_points_diff",
    "h2h_matches",
    "neutral",
]


def _h2h_features(home: str, away: str, h2h_records) -> dict[str, float]:
    pair_key = tuple(sorted([home, away]))
    records = list(h2h_records[pair_key])
    if not records:
        return {"h2h_goal_diff": 0.0, "h2h_points_diff": 0.0, "h2h_matches": 0.0}

    goal_diffs: list[float] = []
    points_diffs: list[float] = []
    team0 = pair_key[0]
    for rec in records:
        gd_team0 = float(rec[0])
        points_team0 = float(rec[1])
        if points_team0 == 3:
            points_team1 = 0.0
        elif points_team0 == 1:
            points_team1 = 1.0
        else:
            points_team1 = 3.0

        if home == team0:
            goal_diffs.append(gd_team0)
            points_diffs.append(points_team0 - points_team1)
        else:
            goal_diffs.append(-gd_team0)
            points_diffs.append(points_team1 - points_team0)

    return {
        "h2h_goal_diff": float(np.mean(goal_diffs)),
        "h2h_points_diff": float(np.mean(points_diffs)),
        "h2h_matches": float(len(records)),
    }


def _make_feature_row(home: str, away: str, neutral: bool, states: dict[str, TeamState], h2h_records) -> dict[str, Any]:
    home_state = states[home]
    away_state = states[away]
    home_adv = 0.0 if neutral else HOME_ADVANTAGE
    elo_diff_raw = home_state.elo - away_state.elo
    elo_diff = elo_diff_raw + home_adv

    home_gf_10 = _avg_last(home_state.recent_goals_for, 10, 1.25)
    home_ga_10 = _avg_last(home_state.recent_goals_against, 10, 1.25)
    away_gf_10 = _avg_last(away_state.recent_goals_for, 10, 1.25)
    away_ga_10 = _avg_last(away_state.recent_goals_against, 10, 1.25)

    row = {
        "elo_diff": elo_diff,
        "elo_diff_raw": elo_diff_raw,
        "abs_elo_diff": abs(elo_diff_raw),
        "home_elo": home_state.elo,
        "away_elo": away_state.elo,
        "home_matches": home_state.matches,
        "away_matches": away_state.matches,
        "experience_diff": np.log1p(home_state.matches) - np.log1p(away_state.matches),
        "form_points_diff_3": _avg_last(home_state.recent_points, 3, 1.0) - _avg_last(away_state.recent_points, 3, 1.0),
        "form_points_diff_5": _avg_last(home_state.recent_points, 5, 1.0) - _avg_last(away_state.recent_points, 5, 1.0),
        "form_points_diff_10": _avg_last(home_state.recent_points, 10, 1.0) - _avg_last(away_state.recent_points, 10, 1.0),
        "form_points_diff_15": _avg_last(home_state.recent_points, 15, 1.0) - _avg_last(away_state.recent_points, 15, 1.0),
        "win_rate_diff_5": _rate_last(home_state.recent_points, 5, lambda x: x == 3, 0.33) - _rate_last(away_state.recent_points, 5, lambda x: x == 3, 0.33),
        "win_rate_diff_10": _rate_last(home_state.recent_points, 10, lambda x: x == 3, 0.33) - _rate_last(away_state.recent_points, 10, lambda x: x == 3, 0.33),
        "draw_rate_sum_5": _rate_last(home_state.recent_points, 5, lambda x: x == 1, 0.25) + _rate_last(away_state.recent_points, 5, lambda x: x == 1, 0.25),
        "goal_diff_form_diff_3": _avg_last(home_state.recent_goal_diff, 3, 0.0) - _avg_last(away_state.recent_goal_diff, 3, 0.0),
        "goal_diff_form_diff_5": _avg_last(home_state.recent_goal_diff, 5, 0.0) - _avg_last(away_state.recent_goal_diff, 5, 0.0),
        "goal_diff_form_diff_10": _avg_last(home_state.recent_goal_diff, 10, 0.0) - _avg_last(away_state.recent_goal_diff, 10, 0.0),
        "goal_diff_form_diff_15": _avg_last(home_state.recent_goal_diff, 15, 0.0) - _avg_last(away_state.recent_goal_diff, 15, 0.0),
        "home_gf_5": _avg_last(home_state.recent_goals_for, 5, 1.25),
        "home_ga_5": _avg_last(home_state.recent_goals_against, 5, 1.25),
        "away_gf_5": _avg_last(away_state.recent_goals_for, 5, 1.25),
        "away_ga_5": _avg_last(away_state.recent_goals_against, 5, 1.25),
        "home_gf_10": home_gf_10,
        "home_ga_10": home_ga_10,
        "away_gf_10": away_gf_10,
        "away_ga_10": away_ga_10,
        "attack_diff_10": home_gf_10 - away_gf_10,
        "defense_diff_10": away_ga_10 - home_ga_10,
        "total_goals_diff_10": (home_gf_10 + home_ga_10) - (away_gf_10 + away_ga_10),
        "neutral": int(neutral),
    }
    row.update(_h2h_features(home, away, h2h_records))
    return row


def build_training_table(df: pd.DataFrame, min_year: int = 1950) -> pd.DataFrame:
    """Create a machine-learning table from historical matches.

    Features for each row are calculated only from matches that happened earlier.
    This prevents looking into the future.
    """
    states: dict[str, TeamState] = defaultdict(TeamState)
    h2h_records: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=8))
    rows: list[dict] = []

    df = df.sort_values("date").reset_index(drop=True)

    for _, m in df.iterrows():
        home = str(m["home_team"])
        away = str(m["away_team"])
        hs = int(m["home_score"])
        aas = int(m["away_score"])
        neutral = bool(m["neutral"])
        date = m["date"]

        if date.year >= min_year:
            row = _make_feature_row(home, away, neutral, states, h2h_records)
            row.update({
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": aas,
                "target": _result_class(hs, aas),
            })
            rows.append(row)

        _update_states_after_match(states, h2h_records, home, away, hs, aas, neutral)

    table = pd.DataFrame(rows)
    table = table.dropna(subset=FEATURE_COLUMNS + ["target"])
    return table


def _update_states_after_match(states, h2h_records, home: str, away: str, hs: int, aas: int, neutral: bool) -> None:
    home_state = states[home]
    away_state = states[away]

    new_home_elo, new_away_elo = update_elo(home_state.elo, away_state.elo, hs, aas, neutral)
    home_state.elo = new_home_elo
    away_state.elo = new_away_elo
    home_state.matches += 1
    away_state.matches += 1

    home_state.recent_points.append(_points_for(hs, aas))
    away_state.recent_points.append(_points_for(aas, hs))
    home_state.recent_goal_diff.append(hs - aas)
    away_state.recent_goal_diff.append(aas - hs)
    home_state.recent_goals_for.append(hs)
    home_state.recent_goals_against.append(aas)
    away_state.recent_goals_for.append(aas)
    away_state.recent_goals_against.append(hs)

    pair_key = tuple(sorted([home, away]))
    team0 = pair_key[0]
    if home == team0:
        gd_team0 = hs - aas
        points_team0 = _points_for(hs, aas)
    else:
        gd_team0 = aas - hs
        points_team0 = _points_for(aas, hs)
    h2h_records[pair_key].append((gd_team0, points_team0))


def build_current_states(df: pd.DataFrame) -> tuple[dict[str, TeamState], dict[tuple[str, str], deque]]:
    states: dict[str, TeamState] = defaultdict(TeamState)
    h2h_records: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=8))

    for _, m in df.sort_values("date").iterrows():
        _update_states_after_match(
            states=states,
            h2h_records=h2h_records,
            home=str(m["home_team"]),
            away=str(m["away_team"]),
            hs=int(m["home_score"]),
            aas=int(m["away_score"]),
            neutral=bool(m["neutral"]),
        )

    return states, h2h_records


def make_match_features(home: str, away: str, neutral: bool, states: dict[str, TeamState], h2h_records) -> pd.DataFrame:
    row = _make_feature_row(home, away, neutral, states, h2h_records)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
