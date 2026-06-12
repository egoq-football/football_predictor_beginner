from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .elo import BASE_ELO, update_elo


@dataclass
class TeamState:
    elo: float = BASE_ELO
    matches: int = 0
    recent_points: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_goal_diff: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_goals_for: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_goals_against: deque = field(default_factory=lambda: deque(maxlen=10))


def _avg(values: deque, default: float = 0.0) -> float:
    return float(np.mean(values)) if values else default


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
    "home_elo",
    "away_elo",
    "home_matches",
    "away_matches",
    "form_points_diff_5",
    "form_points_diff_10",
    "goal_diff_form_diff_5",
    "goal_diff_form_diff_10",
    "home_gf_5",
    "home_ga_5",
    "away_gf_5",
    "away_ga_5",
    "h2h_goal_diff",
    "neutral",
]


def build_training_table(df: pd.DataFrame, min_year: int = 1950) -> pd.DataFrame:
    """Create a machine-learning table from historical matches.

    Important: features for each row are calculated only from matches that happened earlier.
    This prevents looking into the future.
    """
    states: dict[str, TeamState] = defaultdict(TeamState)
    h2h_goal_diffs: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=5))
    rows: list[dict] = []

    df = df.sort_values("date").reset_index(drop=True)

    for _, m in df.iterrows():
        home = str(m["home_team"])
        away = str(m["away_team"])
        hs = int(m["home_score"])
        aas = int(m["away_score"])
        neutral = bool(m["neutral"])
        date = m["date"]

        home_state = states[home]
        away_state = states[away]

        pair_key = tuple(sorted([home, away]))
        previous_h2h = h2h_goal_diffs[pair_key]
        h2h_home_perspective = 0.0
        if previous_h2h:
            # Values are stored from alphabetical first team's perspective.
            base_value = float(np.mean(previous_h2h))
            h2h_home_perspective = base_value if home == pair_key[0] else -base_value

        home_recent_5_points = list(home_state.recent_points)[-5:]
        away_recent_5_points = list(away_state.recent_points)[-5:]
        home_recent_5_gd = list(home_state.recent_goal_diff)[-5:]
        away_recent_5_gd = list(away_state.recent_goal_diff)[-5:]

        if date.year >= min_year:
            rows.append(
                {
                    "date": date,
                    "home_team": home,
                    "away_team": away,
                    "home_score": hs,
                    "away_score": aas,
                    "target": _result_class(hs, aas),
                    "elo_diff": home_state.elo - away_state.elo + (0 if neutral else 60),
                    "home_elo": home_state.elo,
                    "away_elo": away_state.elo,
                    "home_matches": home_state.matches,
                    "away_matches": away_state.matches,
                    "form_points_diff_5": np.mean(home_recent_5_points or [1.0]) - np.mean(away_recent_5_points or [1.0]),
                    "form_points_diff_10": _avg(home_state.recent_points, 1.0) - _avg(away_state.recent_points, 1.0),
                    "goal_diff_form_diff_5": np.mean(home_recent_5_gd or [0.0]) - np.mean(away_recent_5_gd or [0.0]),
                    "goal_diff_form_diff_10": _avg(home_state.recent_goal_diff, 0.0) - _avg(away_state.recent_goal_diff, 0.0),
                    "home_gf_5": float(np.mean(list(home_state.recent_goals_for)[-5:] or [1.2])),
                    "home_ga_5": float(np.mean(list(home_state.recent_goals_against)[-5:] or [1.2])),
                    "away_gf_5": float(np.mean(list(away_state.recent_goals_for)[-5:] or [1.2])),
                    "away_ga_5": float(np.mean(list(away_state.recent_goals_against)[-5:] or [1.2])),
                    "h2h_goal_diff": h2h_home_perspective,
                    "neutral": int(neutral),
                }
            )

        # Update states after the features are created.
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

        h2h_value = (hs - aas) if home == pair_key[0] else (aas - hs)
        h2h_goal_diffs[pair_key].append(h2h_value)

    table = pd.DataFrame(rows)
    table = table.dropna(subset=FEATURE_COLUMNS + ["target"])
    return table


def build_current_states(df: pd.DataFrame) -> tuple[dict[str, TeamState], dict[tuple[str, str], deque]]:
    states: dict[str, TeamState] = defaultdict(TeamState)
    h2h_goal_diffs: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=5))

    for _, m in df.sort_values("date").iterrows():
        home = str(m["home_team"])
        away = str(m["away_team"])
        hs = int(m["home_score"])
        aas = int(m["away_score"])
        neutral = bool(m["neutral"])

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
        h2h_value = (hs - aas) if home == pair_key[0] else (aas - hs)
        h2h_goal_diffs[pair_key].append(h2h_value)

    return states, h2h_goal_diffs


def make_match_features(home: str, away: str, neutral: bool, states: dict[str, TeamState], h2h_goal_diffs) -> pd.DataFrame:
    home_state = states[home]
    away_state = states[away]
    pair_key = tuple(sorted([home, away]))
    previous_h2h = h2h_goal_diffs[pair_key]
    h2h_home_perspective = 0.0
    if previous_h2h:
        base_value = float(np.mean(previous_h2h))
        h2h_home_perspective = base_value if home == pair_key[0] else -base_value

    home_recent_5_points = list(home_state.recent_points)[-5:]
    away_recent_5_points = list(away_state.recent_points)[-5:]
    home_recent_5_gd = list(home_state.recent_goal_diff)[-5:]
    away_recent_5_gd = list(away_state.recent_goal_diff)[-5:]

    row = {
        "elo_diff": home_state.elo - away_state.elo + (0 if neutral else 60),
        "home_elo": home_state.elo,
        "away_elo": away_state.elo,
        "home_matches": home_state.matches,
        "away_matches": away_state.matches,
        "form_points_diff_5": np.mean(home_recent_5_points or [1.0]) - np.mean(away_recent_5_points or [1.0]),
        "form_points_diff_10": _avg(home_state.recent_points, 1.0) - _avg(away_state.recent_points, 1.0),
        "goal_diff_form_diff_5": np.mean(home_recent_5_gd or [0.0]) - np.mean(away_recent_5_gd or [0.0]),
        "goal_diff_form_diff_10": _avg(home_state.recent_goal_diff, 0.0) - _avg(away_state.recent_goal_diff, 0.0),
        "home_gf_5": float(np.mean(list(home_state.recent_goals_for)[-5:] or [1.2])),
        "home_ga_5": float(np.mean(list(home_state.recent_goals_against)[-5:] or [1.2])),
        "away_gf_5": float(np.mean(list(away_state.recent_goals_for)[-5:] or [1.2])),
        "away_ga_5": float(np.mean(list(away_state.recent_goals_against)[-5:] or [1.2])),
        "h2h_goal_diff": h2h_home_perspective,
        "neutral": int(neutral),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
