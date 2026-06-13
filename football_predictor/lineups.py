from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SquadAssessment:
    team: str
    available: bool
    expected_strength: float
    selected_strength: float
    relative_strength: float
    missing_key_players: int
    players_used: int
    explanation: str


def _player_effective_rating(row: pd.Series) -> float:
    rating = float(row.get("rating", 0.0) or 0.0)
    minutes = float(row.get("club_minutes_90d", 0.0) or 0.0)
    caps = float(row.get("national_caps", 0.0) or 0.0)
    availability = 1.0 if bool(row.get("available", True)) else 0.0
    fitness = np.clip(minutes / 900.0, 0.55, 1.10)
    experience = 1.0 + min(caps, 100.0) / 1000.0
    return rating * fitness * experience * availability


def assess_squad(
    player_pool: pd.DataFrame,
    team: str,
    selected_players: list[str] | None = None,
) -> SquadAssessment:
    team_rows = player_pool[player_pool["team"] == team].copy()
    if team_rows.empty or team_rows["rating"].notna().sum() < 11:
        return SquadAssessment(team, False, 0.0, 0.0, 1.0, 0, 0, "Нет достаточных данных о составе.")
    team_rows["effective"] = team_rows.apply(_player_effective_rating, axis=1)
    expected = float(team_rows.nlargest(11, "effective")["effective"].sum())
    if selected_players:
        selected = team_rows[team_rows["player"].isin(selected_players)].nlargest(11, "effective")
    else:
        selected = team_rows[team_rows["available"]].nlargest(11, "effective")
    selected_strength = float(selected["effective"].sum())
    relative = selected_strength / expected if expected > 0 else 1.0
    key_names = set(team_rows.nlargest(5, "effective")["player"])
    selected_names = set(selected["player"])
    missing = len(key_names - selected_names)
    return SquadAssessment(
        team=team,
        available=True,
        expected_strength=expected,
        selected_strength=selected_strength,
        relative_strength=float(np.clip(relative, 0.50, 1.08)),
        missing_key_players=missing,
        players_used=len(selected),
        explanation=(
            f"Использовано игроков: {len(selected)}; сила относительно оптимального состава: "
            f"{relative * 100:.1f}%; отсутствуют ключевые игроки: {missing}."
        ),
    )
