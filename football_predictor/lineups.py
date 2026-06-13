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


def _history_strength(
    lineup_history: pd.DataFrame,
    team: str,
    selected_players: list[str],
    match_date: str | pd.Timestamp | None,
) -> SquadAssessment | None:
    if lineup_history is None or lineup_history.empty or not selected_players:
        return None
    history = lineup_history[lineup_history["team"] == team].copy()
    if match_date is not None:
        cutoff = pd.Timestamp(match_date)
        history = history[pd.to_datetime(history["date"], errors="coerce") < cutoff]
    if history.empty:
        return SquadAssessment(
            team=team,
            available=True,
            expected_strength=1.0,
            selected_strength=1.0,
            relative_strength=1.0,
            missing_key_players=0,
            players_used=len(selected_players),
            explanation="Стартовый состав загружен автоматически, но истории составов пока недостаточно для надёжной оценки силы; применена нейтральная поправка.",
        )

    history["starter"] = history["starter"].fillna(False).astype(bool)
    history["minutes"] = pd.to_numeric(history["minutes"], errors="coerce").fillna(0)
    max_date = pd.to_datetime(history["date"], errors="coerce").max()
    age_days = (max_date - pd.to_datetime(history["date"], errors="coerce")).dt.days.clip(lower=0)
    history["recency"] = np.exp(-age_days / 240.0)
    history["appearance_value"] = history["recency"] * (
        1.0 + 1.6 * history["starter"].astype(float) + np.clip(history["minutes"], 0, 120) / 90.0
    )
    player_value = history.groupby("player", as_index=True)["appearance_value"].sum().sort_values(ascending=False)
    if player_value.empty:
        return None

    expected_names = list(player_value.head(11).index)
    expected_strength = float(player_value.head(11).sum())
    selected_strength = float(player_value.reindex(selected_players).fillna(player_value.median() * 0.25).sum())
    relative = selected_strength / expected_strength if expected_strength > 0 else 1.0
    missing = len(set(expected_names[:5]) - set(selected_players))
    return SquadAssessment(
        team=team,
        available=True,
        expected_strength=expected_strength,
        selected_strength=selected_strength,
        relative_strength=float(np.clip(relative, 0.78, 1.08)),
        missing_key_players=missing,
        players_used=len(selected_players),
        explanation=(
            f"Стартовый состав получен автоматически. Сила относительно наиболее часто использовавшегося состава: "
            f"{relative * 100:.1f}%; отсутствуют {missing} из пяти наиболее значимых игроков по истории составов."
        ),
    )


def assess_squad(
    player_pool: pd.DataFrame,
    team: str,
    selected_players: list[str] | None = None,
    lineup_history: pd.DataFrame | None = None,
    match_date: str | pd.Timestamp | None = None,
) -> SquadAssessment:
    selected_players = selected_players or []
    team_rows = player_pool[player_pool["team"] == team].copy() if player_pool is not None and not player_pool.empty else pd.DataFrame()

    if not team_rows.empty and team_rows["rating"].notna().sum() >= 11:
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
                f"Стартовый состав получен автоматически; использовано игроков: {len(selected)}; "
                f"сила относительно оптимального состава: {relative * 100:.1f}%; ключевых потерь: {missing}."
            ),
        )

    history_based = _history_strength(lineup_history, team, selected_players, match_date)
    if history_based is not None:
        return history_based

    if selected_players:
        return SquadAssessment(
            team=team,
            available=True,
            expected_strength=1.0,
            selected_strength=1.0,
            relative_strength=1.0,
            missing_key_players=0,
            players_used=len(selected_players),
            explanation="Стартовый состав загружен автоматически, но открытый источник не предоставляет достаточных рейтингов игроков; применена нейтральная поправка.",
        )

    return SquadAssessment(team, False, 0.0, 0.0, 1.0, 0, 0, "Стартовые составы ещё не опубликованы; прогноз рассчитан без поправки на состав.")
