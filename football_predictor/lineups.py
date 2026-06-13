from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

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


def _player_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _player_effective_rating(row: pd.Series) -> float:
    # ``rating`` is a transparent national-team usage index, not a commercial
    # player ability rating.  Missing club minutes must remain neutral rather
    # than being replaced with a fabricated value.
    rating = float(row.get("rating", 0.0) or 0.0)
    minutes = pd.to_numeric(pd.Series([row.get("club_minutes_90d")]), errors="coerce").iloc[0]
    fitness = 1.0 if pd.isna(minutes) else float(np.clip(minutes / 900.0, 0.70, 1.08))
    availability = 1.0 if bool(row.get("available", True)) else 0.0
    return rating * fitness * availability


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
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
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
            explanation="Официальный состав получен, но истории составов пока недостаточно; применена нейтральная поправка.",
        )

    history["starter"] = history["starter"].fillna(False).astype(bool)
    history["minutes"] = pd.to_numeric(history["minutes"], errors="coerce").fillna(0)
    max_date = pd.to_datetime(history["date"], errors="coerce").max()
    age_days = (max_date - pd.to_datetime(history["date"], errors="coerce")).dt.days.clip(lower=0)
    history["recency"] = np.exp(-age_days / 240.0)
    history["appearance_value"] = history["recency"] * (
        1.0 + 1.6 * history["starter"].astype(float) + np.clip(history["minutes"], 0, 120) / 90.0
    )
    history["player_key"] = history["player"].map(_player_key)
    player_value = history.groupby("player_key", as_index=True)["appearance_value"].sum().sort_values(ascending=False)
    display_names = history.drop_duplicates("player_key").set_index("player_key")["player"].to_dict()
    if player_value.empty:
        return None

    selected_keys = {_player_key(name) for name in selected_players if _player_key(name)}
    expected_keys = list(player_value.head(11).index)
    fallback = float(player_value.median() * 0.25) if not player_value.empty else 0.0
    expected_strength = float(player_value.head(11).sum())
    selected_strength = float(sum(player_value.get(key, fallback) for key in selected_keys))
    relative = selected_strength / expected_strength if expected_strength > 0 else 1.0
    key_five = set(expected_keys[:5])
    missing = len(key_five - selected_keys)
    missing_names = [display_names.get(key, key) for key in expected_keys[:5] if key not in selected_keys]
    detail = f"; вероятные ключевые отсутствия: {', '.join(missing_names[:3])}" if missing_names else ""
    return SquadAssessment(
        team=team,
        available=True,
        expected_strength=expected_strength,
        selected_strength=selected_strength,
        relative_strength=float(np.clip(relative, 0.82, 1.08)),
        missing_key_players=missing,
        players_used=len(selected_players),
        explanation=(
            f"Официальный состав учтён. Индекс состава относительно привычной стартовой основы: "
            f"{relative * 100:.1f}%; отсутствуют {missing} из пяти наиболее часто используемых игроков{detail}."
        ),
    )


def assess_squad(
    player_pool: pd.DataFrame,
    team: str,
    selected_players: list[str] | None = None,
    lineup_history: pd.DataFrame | None = None,
    match_date: str | pd.Timestamp | None = None,
) -> SquadAssessment:
    selected_players = [str(name).strip() for name in (selected_players or []) if str(name).strip()]

    # Never infer a lineup adjustment from an expected squad.  Before official
    # lineups appear the prediction must remain explicitly lineup-neutral.
    if not selected_players:
        return SquadAssessment(
            team, False, 0.0, 0.0, 1.0, 0, 0,
            "Стартовый состав ещё не опубликован; прогноз рассчитан без поправки на игроков.",
        )

    team_rows = player_pool[player_pool["team"] == team].copy() if player_pool is not None and not player_pool.empty else pd.DataFrame()
    if not team_rows.empty and team_rows["rating"].notna().sum() >= 11:
        team_rows["player_key"] = team_rows["player"].map(_player_key)
        team_rows["effective"] = team_rows.apply(_player_effective_rating, axis=1)
        team_rows = team_rows.sort_values("effective", ascending=False).drop_duplicates("player_key")
        expected = float(team_rows.head(11)["effective"].sum())
        selected_keys = {_player_key(name) for name in selected_players if _player_key(name)}
        selected = team_rows[team_rows["player_key"].isin(selected_keys)].head(11)

        # If names from two sources do not match well, history matching is safer.
        if len(selected) >= 7:
            selected_strength = float(selected["effective"].sum())
            unknown_count = max(0, min(11, len(selected_players)) - len(selected))
            if unknown_count:
                selected_strength += unknown_count * float(team_rows.head(18)["effective"].median())
            relative = selected_strength / expected if expected > 0 else 1.0
            key_names = set(team_rows.head(5)["player_key"])
            missing = len(key_names - selected_keys)
            return SquadAssessment(
                team=team,
                available=True,
                expected_strength=expected,
                selected_strength=selected_strength,
                relative_strength=float(np.clip(relative, 0.82, 1.08)),
                missing_key_players=missing,
                players_used=len(selected_players),
                explanation=(
                    f"Официальный состав учтён; сопоставлено {len(selected)} игроков. "
                    f"Индекс использования состава относительно привычной основы: {relative * 100:.1f}%; "
                    f"ключевых отсутствий: {missing}. Это индекс использования в сборной, а не коммерческий рейтинг игроков."
                ),
            )

    history_based = _history_strength(lineup_history, team, selected_players, match_date)
    if history_based is not None:
        return history_based

    return SquadAssessment(
        team=team,
        available=True,
        expected_strength=1.0,
        selected_strength=1.0,
        relative_strength=1.0,
        missing_key_players=0,
        players_used=len(selected_players),
        explanation="Официальный состав загружен, но открытых данных недостаточно для надёжной оценки игроков; применена нейтральная поправка.",
    )
