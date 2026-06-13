from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import ENRICHED_STATS_PATH, MATCH_LINEUPS_PATH, PLAYER_POOL_PATH
from .data_loader import normalize_team_name
from .world_cup_live import FOOTBALL_DATA_BASE, _headers


STAT_ALIASES = {
    "shots": {"shots", "total_shots", "shots_total"},
    "shots_on_target": {"shots_on_goal", "shots_on_target", "shotsongoal"},
    "corners": {"corner_kicks", "corners", "cornerkicks"},
    "yellow_cards": {"yellow_cards", "yellowcards"},
    "red_cards": {"red_cards", "redcards"},
    "possession": {"ball_possession", "possession", "ballpossession"},
}


def _safe_number(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, str):
        value = value.replace("%", "").strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _team_name(obj: Any) -> str:
    if not isinstance(obj, dict):
        return normalize_team_name(str(obj or ""))
    return normalize_team_name(str(obj.get("name") or obj.get("shortName") or obj.get("tla") or ""))


def _statistics_map(team_obj: dict[str, Any]) -> dict[str, float]:
    raw = team_obj.get("statistics") or {}
    out: dict[str, float] = {}
    if isinstance(raw, list):
        items = []
        for item in raw:
            if isinstance(item, dict):
                key = item.get("type") or item.get("name") or item.get("key")
                value = item.get("value")
                items.append((key, value))
    elif isinstance(raw, dict):
        items = list(raw.items())
    else:
        items = []
    normalized = {str(key or "").lower().replace(" ", "_").replace("-", "_"): _safe_number(value) for key, value in items}
    for canonical, aliases in STAT_ALIASES.items():
        for alias in aliases:
            if alias in normalized and pd.notna(normalized[alias]):
                out[canonical] = normalized[alias]
                break
    return out


def _lineup_names(team_obj: dict[str, Any]) -> list[str]:
    entries = team_obj.get("lineup") or team_obj.get("startingEleven") or team_obj.get("startingXI") or []
    names: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            names.append(entry.strip())
        elif isinstance(entry, dict):
            player = entry.get("player") if isinstance(entry.get("player"), dict) else entry
            name = player.get("name") or player.get("shortName")
            if name:
                names.append(str(name).strip())
    return [name for name in names if name][:11]


def _half_time_score(payload: dict[str, Any], side: str) -> float:
    score = payload.get("score") or {}
    half = score.get("halfTime") or {}
    return _safe_number(half.get(side))


def _parse_match_detail(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    home_obj = payload.get("homeTeam") or {}
    away_obj = payload.get("awayTeam") or {}
    home = _team_name(home_obj)
    away = _team_name(away_obj)
    date = pd.to_datetime(payload.get("utcDate"), utc=True, errors="coerce")
    if not home or not away or pd.isna(date):
        return None, []

    hs = _statistics_map(home_obj)
    aws = _statistics_map(away_obj)
    row = {
        "date": date.date().isoformat(),
        "home_team": home,
        "away_team": away,
        "home_xg": np.nan,
        "away_xg": np.nan,
        "home_shots": hs.get("shots", np.nan),
        "away_shots": aws.get("shots", np.nan),
        "home_shots_on_target": hs.get("shots_on_target", np.nan),
        "away_shots_on_target": aws.get("shots_on_target", np.nan),
        "home_corners": hs.get("corners", np.nan),
        "away_corners": aws.get("corners", np.nan),
        "home_yellow_cards": hs.get("yellow_cards", np.nan),
        "away_yellow_cards": aws.get("yellow_cards", np.nan),
        "home_red_cards": hs.get("red_cards", np.nan),
        "away_red_cards": aws.get("red_cards", np.nan),
        "home_possession": hs.get("possession", np.nan),
        "away_possession": aws.get("possession", np.nan),
        "home_ppda": np.nan,
        "away_ppda": np.nan,
        "home_ht_score": _half_time_score(payload, "home"),
        "away_ht_score": _half_time_score(payload, "away"),
        "referee": "",
        "source": "football-data.org",
        "source_match_id": str(payload.get("id") or ""),
    }
    referees = payload.get("referees") or []
    if referees and isinstance(referees[0], dict):
        row["referee"] = str(referees[0].get("name") or "")

    lineups: list[dict[str, Any]] = []
    for team, team_obj in ((home, home_obj), (away, away_obj)):
        for player in _lineup_names(team_obj):
            lineups.append({"date": row["date"], "team": team, "player": player, "starter": True, "minutes": 0})
    return row, lineups


def _merge_non_null(existing: pd.DataFrame, addition: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if existing.empty:
        return addition.copy()
    if addition.empty:
        return existing.copy()
    all_columns = list(dict.fromkeys(list(existing.columns) + list(addition.columns)))
    existing = existing.reindex(columns=all_columns)
    addition = addition.reindex(columns=all_columns)
    combined = pd.concat([existing, addition], ignore_index=True)
    combined = combined.sort_values(keys).reset_index(drop=True)
    rows: list[pd.Series] = []
    for _, group in combined.groupby(keys, dropna=False, sort=False):
        output = group.iloc[0].copy()
        for column in all_columns:
            values = group[column].dropna()
            if not values.empty:
                output[column] = values.iloc[-1]
        rows.append(output)
    return pd.DataFrame(rows, columns=all_columns)


def _update_player_pool(lineups: pd.DataFrame, path: str | Path = PLAYER_POOL_PATH) -> None:
    if lineups.empty:
        return
    history = lineups.copy()
    history["starter"] = history["starter"].fillna(False).astype(bool)
    usage = history.groupby(["team", "player"], as_index=False).agg(
        appearances=("date", "count"), starts=("starter", "sum"), minutes=("minutes", "sum")
    )
    usage["position"] = ""
    usage["rating"] = 65.0 + np.clip(usage["starts"] * 1.1 + usage["appearances"] * 0.45, 0, 18)
    usage["club_minutes_90d"] = 900.0
    usage["national_caps"] = usage["appearances"]
    usage["available"] = True
    pool = usage[["team", "player", "position", "rating", "club_minutes_90d", "national_caps", "available"]]
    old = pd.read_csv(path) if Path(path).exists() and Path(path).stat().st_size else pd.DataFrame()
    merged = pd.concat([old, pool], ignore_index=True)
    merged = merged.drop_duplicates(["team", "player"], keep="last")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)


def update_football_data_history(
    api_key: str,
    seasons: tuple[int, ...] = (2010, 2014, 2018, 2022, 2026),
    max_details: int = 90,
    pause_seconds: float = 6.2,
) -> dict[str, int]:
    """Enrich World Cup history with half-time scores, stats and lineups.

    The API may restrict historical seasons or deep statistics by plan. Restricted
    seasons are skipped without deleting data already collected from other sources.
    """
    key = str(api_key or "").strip()
    if not key:
        return {"matches": 0, "lineups": 0, "restricted_seasons": 0}

    existing = pd.read_csv(ENRICHED_STATS_PATH) if Path(ENRICHED_STATS_PATH).exists() else pd.DataFrame()
    known_ids = set(existing.get("source_match_id", pd.Series(dtype=str)).dropna().astype(str))
    stat_rows: list[dict[str, Any]] = []
    lineup_rows: list[dict[str, Any]] = []
    restricted = 0

    for season in seasons:
        try:
            response = requests.get(
                f"{FOOTBALL_DATA_BASE}/competitions/WC/matches",
                params={"season": season, "status": "FINISHED"},
                headers=_headers(key, unfold_lineups=False),
                timeout=30,
            )
            if response.status_code in {403, 404}:
                restricted += 1
                continue
            response.raise_for_status()
            items = response.json().get("matches", []) or []
        except Exception:
            continue

        for summary in items:
            match_id = str(summary.get("id") or "")
            if not match_id or match_id in known_ids:
                continue
            if len(stat_rows) >= max_details:
                break
            try:
                detail = requests.get(
                    f"{FOOTBALL_DATA_BASE}/matches/{match_id}",
                    headers=_headers(key, unfold_lineups=True),
                    timeout=30,
                )
                if detail.status_code in {403, 404}:
                    continue
                detail.raise_for_status()
                row, lineups = _parse_match_detail(detail.json())
                if row is not None:
                    stat_rows.append(row)
                    lineup_rows.extend(lineups)
                    known_ids.add(match_id)
                time.sleep(max(pause_seconds, 0.0))
            except Exception:
                continue
        if len(stat_rows) >= max_details:
            break

    if stat_rows:
        merged_stats = _merge_non_null(existing, pd.DataFrame(stat_rows), ["date", "home_team", "away_team"])
        Path(ENRICHED_STATS_PATH).parent.mkdir(parents=True, exist_ok=True)
        merged_stats.to_csv(ENRICHED_STATS_PATH, index=False)

    if lineup_rows:
        old_lineups = pd.read_csv(MATCH_LINEUPS_PATH) if Path(MATCH_LINEUPS_PATH).exists() else pd.DataFrame()
        merged_lineups = pd.concat([old_lineups, pd.DataFrame(lineup_rows)], ignore_index=True)
        merged_lineups = merged_lineups.drop_duplicates(["date", "team", "player"], keep="last")
        merged_lineups.to_csv(MATCH_LINEUPS_PATH, index=False)
        _update_player_pool(merged_lineups)

    return {"matches": len(stat_rows), "lineups": len(lineup_rows), "restricted_seasons": restricted}
