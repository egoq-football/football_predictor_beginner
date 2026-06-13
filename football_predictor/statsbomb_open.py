from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import ENRICHED_STATS_PATH, MATCH_LINEUPS_PATH, OPTIONAL_STATS_COLUMNS, PLAYER_POOL_PATH
from .data_loader import clean_optional_stats_frame, normalize_team_name
from .player_usage import rebuild_player_pool_from_lineups

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
HEADERS = {"User-Agent": "world-cup-2026-predictor/4.3"}


def _get_json(url: str, timeout: int = 45) -> Any:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _team_name(obj: Any, side: str | None = None) -> str:
    """Read team names from both match and event StatsBomb schemas."""
    if isinstance(obj, dict):
        preferred = []
        if side == "home":
            preferred.extend(["home_team_name", "homeTeamName"])
        elif side == "away":
            preferred.extend(["away_team_name", "awayTeamName"])
        preferred.extend([
            "name", "team_name", "home_team_name", "away_team_name",
            "short_name", "shortName", "country_name",
        ])
        for key in preferred:
            value = obj.get(key)
            if value not in (None, "") and not isinstance(value, (dict, list)):
                name = normalize_team_name(value)
                if name:
                    return name
        for key in ("team", "home_team", "away_team"):
            nested = obj.get(key)
            if isinstance(nested, dict):
                name = _team_name(nested, side=side)
                if name:
                    return name
        return ""
    return normalize_team_name(obj)


def _event_team(event: dict[str, Any]) -> str:
    return _team_name(event.get("team"))


def _shot_on_target(outcome: str) -> bool:
    return outcome in {"Goal", "Saved", "Saved To Post", "Saved to Post"}


def _card_name(event: dict[str, Any]) -> str:
    for node_name in ("bad_behaviour", "foul_committed"):
        node = event.get(node_name) or {}
        card = node.get("card") or {}
        name = card.get("name")
        if name:
            return str(name)
    return ""


def _opponent(team: str, home: str, away: str) -> str:
    return away if team == home else home


def _parse_events(match: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    home = _team_name(match.get("home_team"), side="home")
    away = _team_name(match.get("away_team"), side="away")
    date = pd.to_datetime(match.get("match_date"), errors="coerce")
    if not home or not away or home == away or pd.isna(date):
        return None

    stats = {home: defaultdict(float), away: defaultdict(float)}
    ht = {home: 0, away: 0}
    possession_events = defaultdict(int)
    recognized_events = 0

    for event in events or []:
        team = _event_team(event)
        if team not in stats:
            continue
        recognized_events += 1
        event_type = str((event.get("type") or {}).get("name") or "")
        possession_team = _team_name(event.get("possession_team"))
        if possession_team in stats:
            possession_events[possession_team] += 1

        if event_type == "Shot":
            stats[team]["shots"] += 1
            shot = event.get("shot") or {}
            xg = shot.get("statsbomb_xg")
            if xg is not None:
                try:
                    stats[team]["xg"] += float(xg)
                except (TypeError, ValueError):
                    pass
            outcome = str((shot.get("outcome") or {}).get("name") or "")
            if _shot_on_target(outcome):
                stats[team]["shots_on_target"] += 1
            if outcome == "Goal" and int(event.get("period") or 0) == 1:
                ht[team] += 1
        elif event_type == "Pass":
            pass_type = str(((event.get("pass") or {}).get("type") or {}).get("name") or "")
            if pass_type == "Corner":
                stats[team]["corners"] += 1
        elif event_type == "Own Goal Against" and int(event.get("period") or 0) == 1:
            ht[_opponent(team, home, away)] += 1
        elif event_type == "Own Goal For" and int(event.get("period") or 0) == 1:
            ht[team] += 1

        card = _card_name(event).lower()
        if card:
            if "yellow" in card:
                stats[team]["yellow_cards"] += 1
            if "red" in card or "second yellow" in card:
                stats[team]["red_cards"] += 1

    if recognized_events < 20:
        return None

    poss_total = sum(possession_events.values())
    home_poss = 100.0 * possession_events.get(home, 0) / poss_total if poss_total else np.nan
    away_poss = 100.0 * possession_events.get(away, 0) / poss_total if poss_total else np.nan
    referee = _team_name(match.get("referee")) if isinstance(match.get("referee"), dict) else str(match.get("referee") or "")

    return {
        "date": date.date().isoformat(),
        "home_team": home,
        "away_team": away,
        "home_xg": stats[home]["xg"] if stats[home]["shots"] else np.nan,
        "away_xg": stats[away]["xg"] if stats[away]["shots"] else np.nan,
        "home_shots": int(stats[home]["shots"]),
        "away_shots": int(stats[away]["shots"]),
        "home_shots_on_target": int(stats[home]["shots_on_target"]),
        "away_shots_on_target": int(stats[away]["shots_on_target"]),
        "home_corners": int(stats[home]["corners"]),
        "away_corners": int(stats[away]["corners"]),
        "home_yellow_cards": int(stats[home]["yellow_cards"]),
        "away_yellow_cards": int(stats[away]["yellow_cards"]),
        "home_red_cards": int(stats[home]["red_cards"]),
        "away_red_cards": int(stats[away]["red_cards"]),
        "home_possession": home_poss,
        "away_possession": away_poss,
        "home_ppda": np.nan,
        "away_ppda": np.nan,
        "home_ht_score": ht[home],
        "away_ht_score": ht[away],
        "referee": referee,
        "source": "StatsBomb Open Data",
        "source_match_id": str(match.get("match_id") or ""),
    }


def _minutes(value: Any, default: float) -> float:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        parts = text.split(":")
        return float(parts[0]) + (float(parts[1]) / 60.0 if len(parts) > 1 else 0.0)
    except (TypeError, ValueError):
        return default


def _parse_lineups(match: dict[str, Any], payload: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    date = pd.to_datetime(match.get("match_date"), errors="coerce")
    date_text = date.date().isoformat() if not pd.isna(date) else ""
    lineup_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    for team_block in payload or []:
        team = _team_name(team_block.get("team_name") or team_block.get("team"))
        if not team or not date_text:
            continue
        for player in team_block.get("lineup", []) or []:
            name = str(player.get("player_name") or player.get("player_nickname") or "").strip()
            if not name:
                continue
            positions = player.get("positions") or []
            starter = False
            minutes = 0.0
            position_name = ""
            for position in positions:
                raw_position = position.get("position")
                if isinstance(raw_position, dict):
                    raw_position = raw_position.get("name")
                position_name = position_name or str(raw_position or "")
                start_min = _minutes(position.get("from"), 0.0)
                end_min = _minutes(position.get("to"), 90.0)
                minutes += max(0.0, min(end_min, 130.0) - max(start_min, 0.0))
                if start_min <= 0.01:
                    starter = True
            lineup_rows.append({
                "date": date_text, "team": team, "player": name,
                "starter": starter, "minutes": min(minutes, 130.0),
            })
            player_rows.append({"team": team, "player": name, "position": position_name})
    return lineup_rows, player_rows


def _read_csv(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])


def _process_match(match: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    match_id = str(match.get("match_id") or "")
    if not match_id:
        return None, [], []
    try:
        events = _get_json(f"{BASE}/events/{match_id}.json")
        stats = _parse_events(match, events)
    except Exception:
        return None, [], []
    if stats is None:
        return None, [], []
    try:
        lineups = _get_json(f"{BASE}/lineups/{match_id}.json")
        lineup_rows, player_rows = _parse_lineups(match, lineups)
    except Exception:
        lineup_rows, player_rows = [], []
    return stats, lineup_rows, player_rows


def update_statsbomb_open_data(
    enriched_path: str | Path = ENRICHED_STATS_PATH,
    lineups_path: str | Path = MATCH_LINEUPS_PATH,
    players_path: str | Path = PLAYER_POOL_PATH,
    max_new_matches: int = 420,
    max_workers: int = 6,
) -> dict[str, int]:
    """Collect and validate open international StatsBomb event data."""
    competitions = _get_json(f"{BASE}/competitions.json")
    selected: list[tuple[int, int]] = []
    for item in competitions:
        name = str(item.get("competition_name") or "")
        gender = str(item.get("competition_gender") or "").lower()
        season_name = str(item.get("season_name") or "")
        lower_name = name.lower()
        allowed = any(token in lower_name for token in (
            "world cup", "uefa euro", "copa america", "copa américa",
            "africa cup", "african cup", "afcon", "asian cup", "gold cup",
        ))
        if allowed and bool(item.get("competition_international", False)) and gender in {"male", "men", ""}:
            try:
                year = int(season_name[:4])
            except (TypeError, ValueError):
                year = 0
            if year >= 2010:
                selected.append((int(item["competition_id"]), int(item["season_id"])))

    existing_raw = _read_csv(enriched_path, OPTIONAL_STATS_COLUMNS)
    existing, initial_audit = clean_optional_stats_frame(existing_raw)
    existing_pairs = {
        (str(row.source).lower(), str(row.source_match_id))
        for row in existing[["source", "source_match_id"]].itertuples(index=False)
        if str(row.source_match_id).strip()
    }

    candidates: list[dict[str, Any]] = []
    for competition_id, season_id in selected:
        try:
            matches = _get_json(f"{BASE}/matches/{competition_id}/{season_id}.json")
        except Exception:
            continue
        for match in matches:
            match_id = str(match.get("match_id") or "")
            if not match_id or ("statsbomb open data", match_id) in existing_pairs:
                continue
            candidates.append(match)
            if len(candidates) >= max_new_matches:
                break
        if len(candidates) >= max_new_matches:
            break

    stats_rows: list[dict[str, Any]] = []
    lineup_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    if candidates:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = [executor.submit(_process_match, match) for match in candidates]
            for future in as_completed(futures):
                try:
                    stats, lrows, prows = future.result()
                except Exception:
                    continue
                if stats is not None:
                    stats_rows.append(stats)
                    lineup_rows.extend(lrows)
                    player_rows.extend(prows)

    combined = pd.concat([existing, pd.DataFrame(stats_rows)], ignore_index=True, sort=False)
    cleaned, final_audit = clean_optional_stats_frame(combined)
    Path(enriched_path).parent.mkdir(parents=True, exist_ok=True)
    cleaned.reindex(columns=OPTIONAL_STATS_COLUMNS).to_csv(enriched_path, index=False)

    old_lineups = _read_csv(lineups_path, ["date", "team", "player", "starter", "minutes"])
    merged_lineups = pd.concat([old_lineups, pd.DataFrame(lineup_rows)], ignore_index=True, sort=False)
    if not merged_lineups.empty:
        merged_lineups["date"] = pd.to_datetime(merged_lineups["date"], errors="coerce").dt.date.astype("string")
        merged_lineups["team"] = merged_lineups["team"].map(normalize_team_name)
        merged_lineups["player"] = merged_lineups["player"].fillna("").astype(str).str.strip()
        merged_lineups = merged_lineups[
            merged_lineups["date"].notna() & (merged_lineups["team"] != "") & (merged_lineups["player"] != "")
        ]
        merged_lineups = merged_lineups.drop_duplicates(["date", "team", "player"], keep="last")
    merged_lineups.to_csv(lineups_path, index=False)
    pool = rebuild_player_pool_from_lineups(merged_lineups, players_path)

    return {
        "matches_added": len(stats_rows),
        "lineup_rows_added": len(lineup_rows),
        "players_in_pool": len(pool),
        "invalid_rows_removed": initial_audit["invalid_rows"] + final_audit["invalid_rows"],
        "duplicate_rows_removed": initial_audit["duplicate_rows"] + final_audit["duplicate_rows"],
    }
