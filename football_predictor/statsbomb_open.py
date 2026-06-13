from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import ENRICHED_STATS_PATH, MATCH_LINEUPS_PATH, PLAYER_POOL_PATH
from .data_loader import normalize_team_name

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
HEADERS = {"User-Agent": "world-cup-2026-predictor/4.1"}


def _get_json(url: str, timeout: int = 45) -> Any:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _team_name(obj: Any) -> str:
    if isinstance(obj, dict):
        return normalize_team_name(str(obj.get("name") or obj.get("team_name") or ""))
    return normalize_team_name(str(obj or ""))


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


def _parse_events(match: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    home = _team_name(match.get("home_team"))
    away = _team_name(match.get("away_team"))
    date = pd.to_datetime(match.get("match_date"), errors="coerce")
    stats = {
        home: defaultdict(float),
        away: defaultdict(float),
    }
    ht = {home: 0, away: 0}
    possession_events = defaultdict(int)

    for event in events:
        team = _event_team(event)
        if team not in stats:
            continue
        event_type = str((event.get("type") or {}).get("name") or "")
        possession_team = _team_name(event.get("possession_team"))
        if possession_team:
            possession_events[possession_team] += 1
        if event_type == "Shot":
            stats[team]["shots"] += 1
            shot = event.get("shot") or {}
            xg = shot.get("statsbomb_xg")
            if xg is not None:
                stats[team]["xg"] += float(xg)
            outcome = str((shot.get("outcome") or {}).get("name") or "")
            if _shot_on_target(outcome):
                stats[team]["shots_on_target"] += 1
            if outcome == "Goal" and int(event.get("period") or 0) == 1:
                ht[team] += 1
        elif event_type == "Pass":
            ptype = str(((event.get("pass") or {}).get("type") or {}).get("name") or "")
            if ptype == "Corner":
                stats[team]["corners"] += 1
        card = _card_name(event)
        if card:
            if "Yellow" in card:
                stats[team]["yellow_cards"] += 1
            if "Red" in card:
                stats[team]["red_cards"] += 1

    poss_total = sum(possession_events.values())
    home_poss = 100.0 * possession_events.get(home, 0) / poss_total if poss_total else np.nan
    away_poss = 100.0 * possession_events.get(away, 0) / poss_total if poss_total else np.nan

    return {
        "date": date.date().isoformat() if not pd.isna(date) else "",
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
        "source": "StatsBomb Open Data",
        "source_match_id": str(match.get("match_id") or ""),
    }


def _parse_lineups(match: dict[str, Any], payload: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    date = pd.to_datetime(match.get("match_date"), errors="coerce")
    date_text = date.date().isoformat() if not pd.isna(date) else ""
    lineup_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    for team_block in payload:
        team = _team_name(team_block.get("team_name") or team_block.get("team"))
        for player in team_block.get("lineup", []) or []:
            name = str(player.get("player_name") or player.get("player_nickname") or "").strip()
            if not name:
                continue
            positions = player.get("positions") or []
            starter = False
            minutes = 0.0
            position = ""
            for pos in positions:
                position = position or str(pos.get("position") or "")
                start = str(pos.get("from") or "00:00")
                end = str(pos.get("to") or "90:00")
                try:
                    sm, ss = start.split(":")[:2]
                    em, es = end.split(":")[:2]
                    start_min = float(sm) + float(ss) / 60
                    end_min = float(em) + float(es) / 60
                    minutes += max(0.0, end_min - start_min)
                    if start_min <= 0.01:
                        starter = True
                except Exception:
                    pass
            lineup_rows.append({"date": date_text, "team": team, "player": name, "starter": starter, "minutes": minutes})
            player_rows.append({"team": team, "player": name, "position": position})
    return lineup_rows, player_rows


def update_statsbomb_open_data(
    enriched_path: str | Path = ENRICHED_STATS_PATH,
    lineups_path: str | Path = MATCH_LINEUPS_PATH,
    players_path: str | Path = PLAYER_POOL_PATH,
    max_new_matches: int = 160,
) -> dict[str, int]:
    """Download only open StatsBomb World Cup data and merge it into local CSV files."""
    competitions = _get_json(f"{BASE}/competitions.json")
    selected = []
    for item in competitions:
        name = str(item.get("competition_name") or "")
        gender = str(item.get("competition_gender") or "").lower()
        season_name = str(item.get("season_name") or "")
        if "world cup" in name.lower() and gender in {"male", "men", ""}:
            try:
                year = int(season_name[:4])
            except Exception:
                year = 0
            if year >= 2010:
                selected.append((item.get("competition_id"), item.get("season_id")))

    existing = pd.read_csv(enriched_path) if Path(enriched_path).exists() else pd.DataFrame()
    existing_ids = set(existing.get("source_match_id", pd.Series(dtype=str)).astype(str)) if not existing.empty else set()
    stats_rows: list[dict[str, Any]] = []
    lineup_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []

    for competition_id, season_id in selected:
        try:
            matches = _get_json(f"{BASE}/matches/{competition_id}/{season_id}.json")
        except Exception:
            continue
        for match in matches:
            match_id = str(match.get("match_id") or "")
            if not match_id or match_id in existing_ids:
                continue
            if len(stats_rows) >= max_new_matches:
                break
            try:
                events = _get_json(f"{BASE}/events/{match_id}.json")
                stats_rows.append(_parse_events(match, events))
            except Exception:
                continue
            try:
                lineups = _get_json(f"{BASE}/lineups/{match_id}.json")
                lrows, prows = _parse_lineups(match, lineups)
                lineup_rows.extend(lrows)
                player_rows.extend(prows)
            except Exception:
                pass

    if stats_rows:
        new_stats = pd.DataFrame(stats_rows)
        merged = pd.concat([existing, new_stats], ignore_index=True)
        keys = [c for c in ["date", "home_team", "away_team"] if c in merged.columns]
        merged = merged.drop_duplicates(keys, keep="last") if keys else merged
        Path(enriched_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(enriched_path, index=False)

    if lineup_rows:
        old = pd.read_csv(lineups_path) if Path(lineups_path).exists() else pd.DataFrame()
        merged = pd.concat([old, pd.DataFrame(lineup_rows)], ignore_index=True)
        merged = merged.drop_duplicates(["date", "team", "player"], keep="last")
        merged.to_csv(lineups_path, index=False)

    if player_rows:
        appearances = pd.DataFrame(player_rows).groupby(["team", "player", "position"], dropna=False).size().reset_index(name="appearances")
        lineups = pd.DataFrame(lineup_rows)
        starts = lineups.groupby(["team", "player"])["starter"].sum().reset_index(name="starts") if not lineups.empty else pd.DataFrame(columns=["team", "player", "starts"])
        minutes = lineups.groupby(["team", "player"])["minutes"].sum().reset_index(name="minutes") if not lineups.empty else pd.DataFrame(columns=["team", "player", "minutes"])
        pool = appearances.merge(starts, on=["team", "player"], how="left").merge(minutes, on=["team", "player"], how="left")
        pool[["starts", "minutes"]] = pool[["starts", "minutes"]].fillna(0)
        # Transparent proxy, not a commercial player rating: recent national-team usage only.
        pool["rating"] = 65.0 + np.clip(pool["starts"] * 1.2 + pool["appearances"] * 0.5, 0, 18)
        pool["club_minutes_90d"] = 900.0
        pool["national_caps"] = pool["appearances"]
        pool["available"] = True
        pool = pool[["team", "player", "position", "rating", "club_minutes_90d", "national_caps", "available"]]
        old_pool = pd.read_csv(players_path) if Path(players_path).exists() else pd.DataFrame()
        merged_pool = pd.concat([old_pool, pool], ignore_index=True)
        merged_pool = merged_pool.drop_duplicates(["team", "player"], keep="last")
        merged_pool.to_csv(players_path, index=False)

    return {"matches": len(stats_rows), "lineup_rows": len(lineup_rows), "players": len(player_rows)}
