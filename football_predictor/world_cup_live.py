from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import os

import numpy as np
import pandas as pd
import requests

from .config import FIXTURES_PATH, HOST_TEAMS, MATCH_LINEUPS_PATH
from .context import MatchContext, infer_group_motivation
from .data_loader import load_match_lineups, normalize_team_name

FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary"
WORLD_CUP_COMPETITION_CODES = ("WC", "2000")
FINISHED_STATUSES = {"FINISHED", "AWARDED"}
ACTIVE_STATUSES = {"SCHEDULED", "TIMED", "IN_PLAY", "PAUSED", "LIVE"}


@dataclass
class Fixture:
    fixture_id: str
    group_name: str
    kickoff_utc: pd.Timestamp
    home_team: str
    away_team: str
    status: str = "SCHEDULED"
    home_score: int | None = None
    away_score: int | None = None
    stage: str = "group"
    source: str = "local schedule"
    source_match_id: str = ""
    venue: str = ""
    updated_at: str = ""

    def label(self) -> str:
        kickoff = self.kickoff_utc.tz_convert("UTC") if self.kickoff_utc.tzinfo else self.kickoff_utc.tz_localize("UTC")
        date_text = kickoff.strftime("%d.%m %H:%M UTC")
        stage_text = f"Группа {self.group_name}" if self.stage == "group" and self.group_name else "Плей-офф"
        return f"{date_text} · {stage_text} · {self.home_team} — {self.away_team}"

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["kickoff_utc"] = self.kickoff_utc.isoformat()
        return out


@dataclass
class LineupSnapshot:
    available: bool
    home_players: list[str]
    away_players: list[str]
    source: str
    message: str


@dataclass
class AutomaticMatchContext:
    fixture: Fixture
    context: MatchContext
    neutral: bool
    standings: pd.DataFrame
    source_notes: list[str]


def _api_key(explicit: str | None = None) -> str:
    return (explicit or os.getenv("FOOTBALL_DATA_API_KEY") or "").strip()


def _headers(api_key: str, unfold_lineups: bool = False) -> dict[str, str]:
    headers = {"X-Auth-Token": api_key, "User-Agent": "world-cup-2026-predictor/4.1"}
    if unfold_lineups:
        headers["X-Unfold-Lineups"] = "true"
    return headers


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_group(raw: Any) -> str:
    text = str(raw or "").strip().upper()
    for prefix in ("GROUP_", "GROUP ", "GROUP-"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if len(text) == 1 and text.isalpha():
        return text
    return ""


def _normalize_stage(raw: Any, group_name: str) -> str:
    text = str(raw or "").upper()
    if group_name or "GROUP" in text:
        return "group"
    return "knockout"


def _extract_team_name(team_obj: Any) -> str:
    if isinstance(team_obj, str):
        return normalize_team_name(team_obj)
    if not isinstance(team_obj, dict):
        return ""
    return normalize_team_name(str(team_obj.get("name") or team_obj.get("shortName") or team_obj.get("tla") or ""))


def _extract_score(match: dict[str, Any], side: str) -> int | None:
    score = match.get("score") or {}
    for node_name in ("fullTime", "regularTime", "halfTime"):
        node = score.get(node_name) or {}
        if side in node and node.get(side) is not None:
            return _safe_int(node.get(side))
    return None


def _parse_api_matches(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in payload.get("matches", []) or []:
        home = _extract_team_name(item.get("homeTeam"))
        away = _extract_team_name(item.get("awayTeam"))
        if not home or not away:
            continue
        kickoff = pd.to_datetime(item.get("utcDate"), utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue
        group_name = _normalize_group(item.get("group"))
        stage = _normalize_stage(item.get("stage"), group_name)
        rows.append({
            "fixture_id": str(item.get("id") or f"{home}-{away}-{kickoff.isoformat()}"),
            "group_name": group_name,
            "kickoff_utc": kickoff,
            "home_team": home,
            "away_team": away,
            "status": str(item.get("status") or "SCHEDULED").upper(),
            "home_score": _extract_score(item, "home"),
            "away_score": _extract_score(item, "away"),
            "stage": stage,
            "source": "football-data.org",
            "source_match_id": str(item.get("id") or ""),
            "venue": str(item.get("venue") or ""),
            "updated_at": pd.Timestamp.utcnow().isoformat(),
        })
    return pd.DataFrame(rows)


def fetch_football_data_matches(api_key: str | None = None) -> pd.DataFrame:
    key = _api_key(api_key)
    if not key:
        return pd.DataFrame()
    last_error: Exception | None = None
    for code in WORLD_CUP_COMPETITION_CODES:
        url = f"{FOOTBALL_DATA_BASE}/competitions/{code}/matches"
        try:
            response = requests.get(url, params={"season": 2026}, headers=_headers(key, unfold_lineups=True), timeout=25)
            response.raise_for_status()
            frame = _parse_api_matches(response.json())
            if not frame.empty:
                return frame
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
    if last_error:
        frame = pd.DataFrame()
        frame.attrs["error"] = str(last_error)
        return frame
    return pd.DataFrame()


def load_local_fixtures(path: str | Path = FIXTURES_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True, errors="coerce")
    frame["home_team"] = frame["home_team"].map(normalize_team_name)
    frame["away_team"] = frame["away_team"].map(normalize_team_name)
    frame["status"] = frame.get("status", "SCHEDULED").fillna("SCHEDULED").astype(str).str.upper()
    frame["home_score"] = pd.to_numeric(frame.get("home_score"), errors="coerce")
    frame["away_score"] = pd.to_numeric(frame.get("away_score"), errors="coerce")
    for col, default in {
        "fixture_id": "", "group_name": "", "stage": "group", "source": "local schedule",
        "source_match_id": "", "venue": "", "updated_at": ""
    }.items():
        if col not in frame.columns:
            frame[col] = default
    return frame.dropna(subset=["kickoff_utc", "home_team", "away_team"])


def _fixture_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["home_team"].astype(str) + "|" + frame["away_team"].astype(str) + "|" +
        frame["kickoff_utc"].dt.strftime("%Y-%m-%d")
    )


def get_world_cup_fixtures(api_key: str | None = None, persist: bool = False) -> pd.DataFrame:
    local = load_local_fixtures()
    remote = fetch_football_data_matches(api_key)
    if remote.empty:
        return local.sort_values("kickoff_utc").reset_index(drop=True)
    if local.empty:
        merged = remote
    else:
        local = local.copy()
        remote = remote.copy()
        local["_key"] = _fixture_key(local)
        remote["_key"] = _fixture_key(remote)
        merged = pd.concat([local[~local["_key"].isin(remote["_key"])], remote], ignore_index=True)
        merged = merged.drop(columns=["_key"], errors="ignore")
    merged = merged.sort_values("kickoff_utc").reset_index(drop=True)
    if persist:
        out = merged.copy()
        out["kickoff_utc"] = out["kickoff_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        Path(FIXTURES_PATH).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(FIXTURES_PATH, index=False)
    return merged


def row_to_fixture(row: pd.Series | dict[str, Any]) -> Fixture:
    data = dict(row)
    kickoff = pd.to_datetime(data.get("kickoff_utc"), utc=True)
    return Fixture(
        fixture_id=str(data.get("fixture_id") or ""),
        group_name=str(data.get("group_name") or ""),
        kickoff_utc=kickoff,
        home_team=normalize_team_name(str(data.get("home_team") or "")),
        away_team=normalize_team_name(str(data.get("away_team") or "")),
        status=str(data.get("status") or "SCHEDULED").upper(),
        home_score=_safe_int(data.get("home_score")),
        away_score=_safe_int(data.get("away_score")),
        stage=str(data.get("stage") or "group"),
        source=str(data.get("source") or "local schedule"),
        source_match_id=str(data.get("source_match_id") or ""),
        venue=str(data.get("venue") or ""),
        updated_at=str(data.get("updated_at") or ""),
    )


def selectable_fixtures(frame: pd.DataFrame, include_recent_hours: int = 4) -> pd.DataFrame:
    if frame.empty:
        return frame
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(hours=include_recent_hours)
    active = frame[(frame["kickoff_utc"] >= cutoff) & ~frame["status"].isin(FINISHED_STATUSES)].copy()
    return active if not active.empty else frame.copy()


def standings_before_fixture(fixtures: pd.DataFrame, fixture: Fixture) -> pd.DataFrame:
    teams = sorted(set(fixtures.loc[fixtures["group_name"] == fixture.group_name, "home_team"]) |
                   set(fixtures.loc[fixtures["group_name"] == fixture.group_name, "away_team"]))
    table = {team: {"team": team, "played": 0, "points": 0, "gf": 0, "ga": 0, "gd": 0} for team in teams}
    if fixture.stage != "group" or not fixture.group_name:
        return pd.DataFrame(table.values())
    prior = fixtures[
        (fixtures["group_name"] == fixture.group_name) &
        (fixtures["kickoff_utc"] < fixture.kickoff_utc) &
        fixtures["status"].isin(FINISHED_STATUSES) &
        fixtures["home_score"].notna() & fixtures["away_score"].notna()
    ]
    for _, row in prior.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs = int(row["home_score"])
        aw = int(row["away_score"])
        for team in (home, away):
            if team not in table:
                table[team] = {"team": team, "played": 0, "points": 0, "gf": 0, "ga": 0, "gd": 0}
            table[team]["played"] += 1
        table[home]["gf"] += hs; table[home]["ga"] += aw
        table[away]["gf"] += aw; table[away]["ga"] += hs
        if hs > aw:
            table[home]["points"] += 3
        elif hs < aw:
            table[away]["points"] += 3
        else:
            table[home]["points"] += 1; table[away]["points"] += 1
    for row in table.values():
        row["gd"] = row["gf"] - row["ga"]
    return pd.DataFrame(table.values()).sort_values(["points", "gd", "gf"], ascending=[False, False, False]).reset_index(drop=True)


def _team_table_value(table: pd.DataFrame, team: str, column: str, default: int = 0) -> int:
    if table.empty:
        return default
    row = table[table["team"] == team]
    if row.empty:
        return default
    return int(row.iloc[0][column])


def _previous_match_date(matches: pd.DataFrame, team: str, kickoff: pd.Timestamp) -> pd.Timestamp | None:
    if matches.empty:
        return None
    date_col = pd.to_datetime(matches["date"], utc=True, errors="coerce")
    mask = ((matches["home_team"] == team) | (matches["away_team"] == team)) & (date_col < kickoff)
    subset = date_col[mask].dropna()
    return subset.max() if not subset.empty else None


def _group_round(fixtures: pd.DataFrame, fixture: Fixture) -> int:
    if fixture.stage != "group":
        return 3
    prior = fixtures[(fixtures["group_name"] == fixture.group_name) & (fixtures["kickoff_utc"] < fixture.kickoff_utc)]
    home_count = int(((prior["home_team"] == fixture.home_team) | (prior["away_team"] == fixture.home_team)).sum())
    away_count = int(((prior["home_team"] == fixture.away_team) | (prior["away_team"] == fixture.away_team)).sum())
    return int(np.clip(max(home_count, away_count) + 1, 1, 3))


def automatic_match_context(fixtures: pd.DataFrame, fixture: Fixture, historical_matches: pd.DataFrame, lineups_known: bool) -> AutomaticMatchContext:
    table = standings_before_fixture(fixtures, fixture)
    home_points = _team_table_value(table, fixture.home_team, "points")
    away_points = _team_table_value(table, fixture.away_team, "points")
    home_gd = _team_table_value(table, fixture.home_team, "gd")
    away_gd = _team_table_value(table, fixture.away_team, "gd")
    group_round = _group_round(fixtures, fixture)

    home_prev = _previous_match_date(historical_matches, fixture.home_team, fixture.kickoff_utc)
    away_prev = _previous_match_date(historical_matches, fixture.away_team, fixture.kickoff_utc)
    home_rest = 5 if home_prev is None else max(1, min(14, int((fixture.kickoff_utc - home_prev).total_seconds() // 86400)))
    away_rest = 5 if away_prev is None else max(1, min(14, int((fixture.kickoff_utc - away_prev).total_seconds() // 86400)))

    context = MatchContext(
        stage=fixture.stage,
        group_name=fixture.group_name,
        group_round=group_round,
        home_points=home_points,
        away_points=away_points,
        home_goal_difference=home_gd,
        away_goal_difference=away_gd,
        home_days_rest=home_rest,
        away_days_rest=away_rest,
        lineups_known=lineups_known,
        extra_time_possible=fixture.stage == "knockout",
    )
    context = infer_group_motivation(context)
    # Conservative automatic rotation risk: only after qualification looks likely.
    if fixture.stage == "group" and group_round == 3:
        context.home_rotation_risk = 0.35 if home_points >= 6 else (0.18 if home_points >= 4 else 0.05)
        context.away_rotation_risk = 0.35 if away_points >= 6 else (0.18 if away_points >= 4 else 0.05)
    if home_rest <= 3:
        context.home_rotation_risk = max(context.home_rotation_risk, 0.15)
    if away_rest <= 3:
        context.away_rotation_risk = max(context.away_rotation_risk, 0.15)

    neutral = not (fixture.home_team in HOST_TEAMS or fixture.away_team in HOST_TEAMS)
    notes = [
        f"Дата и стадия взяты из календаря ЧМ-2026 ({fixture.source}).",
        "Очки и разница мячей рассчитаны автоматически по завершённым матчам группы до начала выбранной встречи.",
        "Номер тура, дни отдыха, необходимость победы, достаточность ничьей и риск ротации рассчитаны автоматически.",
    ]
    return AutomaticMatchContext(fixture, context, neutral, table, notes)


def _lineup_names(team_obj: dict[str, Any]) -> list[str]:
    candidates = team_obj.get("lineup") or team_obj.get("startingEleven") or team_obj.get("startingXI") or []
    names: list[str] = []
    for item in candidates:
        if isinstance(item, str):
            names.append(item)
            continue
        if not isinstance(item, dict):
            continue
        player = item.get("player") if isinstance(item.get("player"), dict) else item
        name = player.get("name") or player.get("shortName")
        if name:
            names.append(str(name))
    return names[:11]


def _espn_team_name(value: Any) -> str:
    if isinstance(value, str):
        return normalize_team_name(value)
    if not isinstance(value, dict):
        return ""
    return normalize_team_name(str(
        value.get("displayName")
        or value.get("shortDisplayName")
        or value.get("name")
        or value.get("location")
        or value.get("abbreviation")
        or ""
    ))


def _espn_event_id(fixture: Fixture) -> str:
    """Find the ESPN event id by date and exact team pair.

    ESPN is used only as a no-key fallback when the primary provider does not
    expose the official starters. A date/team match avoids hard-coding event ids.
    """
    try:
        response = requests.get(
            ESPN_SCOREBOARD_URL,
            params={"dates": fixture.kickoff_utc.strftime("%Y%m%d"), "limit": 100},
            headers={"User-Agent": "world-cup-2026-predictor/4.4"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return ""

    wanted = {normalize_team_name(fixture.home_team), normalize_team_name(fixture.away_team)}
    for event in payload.get("events", []) or []:
        competitions = event.get("competitions", []) or []
        if not competitions:
            continue
        competitors = competitions[0].get("competitors", []) or []
        names = {_espn_team_name(item.get("team") or {}) for item in competitors}
        names.discard("")
        if names == wanted:
            return str(event.get("id") or competitions[0].get("id") or "")
    return ""


def _espn_starters_from_payload(payload: dict[str, Any], team_name: str) -> list[str]:
    """Extract only explicitly marked starters from an ESPN summary payload."""
    wanted = normalize_team_name(team_name)
    blocks: list[dict[str, Any]] = []
    blocks.extend(payload.get("rosters", []) or [])

    boxscore = payload.get("boxscore") or {}
    blocks.extend(boxscore.get("players", []) or [])

    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_team = _espn_team_name(block.get("team") or block.get("teamInfo") or {})
        if block_team != wanted:
            continue
        entries = (
            block.get("roster")
            or block.get("athletes")
            or block.get("lineup")
            or block.get("players")
            or []
        )
        names: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            starter_flag = entry.get("starter")
            if starter_flag is None:
                starter_flag = entry.get("isStarter")
            if starter_flag is None:
                starter_flag = entry.get("starting")
            # Do not treat a general squad list as an official starting XI.
            if starter_flag is not True:
                continue
            athlete = entry.get("athlete") if isinstance(entry.get("athlete"), dict) else entry
            name = athlete.get("displayName") or athlete.get("fullName") or athlete.get("name") or athlete.get("shortName")
            if name:
                names.append(str(name).strip())
        # Preserve order and avoid duplicated player records.
        names = list(dict.fromkeys(name for name in names if name))
        if len(names) >= 7:
            return names[:11]
    return []


def _fetch_espn_lineups(fixture: Fixture) -> LineupSnapshot:
    event_id = _espn_event_id(fixture)
    if not event_id:
        return LineupSnapshot(False, [], [], "ESPN", "ESPN не нашёл выбранный матч по дате и командам.")
    try:
        response = requests.get(
            ESPN_SUMMARY_URL,
            params={"event": event_id},
            headers={"User-Agent": "world-cup-2026-predictor/4.4"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        home_players = _espn_starters_from_payload(payload, fixture.home_team)
        away_players = _espn_starters_from_payload(payload, fixture.away_team)
        if len(home_players) >= 7 and len(away_players) >= 7:
            return LineupSnapshot(
                True,
                home_players,
                away_players,
                "ESPN",
                "Официальные стартовые составы получены автоматически из резервного открытого источника ESPN.",
            )
        return LineupSnapshot(False, [], [], "ESPN", "ESPN пока не пометил стартовые составы как официальные.")
    except Exception as exc:  # pragma: no cover - network dependent
        return LineupSnapshot(False, [], [], "ESPN", f"Резервный источник составов недоступен ({exc}).")


def fetch_match_lineups(
    source_match_id: str,
    api_key: str | None = None,
    fixture: Fixture | None = None,
) -> LineupSnapshot:
    """Load official starters, first from football-data.org, then from ESPN.

    The fallback is intentionally no-key and only accepts players explicitly
    marked as starters. Predicted or probable lineups are never used.
    """
    key = _api_key(api_key)
    primary_message = ""

    if key and source_match_id:
        try:
            response = requests.get(
                f"{FOOTBALL_DATA_BASE}/matches/{source_match_id}",
                headers=_headers(key, unfold_lineups=True),
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            home_players = _lineup_names(payload.get("homeTeam") or {})
            away_players = _lineup_names(payload.get("awayTeam") or {})
            if len(home_players) >= 7 and len(away_players) >= 7:
                return LineupSnapshot(
                    True,
                    home_players,
                    away_players,
                    "football-data.org",
                    "Официальные стартовые составы получены автоматически через football-data.org.",
                )
            primary_message = "football-data.org пока не вернул полный стартовый состав."
        except Exception as exc:  # pragma: no cover - network dependent
            primary_message = f"football-data.org не вернул составы ({exc})."
    elif not key:
        primary_message = "Ключ football-data.org не настроен."
    else:
        primary_message = "У матча отсутствует идентификатор football-data.org."

    if fixture is not None:
        fallback = _fetch_espn_lineups(fixture)
        if fallback.available:
            return fallback
        return LineupSnapshot(
            False,
            [],
            [],
            fallback.source,
            f"{primary_message} {fallback.message} Прогноз выполнен без поправки на состав.",
        )

    return LineupSnapshot(
        False,
        [],
        [],
        "football-data.org",
        f"{primary_message} Прогноз выполнен без поправки на состав.",
    )

def append_lineup_snapshot(fixture: Fixture, snapshot: LineupSnapshot, path: str | Path = MATCH_LINEUPS_PATH) -> None:
    if not snapshot.available:
        return
    rows = []
    date_value = fixture.kickoff_utc.date().isoformat()
    for team, players in ((fixture.home_team, snapshot.home_players), (fixture.away_team, snapshot.away_players)):
        for player in players:
            rows.append({"date": date_value, "team": team, "player": player, "starter": True, "minutes": 0})
    if not rows:
        return
    current = load_match_lineups(path)
    addition = pd.DataFrame(rows)
    addition["date"] = pd.to_datetime(addition["date"])
    merged = pd.concat([current, addition], ignore_index=True)
    merged = merged.drop_duplicates(["date", "team", "player"], keep="last").sort_values(["date", "team", "player"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
