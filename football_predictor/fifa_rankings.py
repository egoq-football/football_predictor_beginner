from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import date as date_cls
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import FIFA_CURRENT_PATH, FIFA_HISTORY_PATH
from .data_loader import normalize_team_name

FIFA_API_URL = "https://www.fifa.com/api/ranking-overview"
FIFA_PAGE_URL = "https://inside.fifa.com/fifa-world-ranking/men"
FALLBACK_JSON_URL = "https://supersubbetting.com/data/fifa-world-ranking-men.json"


@dataclass(frozen=True)
class RankingInfo:
    team: str
    date: pd.Timestamp
    points: float
    rank: int | None = None
    source: str = ""


def _headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
    }


def _candidate_lists(payload: Any):
    if isinstance(payload, list):
        yield payload
        for value in payload:
            yield from _candidate_lists(value)
    elif isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, (list, dict)):
                yield from _candidate_lists(value)


def _value(item: dict[str, Any], names: list[str]):
    for name in names:
        value = item.get(name)
        if value not in (None, ""):
            return value
    return None


def _payload_frame(payload: Any, source: str) -> pd.DataFrame:
    best: list[dict[str, Any]] = []
    for candidate in _candidate_lists(payload):
        normalized = []
        for item in candidate:
            if not isinstance(item, dict):
                continue
            nested = item.get("rankingItem") if isinstance(item.get("rankingItem"), dict) else {}
            team_obj = item.get("team") if isinstance(item.get("team"), dict) else {}
            name = _value(item, ["team", "name", "country", "country_full", "countryName"])
            if isinstance(name, dict):
                name = name.get("name") or name.get("text")
            name = name or nested.get("name") or team_obj.get("name")
            points = _value(item, ["points", "totalPoints", "total_points"]) or nested.get("totalPoints")
            rank = _value(item, ["rank", "position"]) or nested.get("rank")
            if name is None or points is None:
                continue
            try:
                normalized.append({
                    "team": normalize_team_name(str(name)),
                    "points": float(str(points).replace(",", "")),
                    "rank": int(float(str(rank).replace("#", ""))) if rank is not None else np.nan,
                    "date": pd.Timestamp.utcnow().date().isoformat(),
                    "source": source,
                })
            except (TypeError, ValueError):
                continue
        if len(normalized) > len(best):
            best = normalized
    return pd.DataFrame(best).drop_duplicates("team") if best else pd.DataFrame()


def download_current_ranking(path: str | Path = FIFA_CURRENT_PATH) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # FIFA's public endpoint often expects dateId as the number of days since
    # 1985-01-01. Try the latest days and the official 11 June 2026 release.
    epoch = date_cls(1985, 1, 1)
    today_id = (date_cls.today() - epoch).days
    june_2026_id = (date_cls(2026, 6, 11) - epoch).days
    date_ids = [today_id - offset for offset in range(0, 8)] + [june_2026_id]
    attempts = [(f"{FIFA_API_URL}?locale=en&dateId=id{value}", "FIFA current") for value in dict.fromkeys(date_ids)]
    attempts += [
        (FIFA_API_URL, "FIFA current"),
        (f"{FIFA_API_URL}?locale=en", "FIFA current"),
        (FALLBACK_JSON_URL, "FIFA ranking mirror"),
    ]
    for url, source in attempts:
        try:
            response = requests.get(url, headers=_headers(), timeout=20)
            response.raise_for_status()
            frame = _payload_frame(response.json(), source)
            if len(frame) >= 100:
                frame.to_csv(path, index=False)
                return frame
        except Exception:
            continue
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=["team", "points", "rank", "date", "source"])



class FifaRankingHistory:
    """Historical FIFA points lookup using the latest release before each match."""

    def __init__(
        self,
        history_path: str | Path = FIFA_HISTORY_PATH,
        current_path: str | Path = FIFA_CURRENT_PATH,
        refresh_current: bool = False,
    ) -> None:
        history_path = Path(history_path)
        if history_path.exists():
            history = pd.read_csv(history_path)
        else:
            history = pd.DataFrame(columns=["team", "total_points", "date"])
        if not history.empty:
            history = history.rename(columns={"total_points": "points"})
            history["team"] = history["team"].map(normalize_team_name)
            history["date"] = pd.to_datetime(history["date"], errors="coerce")
            history["points"] = pd.to_numeric(history["points"], errors="coerce")
            history = history.dropna(subset=["team", "date", "points"])
            history["rank"] = history.groupby("date")["points"].rank(method="min", ascending=False)
            history["source"] = "historical FIFA"
            history = history[["team", "date", "points", "rank", "source"]]

        current_path = Path(current_path)
        if refresh_current or not current_path.exists():
            current = download_current_ranking(current_path)
        else:
            try:
                current = pd.read_csv(current_path)
            except Exception:
                current = pd.DataFrame()
        if not current.empty:
            current["team"] = current["team"].map(normalize_team_name)
            current["date"] = pd.to_datetime(current["date"], errors="coerce")
            current["points"] = pd.to_numeric(current["points"], errors="coerce")
            current["rank"] = pd.to_numeric(current.get("rank"), errors="coerce")
            current["source"] = current.get("source", "current FIFA")
            current = current[["team", "date", "points", "rank", "source"]].dropna(subset=["team", "date", "points"])

        frames = [frame for frame in (history, current) if frame is not None and not frame.empty]
        self.frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["team", "date", "points", "rank", "source"])
        self.frame = self.frame.sort_values(["team", "date"]).drop_duplicates(["team", "date"], keep="last")
        self._groups = {team: group.reset_index(drop=True) for team, group in self.frame.groupby("team")}

    def lookup(self, team: str, on_date: str | pd.Timestamp) -> RankingInfo | None:
        team = normalize_team_name(team)
        group = self._groups.get(team)
        if group is None or group.empty:
            return None
        date = pd.Timestamp(on_date)
        if date.tzinfo is not None:
            date = date.tz_convert("UTC").tz_localize(None)
        eligible = group[group["date"] <= date]
        if eligible.empty:
            return None
        row = eligible.iloc[-1]
        rank = None if pd.isna(row["rank"]) else int(row["rank"])
        return RankingInfo(
            team=team,
            date=pd.Timestamp(row["date"]),
            points=float(row["points"]),
            rank=rank,
            source=str(row["source"]),
        )

    def current_lookup(self, team: str) -> RankingInfo | None:
        team = normalize_team_name(team)
        group = self._groups.get(team)
        if group is None or group.empty:
            return None
        row = group.iloc[-1]
        rank = None if pd.isna(row["rank"]) else int(row["rank"])
        return RankingInfo(team, pd.Timestamp(row["date"]), float(row["points"]), rank, str(row["source"]))

    def points(self, team: str, on_date: str | pd.Timestamp, default: float = 1400.0) -> float:
        info = self.lookup(team, on_date)
        return float(info.points) if info else float(default)

    def rank(self, team: str, on_date: str | pd.Timestamp) -> int | None:
        info = self.lookup(team, on_date)
        return info.rank if info else None


def refresh_current_ranking_if_stale(
    path: str | Path = FIFA_CURRENT_PATH,
    max_age_hours: int = 12,
) -> pd.DataFrame:
    """Refresh the public FIFA snapshot automatically, keeping the last good file on failure."""
    path = Path(path)
    stale = True
    if path.exists() and path.stat().st_size > 0:
        age_seconds = pd.Timestamp.now(tz="UTC").timestamp() - path.stat().st_mtime
        stale = age_seconds > max_age_hours * 3600
        if not stale:
            try:
                return pd.read_csv(path)
            except Exception:
                stale = True
    if stale:
        return download_current_ranking(path)
    return pd.DataFrame()
