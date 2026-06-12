from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import pandas as pd
import requests

FIFA_PAGE_URL = "https://inside.fifa.com/fifa-world-ranking/men"
FIFA_API_URL = "https://www.fifa.com/api/ranking-overview"
FALLBACK_JSON_URL = "https://supersubbetting.com/data/fifa-world-ranking-men.json"
DEFAULT_CACHE = Path("data/fifa_rankings.csv")

# FIFA and the match-results dataset sometimes use different English names.
FIFA_TO_RESULTS = {
    "USA": "United States",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "Kyrgyz Republic": "Kyrgyzstan",
    "China PR": "China",
    "The Gambia": "Gambia",
    "Curaçao": "Curacao",
    "Hong Kong, China": "Hong Kong",
    "Chinese Taipei": "Taiwan",
    "United Arab Emirates": "United Arab Emirates",
    "St Kitts and Nevis": "Saint Kitts and Nevis",
    "St Lucia": "Saint Lucia",
    "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "US Virgin Islands": "United States Virgin Islands",
    "São Tomé and Príncipe": "São Tomé and Príncipe",
    "Eswatini": "Eswatini",
    "Macau": "Macau",
}


@dataclass(frozen=True)
class FifaRankingInfo:
    team: str
    rank: int
    points: float
    code: str = ""
    confederation: str = ""


def _headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124 Safari/537.36"
        ),
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
    }


def _dig(mapping: dict[str, Any], keys: tuple[str, ...], default=None):
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _first_value(item: dict[str, Any], candidates: list[Any], default=None):
    for candidate in candidates:
        value = _dig(item, candidate) if isinstance(candidate, tuple) else item.get(candidate)
        if value not in (None, ""):
            return value
    return default


def _normalize_item(item: dict[str, Any]) -> dict[str, Any] | None:
    name = _first_value(
        item,
        [
            "team", "name", "country", "country_full", "countryName",
            ("rankingItem", "name"), ("team", "name"), ("country", "name"),
        ],
    )
    rank = _first_value(item, ["rank", "position", ("rankingItem", "rank")])
    points = _first_value(
        item,
        ["points", "totalPoints", "total_points", ("rankingItem", "totalPoints")],
    )
    code = _first_value(
        item,
        ["code", "countryCode", "country_abrv", "abbr", ("rankingItem", "countryCode")],
        "",
    )
    confed = _first_value(
        item,
        ["confederation", "confed", ("rankingItem", "confederation")],
        "",
    )

    if isinstance(name, dict):
        name = name.get("name") or name.get("text")
    if name is None or rank is None or points is None:
        return None
    try:
        rank_i = int(float(str(rank).replace("#", "").replace(",", "")))
        points_f = float(str(points).replace(",", ""))
    except (TypeError, ValueError):
        return None

    fifa_name = str(name).strip()
    result_name = FIFA_TO_RESULTS.get(fifa_name, fifa_name)
    return {
        "team": result_name,
        "fifa_name": fifa_name,
        "rank": rank_i,
        "points": points_f,
        "code": str(code or "").strip(),
        "confederation": str(confed or "").strip(),
    }


def _candidate_lists(payload: Any):
    """Yield lists that may contain ranking records from varied JSON schemas."""
    if isinstance(payload, list):
        yield payload
        for value in payload:
            yield from _candidate_lists(value)
    elif isinstance(payload, dict):
        for key in ("rankings", "ranking", "items", "data", "teams", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                yield value
        for value in payload.values():
            if isinstance(value, (dict, list)):
                yield from _candidate_lists(value)


def _payload_to_frame(payload: Any) -> pd.DataFrame:
    best_rows: list[dict[str, Any]] = []
    for candidate in _candidate_lists(payload):
        rows = []
        for item in candidate:
            if isinstance(item, dict):
                row = _normalize_item(item)
                if row:
                    rows.append(row)
        if len(rows) > len(best_rows):
            best_rows = rows
    if not best_rows:
        return pd.DataFrame(columns=["team", "fifa_name", "rank", "points", "code", "confederation"])
    frame = pd.DataFrame(best_rows).drop_duplicates(subset=["team"], keep="first")
    return frame.sort_values("rank").reset_index(drop=True)


def _discover_date_ids(session: requests.Session) -> list[str]:
    ids: list[str] = []
    try:
        response = session.get(FIFA_PAGE_URL, headers=_headers(), timeout=8)
        response.raise_for_status()
        # Current FIFA pages embed ranking date identifiers in script JSON.
        ids = re.findall(r'"id"\s*:\s*"(id\d+)"', response.text)
        ids += re.findall(r'dateId=(id\d+)', response.text)
    except requests.RequestException:
        pass
    # The current June 2026 release page resolves to id9054. Keep it as a
    # fallback, while discovered IDs take priority after de-duplication.
    ids += ["id9054"]
    return list(dict.fromkeys(ids))[:3]


def _fetch_official() -> pd.DataFrame:
    session = requests.Session()
    # First try the endpoint without a date, which on some FIFA deployments
    # returns the latest release.
    urls = [FIFA_API_URL]
    for date_id in _discover_date_ids(session):
        urls.append(f"{FIFA_API_URL}?locale=en&dateId={date_id}")

    for url in urls:
        try:
            response = session.get(url, headers=_headers(), timeout=8, params=None if "?" in url else {"locale": "en"})
            response.raise_for_status()
            frame = _payload_to_frame(response.json())
            if len(frame) >= 100:
                frame["source"] = "FIFA"
                return frame
        except (requests.RequestException, ValueError):
            continue
    return pd.DataFrame()


def _fetch_fallback_json() -> pd.DataFrame:
    response = requests.get(FALLBACK_JSON_URL, headers=_headers(), timeout=12)
    response.raise_for_status()
    frame = _payload_to_frame(response.json())
    if frame.empty:
        raise ValueError("Не удалось распознать резервный JSON рейтинга FIFA.")
    frame["source"] = "резервная копия рейтинга FIFA"
    return frame


def download_fifa_rankings(cache_path: str | Path = DEFAULT_CACHE) -> pd.DataFrame:
    """Download and normalize the current FIFA men's ranking.

    The official FIFA endpoint is attempted first. A public normalized mirror is
    used only when the official endpoint cannot be read by the deployment.
    """
    frame = _fetch_official()
    if frame.empty:
        frame = _fetch_fallback_json()

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def load_fifa_rankings(
    cache_path: str | Path = DEFAULT_CACHE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load FIFA rankings without making normal page loads unnecessarily slow.

    On the first ordinary launch a compact public mirror is tried first and then
    cached locally. The sidebar's explicit "Обновить рейтинг FIFA" action uses
    :func:`download_fifa_rankings`, which still tries the official FIFA endpoint
    before the mirror.
    """
    cache_path = Path(cache_path)

    if force_refresh:
        try:
            return download_fifa_rankings(cache_path)
        except Exception:
            if not cache_path.exists():
                return pd.DataFrame(columns=["team", "fifa_name", "rank", "points", "code", "confederation", "source"])

    if not cache_path.exists():
        try:
            frame = _fetch_fallback_json()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cache_path, index=False)
            return frame
        except Exception:
            return pd.DataFrame(columns=["team", "fifa_name", "rank", "points", "code", "confederation", "source"])

    frame = pd.read_csv(cache_path)
    expected = {"team", "rank", "points"}
    if not expected.issubset(frame.columns):
        return pd.DataFrame(columns=["team", "fifa_name", "rank", "points", "code", "confederation", "source"])
    return frame


def ranking_lookup(frame: pd.DataFrame) -> dict[str, FifaRankingInfo]:
    lookup: dict[str, FifaRankingInfo] = {}
    for _, row in frame.iterrows():
        try:
            lookup[str(row["team"])] = FifaRankingInfo(
                team=str(row["team"]),
                rank=int(row["rank"]),
                points=float(row["points"]),
                code=str(row.get("code", "") or ""),
                confederation=str(row.get("confederation", "") or ""),
            )
        except (TypeError, ValueError):
            continue
    return lookup


def fifa_probabilities(home: str, away: str, neutral: bool, lookup: dict[str, FifaRankingInfo]) -> dict[str, float]:
    """Convert FIFA-point difference into a three-way probability prior."""
    home_info = lookup.get(home)
    away_info = lookup.get(away)
    if not home_info or not away_info:
        return {"home_win": 0.37, "draw": 0.28, "away_win": 0.35, "points_diff": 0.0, "available": 0.0}

    home_bonus = 32.0 if not neutral else 0.0
    diff = home_info.points - away_info.points + home_bonus
    non_draw_home = 1.0 / (1.0 + 10.0 ** (-diff / 420.0))
    draw = 0.29 - min(abs(diff) / 2500.0, 0.11)
    draw = max(0.18, min(draw, 0.30))
    home_win = (1.0 - draw) * non_draw_home
    away_win = (1.0 - draw) * (1.0 - non_draw_home)
    return {
        "home_win": float(home_win),
        "draw": float(draw),
        "away_win": float(away_win),
        "points_diff": float(diff),
        "available": 1.0,
    }
