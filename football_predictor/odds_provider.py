from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

from .data_loader import normalize_team_name


BASE_URL = "https://api.the-odds-api.com/v4"


@dataclass
class MarketQuote:
    market_key: str
    fair_probability: float
    decimal_odds: float
    bookmakers: int


@dataclass
class MarketSnapshot:
    available: bool = False
    source: str = ""
    message: str = "Рыночные данные не подключены; используется только математический фильтр неочевидности."
    quotes: dict[str, MarketQuote] = field(default_factory=dict)

    def quote(self, key: str) -> MarketQuote | None:
        return self.quotes.get(key)


def _team(value: str) -> str:
    return normalize_team_name(str(value or "").replace(" USA", " United States").strip())


def _match_score(name_a: str, name_b: str) -> float:
    a = _team(name_a).lower()
    b = _team(name_b).lower()
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.8
    aset = set(a.replace("-", " ").split())
    bset = set(b.replace("-", " ").split())
    return len(aset & bset) / max(len(aset | bset), 1)


def _discover_world_cup_sport(api_key: str, timeout: int = 15) -> str | None:
    response = requests.get(f"{BASE_URL}/sports", params={"apiKey": api_key}, timeout=timeout)
    response.raise_for_status()
    sports = response.json()
    candidates = []
    for item in sports:
        key = str(item.get("key") or "")
        title = str(item.get("title") or "")
        group = str(item.get("group") or "")
        text = f"{key} {title} {group}".lower()
        if "soccer" in text and ("world cup" in text or "fifa" in text):
            candidates.append((0 if not item.get("has_outrights") else 1, key))
    return sorted(candidates)[0][1] if candidates else None


def _find_event(events: list[dict[str, Any]], home: str, away: str, kickoff: pd.Timestamp) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any] | None] = (-1.0, None)
    kickoff = pd.Timestamp(kickoff)
    if kickoff.tzinfo is None:
        kickoff = kickoff.tz_localize("UTC")
    for event in events:
        h = str(event.get("home_team") or "")
        a = str(event.get("away_team") or "")
        direct = 0.5 * (_match_score(h, home) + _match_score(a, away))
        reverse = 0.5 * (_match_score(h, away) + _match_score(a, home))
        team_score = max(direct, reverse)
        start = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(start):
            time_score = 0.0
        else:
            hours = abs((start - kickoff).total_seconds()) / 3600.0
            time_score = max(0.0, 1.0 - hours / 36.0)
        score = 0.80 * team_score + 0.20 * time_score
        if score > best[0]:
            best = (score, event)
    return best[1] if best[0] >= 0.62 else None


def _consensus_h2h(event: dict[str, Any], home: str, away: str) -> dict[str, MarketQuote]:
    raw: dict[str, list[tuple[float, float]]] = {"home_win": [], "draw": [], "away_win": []}
    for bookmaker in event.get("bookmakers", []) or []:
        market = next((m for m in bookmaker.get("markets", []) or [] if m.get("key") == "h2h"), None)
        if not market:
            continue
        prices: dict[str, float] = {}
        for outcome in market.get("outcomes", []) or []:
            name = str(outcome.get("name") or "")
            try:
                price = float(outcome.get("price"))
            except (TypeError, ValueError):
                continue
            if price <= 1.0:
                continue
            if name.lower() == "draw":
                key = "draw"
            elif _match_score(name, home) >= _match_score(name, away):
                key = "home_win"
            else:
                key = "away_win"
            prices[key] = price
        if len(prices) != 3:
            continue
        implied = {key: 1.0 / price for key, price in prices.items()}
        total = sum(implied.values())
        for key in raw:
            raw[key].append((implied[key] / total, prices[key]))
    out: dict[str, MarketQuote] = {}
    for key, values in raw.items():
        if values:
            out[key] = MarketQuote(
                key,
                fair_probability=float(np.median([x[0] for x in values])),
                decimal_odds=float(np.median([x[1] for x in values])),
                bookmakers=len(values),
            )
    return out


def _consensus_totals(event: dict[str, Any]) -> dict[str, MarketQuote]:
    values: dict[tuple[float, str], list[tuple[float, float]]] = {}
    for bookmaker in event.get("bookmakers", []) or []:
        for market in bookmaker.get("markets", []) or []:
            if market.get("key") not in {"totals", "alternate_totals"}:
                continue
            grouped: dict[float, dict[str, float]] = {}
            for outcome in market.get("outcomes", []) or []:
                try:
                    point = float(outcome.get("point"))
                    price = float(outcome.get("price"))
                except (TypeError, ValueError):
                    continue
                side = str(outcome.get("name") or "").lower()
                if side not in {"over", "under"} or price <= 1.0:
                    continue
                grouped.setdefault(point, {})[side] = price
            for point, sides in grouped.items():
                if set(sides) != {"over", "under"}:
                    continue
                imp_over = 1.0 / sides["over"]
                imp_under = 1.0 / sides["under"]
                total = imp_over + imp_under
                values.setdefault((point, "over"), []).append((imp_over / total, sides["over"]))
                values.setdefault((point, "under"), []).append((imp_under / total, sides["under"]))
    out: dict[str, MarketQuote] = {}
    for (point, side), rows in values.items():
        key = f"{side}_{str(point).replace('.', '_')}"
        out[key] = MarketQuote(
            key,
            fair_probability=float(np.median([x[0] for x in rows])),
            decimal_odds=float(np.median([x[1] for x in rows])),
            bookmakers=len(rows),
        )
    return out


def fetch_market_snapshot(
    home: str,
    away: str,
    kickoff: str | pd.Timestamp,
    api_key: str | None,
    regions: str = "eu,uk",
    timeout: int = 20,
) -> MarketSnapshot:
    key = str(api_key or "").strip()
    if not key:
        return MarketSnapshot()
    try:
        sport = _discover_world_cup_sport(key, timeout=timeout)
        if not sport:
            return MarketSnapshot(False, "The Odds API", "В API не найден активный рынок чемпионата мира; применяется математический фильтр.")
        response = requests.get(
            f"{BASE_URL}/sports/{sport}/odds",
            params={"apiKey": key, "regions": regions, "markets": "h2h,totals", "oddsFormat": "decimal"},
            timeout=timeout,
        )
        response.raise_for_status()
        events = response.json()
        event = _find_event(events, home, away, pd.Timestamp(kickoff))
        if not event:
            return MarketSnapshot(False, "The Odds API", "Рыночная линия выбранного матча пока не найдена; применяется математический фильтр.")
        quotes = _consensus_h2h(event, home, away)
        quotes.update(_consensus_totals(event))
        if not quotes:
            return MarketSnapshot(False, "The Odds API", "Рыночные данные получены, но подходящие рынки отсутствуют.")
        return MarketSnapshot(True, "The Odds API", "Рыночная линия использована только для внутренней проверки неочевидности; коэффициенты на сайте не показываются.", quotes)
    except Exception as exc:  # pragma: no cover - network dependent
        return MarketSnapshot(False, "The Odds API", f"Рыночная линия временно недоступна ({exc}); применяется математический фильтр.")
