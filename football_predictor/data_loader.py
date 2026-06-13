from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .config import (
    DATA_START_DATE,
    ENRICHED_STATS_PATH,
    FIFA_CURRENT_PATH,
    FIFA_HISTORY_PATH,
    GROUPS_PATH,
    MATCH_LINEUPS_PATH,
    OPTIONAL_STATS_COLUMNS,
    PLAYER_POOL_PATH,
    RESULTS_PATH,
    WORLD_CUP_TEAMS,
)

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
FIFA_HISTORY_URL = (
    "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/refs/heads/master/"
    "ranking_fifa_historical.csv"
)

TEAM_ALIASES = {
    "USA": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "IR Iran": "Iran",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Curacao": "Curaçao",
}

DEFUNCT_TEAMS = {
    "Bohemia", "British Guyana", "Burma", "Ceylon", "Czechoslovakia",
    "East Germany", "French Somaliland", "Irish Free State", "Manchukuo",
    "Netherlands Antilles", "New Hebrides", "North Vietnam", "North Yemen",
    "Rhodesia", "Saarland", "South Vietnam", "South Yemen", "Soviet Union",
    "United Arab Republic", "Upper Volta", "West Germany", "Yugoslavia", "Zaire",
}


def normalize_team_name(name: str) -> str:
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def download_file(url: str, path: str | Path, timeout: int = 90) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "world-cup-2026-predictor/4.0"},
    )
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def download_results(path: str | Path = RESULTS_PATH) -> Path:
    return download_file(RESULTS_URL, path)


def download_fifa_history(path: str | Path = FIFA_HISTORY_PATH) -> Path:
    return download_file(FIFA_HISTORY_URL, path)


def load_results(
    path: str | Path = RESULTS_PATH,
    start_date: str = DATA_START_DATE,
    cutoff_date: str | pd.Timestamp | None = None,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(path)
        download_results(path)

    df = pd.read_csv(path)
    required = {
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В results.csv отсутствуют колонки: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)

    cutoff = pd.Timestamp(cutoff_date) if cutoff_date is not None else pd.Timestamp(date.today())
    start = pd.Timestamp(start_date)
    df = df[(df["date"] >= start) & (df["date"] <= cutoff)].copy()

    df = df[
        ~df["home_team"].isin(DEFUNCT_TEAMS)
        & ~df["away_team"].isin(DEFUNCT_TEAMS)
    ].copy()

    # Удаляем повторные записи одного и того же матча.
    # Если дубликаты отличаются заполненностью, оставляем наиболее полную строку.
    match_keys = ["date", "home_team", "away_team"]
    df["_row_completeness"] = df.notna().sum(axis=1)

    df = (
        df.sort_values(match_keys + ["_row_completeness"])
        .drop_duplicates(subset=match_keys, keep="last")
        .drop(columns="_row_completeness")
        .sort_values(match_keys)
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("После фильтрации не осталось завершённых матчей.")
    return df


def load_world_cup_groups(path: str | Path = GROUPS_PATH) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
        df["team"] = df["team"].map(normalize_team_name)
        return df
    rows = [{"group": group, "team": team} for group, teams in WORLD_CUP_TEAMS.items() for team in teams]
    return pd.DataFrame(rows)


def world_cup_team_list() -> list[str]:
    return [team for group in sorted(WORLD_CUP_TEAMS) for team in WORLD_CUP_TEAMS[group]]


def load_optional_stats(path: str | Path = ENRICHED_STATS_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=OPTIONAL_STATS_COLUMNS)
    df = pd.read_csv(path)
    for col in OPTIONAL_STATS_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[OPTIONAL_STATS_COLUMNS].copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)
    numeric = [
        c for c in OPTIONAL_STATS_COLUMNS
        if c not in {"date", "home_team", "away_team", "referee", "source", "source_match_id"}
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["referee", "source", "source_match_id"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df.dropna(subset=["date", "home_team", "away_team"]).sort_values("date")


def merge_optional_stats(
    results: pd.DataFrame,
    optional: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["date", "home_team", "away_team"]

    # Дополнительная защита от повторов в основном наборе.
    results_clean = (
        results.copy()
        .drop_duplicates(subset=keys, keep="last")
        .reset_index(drop=True)
    )

    if optional.empty:
        return results_clean

    optional_clean = optional.copy().replace("", np.nan)

    # Если один матч пришёл несколькими строками из разных источников,
    # собираем по каждому столбцу последнее непустое значение.
    def last_valid(series: pd.Series):
        valid = series.dropna()
        if valid.empty:
            return np.nan
        return valid.iloc[-1]

    optional_clean = (
        optional_clean
        .groupby(keys, as_index=False, sort=False)
        .agg(last_valid)
    )

    return results_clean.merge(
        optional_clean,
        on=keys,
        how="left",
        validate="one_to_one",
    )


def load_player_pool(path: str | Path = PLAYER_POOL_PATH) -> pd.DataFrame:
    path = Path(path)
    cols = ["team", "player", "position", "rating", "club_minutes_90d", "national_caps", "available"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[cols].copy()
    df["team"] = df["team"].map(normalize_team_name)
    for col in ["rating", "club_minutes_90d", "national_caps"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["available"] = df["available"].fillna(True).astype(bool)
    return df


def load_match_lineups(path: str | Path = MATCH_LINEUPS_PATH) -> pd.DataFrame:
    path = Path(path)
    cols = ["date", "team", "player", "starter", "minutes"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[cols].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["team"] = df["team"].map(normalize_team_name)
    df["starter"] = df["starter"].fillna(False).astype(bool)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
    return df.dropna(subset=["date", "team", "player"])


def data_coverage(results: pd.DataFrame, optional: pd.DataFrame, player_pool: pd.DataFrame) -> dict[str, float | int | str]:
    coverage: dict[str, float | int | str] = {
        "results_matches": len(results),
        "results_from": results["date"].min().date().isoformat(),
        "results_to": results["date"].max().date().isoformat(),
        "optional_matches": len(optional),
        "players": len(player_pool),
        "optional_last_date": "",
        "sources": "",
    }
    if optional.empty:
        coverage.update({
            "xg_rows": 0, "corners_rows": 0, "cards_rows": 0, "halftime_rows": 0,
            "xg_coverage": 0.0, "corners_coverage": 0.0, "cards_coverage": 0.0, "halftime_coverage": 0.0,
        })
        return coverage

    n = max(len(optional), 1)
    counts = {
        "xg_rows": int(optional[["home_xg", "away_xg"]].notna().all(axis=1).sum()),
        "corners_rows": int(optional[["home_corners", "away_corners"]].notna().all(axis=1).sum()),
        "cards_rows": int(optional[["home_yellow_cards", "away_yellow_cards"]].notna().all(axis=1).sum()),
        "halftime_rows": int(optional[["home_ht_score", "away_ht_score"]].notna().all(axis=1).sum()),
    }
    coverage.update(counts)
    coverage.update({
        "xg_coverage": counts["xg_rows"] / n,
        "corners_coverage": counts["corners_rows"] / n,
        "cards_coverage": counts["cards_rows"] / n,
        "halftime_coverage": counts["halftime_rows"] / n,
        "optional_last_date": pd.to_datetime(optional["date"], errors="coerce").max().date().isoformat()
        if optional["date"].notna().any() else "",
        "sources": ", ".join(sorted({x for x in optional.get("source", pd.Series(dtype=str)).astype(str) if x and x != "nan"})),
    })
    return coverage
