from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import (
    DATA_START_DATE,
    ENRICHED_STATS_PATH,
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
    "USMNT": "United States",
    "Korea Republic": "South Korea",
    "Korea Rep": "South Korea",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
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

MATCH_KEYS = ["date", "home_team", "away_team"]
TEXT_OPTIONAL_COLUMNS = {"date", "home_team", "away_team", "referee", "source", "source_match_id"}


def normalize_team_name(name: Any) -> str:
    """Return a canonical team name and keep missing values truly empty.

    Converting NaN with ``str`` creates the literal team name ``"nan"``.  That
    was the reason invalid StatsBomb rows survived earlier validation.
    """
    if name is None:
        return ""
    try:
        if pd.isna(name):
            return ""
    except (TypeError, ValueError):
        pass
    value = " ".join(str(name).replace("\u00a0", " ").strip().split())
    if value.lower() in {"", "nan", "none", "null", "n/a"}:
        return ""
    return TEAM_ALIASES.get(value, value)


def download_file(url: str, path: str | Path, timeout: int = 90) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "world-cup-2026-predictor/4.3"},
    )
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def download_results(path: str | Path = RESULTS_PATH) -> Path:
    return download_file(RESULTS_URL, path)


def download_fifa_history(path: str | Path = FIFA_HISTORY_PATH) -> Path:
    return download_file(FIFA_HISTORY_URL, path)


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    truthy = {"true", "1", "yes", "y", "да"}
    return series.map(lambda value: str(value).strip().lower() in truthy)


def _most_complete_duplicates(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    work["_row_completeness"] = work.notna().sum(axis=1)
    work["_original_order"] = np.arange(len(work))
    work = (
        work.sort_values(keys + ["_row_completeness", "_original_order"])
        .drop_duplicates(subset=keys, keep="last")
        .drop(columns=["_row_completeness", "_original_order"])
        .reset_index(drop=True)
    )
    return work


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
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = _boolean_series(df["neutral"])
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df = df[(df["home_team"] != "") & (df["away_team"] != "") & (df["home_team"] != df["away_team"])]
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    cutoff = pd.Timestamp(cutoff_date) if cutoff_date is not None else pd.Timestamp(date.today())
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("UTC").tz_localize(None)
    start = pd.Timestamp(start_date)
    df = df[(df["date"] >= start) & (df["date"] <= cutoff)].copy()
    df = df[
        ~df["home_team"].isin(DEFUNCT_TEAMS)
        & ~df["away_team"].isin(DEFUNCT_TEAMS)
    ].copy()

    # Some public result feeds occasionally repeat one match.  The feature and
    # enrichment join uses date/home/away, so keep the most complete record.
    df = _most_complete_duplicates(df, MATCH_KEYS)
    df = df.sort_values(MATCH_KEYS).reset_index(drop=True)
    if df.empty:
        raise ValueError("После фильтрации не осталось завершённых матчей.")
    return df


def load_world_cup_groups(path: str | Path = GROUPS_PATH) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
        df["team"] = df["team"].map(normalize_team_name)
        return df[df["team"] != ""].copy()
    rows = [{"group": group, "team": team} for group, teams in WORLD_CUP_TEAMS.items() for team in teams]
    return pd.DataFrame(rows)


def world_cup_team_list() -> list[str]:
    return [team for group in sorted(WORLD_CUP_TEAMS) for team in WORLD_CUP_TEAMS[group]]


def _plausible_optional_values(df: pd.DataFrame) -> pd.Series:
    valid = pd.Series(True, index=df.index)
    ranges = {
        "home_xg": (0, 15), "away_xg": (0, 15),
        "home_shots": (0, 80), "away_shots": (0, 80),
        "home_shots_on_target": (0, 40), "away_shots_on_target": (0, 40),
        "home_corners": (0, 35), "away_corners": (0, 35),
        "home_yellow_cards": (0, 15), "away_yellow_cards": (0, 15),
        "home_red_cards": (0, 5), "away_red_cards": (0, 5),
        "home_possession": (0, 100), "away_possession": (0, 100),
        "home_ht_score": (0, 15), "away_ht_score": (0, 15),
    }
    for column, (low, high) in ranges.items():
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            valid &= values.isna() | values.between(low, high)

    for side in ("home", "away"):
        shots = pd.to_numeric(df.get(f"{side}_shots"), errors="coerce")
        on_target = pd.to_numeric(df.get(f"{side}_shots_on_target"), errors="coerce")
        valid &= shots.isna() | on_target.isna() | (on_target <= shots)

    # A parser failure previously created rows where both teams had no shots,
    # no xG and every event count was zero.  Do not treat those as coverage.
    shots_total = pd.to_numeric(df.get("home_shots"), errors="coerce").fillna(0) + pd.to_numeric(df.get("away_shots"), errors="coerce").fillna(0)
    xg_missing = df.get("home_xg", pd.Series(index=df.index, dtype=float)).isna() & df.get("away_xg", pd.Series(index=df.index, dtype=float)).isna()
    event_columns = [
        "home_corners", "away_corners", "home_yellow_cards", "away_yellow_cards",
        "home_red_cards", "away_red_cards", "home_ht_score", "away_ht_score",
    ]
    event_total = sum(pd.to_numeric(df.get(col), errors="coerce").fillna(0) for col in event_columns)
    source = df.get("source", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    parser_failure = source.str.contains("statsbomb") & (shots_total == 0) & xg_missing & (event_total == 0)
    valid &= ~parser_failure
    return valid


def _aggregate_optional_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    def last_valid(series: pd.Series):
        valid = series.dropna()
        if valid.empty:
            return np.nan
        # Empty source strings must not replace a useful source description.
        if series.name in {"source", "source_match_id", "referee"}:
            valid = valid.astype(str)
            valid = valid[~valid.str.strip().str.lower().isin({"", "nan", "none"})]
            if valid.empty:
                return ""
        return valid.iloc[-1]

    work = df.sort_values(MATCH_KEYS + ["source"]).copy()
    return work.groupby(MATCH_KEYS, as_index=False, sort=False, dropna=False).agg(last_valid)


def clean_optional_stats_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize, validate and deduplicate enriched match statistics."""
    raw_rows = len(df)
    work = df.copy()
    for column in OPTIONAL_STATS_COLUMNS:
        if column not in work.columns:
            work[column] = np.nan
    work = work[OPTIONAL_STATS_COLUMNS].copy()
    if work.empty:
        return work, {"raw_rows": raw_rows, "invalid_rows": 0, "duplicate_rows": 0}

    work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    work["home_team"] = work["home_team"].map(normalize_team_name)
    work["away_team"] = work["away_team"].map(normalize_team_name)
    numeric = [column for column in OPTIONAL_STATS_COLUMNS if column not in TEXT_OPTIONAL_COLUMNS]
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    for column in ["referee", "source", "source_match_id"]:
        work[column] = work[column].fillna("").astype(str).str.strip()

    structurally_valid = (
        work["date"].notna()
        & (work["home_team"] != "")
        & (work["away_team"] != "")
        & (work["home_team"] != work["away_team"])
    )
    plausible = _plausible_optional_values(work)
    valid_mask = structurally_valid & plausible
    invalid_rows = int((~valid_mask).sum())
    work = work[valid_mask].copy()

    before_dedup = len(work)
    work = _aggregate_optional_duplicates(work)
    duplicate_rows = before_dedup - len(work)
    work = work.sort_values(MATCH_KEYS).reset_index(drop=True)
    return work, {
        "raw_rows": raw_rows,
        "invalid_rows": invalid_rows,
        "duplicate_rows": duplicate_rows,
    }


def load_optional_stats(path: str | Path = ENRICHED_STATS_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=OPTIONAL_STATS_COLUMNS)
    try:
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=OPTIONAL_STATS_COLUMNS)
    cleaned, audit = clean_optional_stats_frame(raw)
    cleaned.attrs["audit"] = audit
    return cleaned


def merge_optional_stats(results: pd.DataFrame, optional: pd.DataFrame) -> pd.DataFrame:
    results_clean = _most_complete_duplicates(results.copy(), MATCH_KEYS)
    if optional.empty:
        return results_clean
    optional_clean, _ = clean_optional_stats_frame(optional)
    return results_clean.merge(
        optional_clean,
        on=MATCH_KEYS,
        how="left",
        validate="one_to_one",
    )


def load_player_pool(path: str | Path = PLAYER_POOL_PATH) -> pd.DataFrame:
    path = Path(path)
    cols = ["team", "player", "position", "rating", "club_minutes_90d", "national_caps", "available"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[cols].copy()
    df["team"] = df["team"].map(normalize_team_name)
    df["player"] = df["player"].fillna("").astype(str).str.strip()
    df["position"] = df["position"].fillna("").astype(str).str.strip()
    for col in ["rating", "club_minutes_90d", "national_caps"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["available"] = _boolean_series(df["available"].fillna(True))
    df = df[(df["team"] != "") & (df["player"] != "")]
    return df.drop_duplicates(["team", "player"], keep="last").reset_index(drop=True)


def load_match_lineups(path: str | Path = MATCH_LINEUPS_PATH) -> pd.DataFrame:
    path = Path(path)
    cols = ["date", "team", "player", "starter", "minutes"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[cols].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    df["team"] = df["team"].map(normalize_team_name)
    df["player"] = df["player"].fillna("").astype(str).str.strip()
    df["starter"] = _boolean_series(df["starter"].fillna(False))
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).clip(lower=0, upper=130)
    df = df[df["date"].notna() & (df["team"] != "") & (df["player"] != "")]
    return df.drop_duplicates(["date", "team", "player"], keep="last").sort_values("date").reset_index(drop=True)


def data_coverage(results: pd.DataFrame, optional: pd.DataFrame, player_pool: pd.DataFrame) -> dict[str, float | int | str]:
    inherited_audit = dict(getattr(optional, "attrs", {}).get("audit", {}))
    optional_clean, audit = clean_optional_stats_frame(optional)
    if inherited_audit:
        audit = {**audit, **inherited_audit}
    result_keys = results[MATCH_KEYS].drop_duplicates().copy()
    result_keys["date"] = pd.to_datetime(result_keys["date"], errors="coerce").dt.normalize()
    matched = optional_clean.merge(result_keys.assign(_matched=True), on=MATCH_KEYS, how="left")
    matched_rows = matched[matched["_matched"].fillna(False)].drop(columns="_matched")
    unmatched_rows = len(optional_clean) - len(matched_rows)

    coverage: dict[str, float | int | str] = {
        "results_matches": len(result_keys),
        "results_from": results["date"].min().date().isoformat(),
        "results_to": results["date"].max().date().isoformat(),
        "optional_raw_rows": audit["raw_rows"],
        "optional_matches": len(matched_rows),
        "optional_invalid_rows": audit["invalid_rows"],
        "optional_duplicate_rows": audit["duplicate_rows"],
        "optional_unmatched_rows": unmatched_rows,
        "players": len(player_pool),
        "optional_last_date": "",
        "sources": "",
    }
    if matched_rows.empty:
        coverage.update({
            "xg_rows": 0, "corners_rows": 0, "cards_rows": 0, "halftime_rows": 0,
            "xg_coverage": 0.0, "corners_coverage": 0.0, "cards_coverage": 0.0, "halftime_coverage": 0.0,
        })
        return coverage

    n = max(len(matched_rows), 1)
    counts = {
        "xg_rows": int(matched_rows[["home_xg", "away_xg"]].notna().all(axis=1).sum()),
        "corners_rows": int(matched_rows[["home_corners", "away_corners"]].notna().all(axis=1).sum()),
        "cards_rows": int(matched_rows[["home_yellow_cards", "away_yellow_cards"]].notna().all(axis=1).sum()),
        "halftime_rows": int(matched_rows[["home_ht_score", "away_ht_score"]].notna().all(axis=1).sum()),
    }
    coverage.update(counts)
    coverage.update({
        "xg_coverage": counts["xg_rows"] / n,
        "corners_coverage": counts["corners_rows"] / n,
        "cards_coverage": counts["cards_rows"] / n,
        "halftime_coverage": counts["halftime_rows"] / n,
        "optional_last_date": pd.to_datetime(matched_rows["date"], errors="coerce").max().date().isoformat(),
        "sources": ", ".join(sorted({
            value for value in matched_rows["source"].fillna("").astype(str)
            if value.strip() and value.strip().lower() != "nan"
        })),
    })
    return coverage
