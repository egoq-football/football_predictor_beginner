from __future__ import annotations

from pathlib import Path
from datetime import date
import pandas as pd
import requests

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
DATA_START_DATE = pd.Timestamp("2010-01-01")

# Historical national teams that may exist in the full archive but must never
# appear in the current-team selector. The 2010 date filter already excludes
# them in normal operation; this list is an additional safety net.
DEFUNCT_TEAMS = {
    "Bohemia",
    "British Guyana",
    "Burma",
    "Ceylon",
    "Czechoslovakia",
    "East Germany",
    "French Somaliland",
    "Irish Free State",
    "Manchukuo",
    "Netherlands Antilles",
    "New Hebrides",
    "North Vietnam",
    "North Yemen",
    "Rhodesia",
    "Saarland",
    "South Vietnam",
    "South Yemen",
    "Soviet Union",
    "United Arab Republic",
    "Upper Volta",
    "West Germany",
    "Yugoslavia",
    "Zaire",
}


def download_results(save_path: str | Path = "data/results.csv") -> Path:
    """Download the open international-results dataset."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        RESULTS_URL,
        timeout=60,
        headers={"User-Agent": "football-predictor/3.0"},
    )
    response.raise_for_status()
    save_path.write_bytes(response.content)
    return save_path


def load_results(
    path: str | Path = "data/results.csv",
    download_if_missing: bool = True,
    start_date: str | pd.Timestamp = DATA_START_DATE,
) -> pd.DataFrame:
    """Load completed matches beginning with 1 January 2010.

    Future fixtures and rows without a final score are excluded. Because the list
    of teams is built only from this filtered frame, historical/defunct national
    teams no longer appear in the selectors.
    """
    path = Path(path)
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"Не найден файл данных: {path}")
        download_results(path)

    df = pd.read_csv(path)
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В файле данных нет нужных колонок: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)

    start = pd.Timestamp(start_date)
    today = pd.Timestamp(date.today())
    df = df.loc[(df["date"] >= start) & (df["date"] <= today)]
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "После фильтра с 01.01.2010 не осталось завершённых матчей. "
            "Нажми «Скачать/обновить данные»."
        )
    return df


def list_teams(df: pd.DataFrame, min_matches: int = 1) -> list[str]:
    """Return teams with at least ``min_matches`` completed matches since 2010."""
    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    teams = counts[counts >= min_matches].index.astype(str)
    return sorted(team for team in teams if team not in DEFUNCT_TEAMS)
