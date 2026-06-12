from __future__ import annotations

from pathlib import Path
import pandas as pd
import requests

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def download_results(save_path: str | Path = "data/results.csv") -> Path:
    """Download open international football results dataset.

    The file is cached locally so the app can work faster after the first run.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(RESULTS_URL, timeout=60)
    response.raise_for_status()
    save_path.write_bytes(response.content)
    return save_path


def load_results(path: str | Path = "data/results.csv", download_if_missing: bool = True) -> pd.DataFrame:
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
    df = df.sort_values("date").reset_index(drop=True)
    return df


def list_teams(df: pd.DataFrame) -> list[str]:
    teams = sorted(set(df["home_team"]).union(set(df["away_team"])))
    return teams
