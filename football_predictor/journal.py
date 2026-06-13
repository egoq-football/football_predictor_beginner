from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import PREDICTION_LOG_PATH

LOG_COLUMNS = [
    "prediction_id", "created_at_utc", "match_date", "home_team", "away_team",
    "stage", "group_name", "group_round", "model_version",
    "prob_home", "prob_draw", "prob_away",
    "expected_home_goals", "expected_away_goals", "lineups_known",
    "actual_home_score", "actual_away_score", "status",
]


def prediction_to_row(prediction: dict[str, Any]) -> dict[str, Any]:
    context = prediction.get("context", {})
    return {
        "prediction_id": prediction["prediction_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "match_date": prediction["match_date"],
        "home_team": prediction["home"],
        "away_team": prediction["away"],
        "stage": context.get("stage", ""),
        "group_name": context.get("group_name", ""),
        "group_round": context.get("group_round", ""),
        "model_version": prediction.get("model_version", ""),
        "prob_home": prediction["prob_home_win"],
        "prob_draw": prediction["prob_draw"],
        "prob_away": prediction["prob_away_win"],
        "expected_home_goals": prediction["expected_goals_home"],
        "expected_away_goals": prediction["expected_goals_away"],
        "lineups_known": context.get("lineups_known", False),
        "actual_home_score": "",
        "actual_away_score": "",
        "status": "open",
    }


def load_local_journal(path: str | Path = PREDICTION_LOG_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=LOG_COLUMNS)
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[LOG_COLUMNS]


def append_local_prediction(prediction: dict[str, Any], path: str | Path = PREDICTION_LOG_PATH) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = load_local_journal(path)
    row = prediction_to_row(prediction)
    if row["prediction_id"] not in set(df.get("prediction_id", [])):
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def update_actual_result(
    prediction_id: str,
    home_score: int,
    away_score: int,
    path: str | Path = PREDICTION_LOG_PATH,
) -> pd.DataFrame:
    path = Path(path)
    df = load_local_journal(path)
    mask = df["prediction_id"].astype(str) == str(prediction_id)
    if not mask.any():
        raise KeyError("Прогноз с таким ID не найден.")
    df.loc[mask, "actual_home_score"] = int(home_score)
    df.loc[mask, "actual_away_score"] = int(away_score)
    df.loc[mask, "status"] = "completed"
    df.to_csv(path, index=False)
    return df


def journal_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    completed = df[df["status"].astype(str) == "completed"].copy()
    if completed.empty:
        return {"completed": 0, "accuracy": 0.0, "log_loss": 0.0, "brier": 0.0}
    for col in ["prob_home", "prob_draw", "prob_away", "actual_home_score", "actual_away_score"]:
        completed[col] = pd.to_numeric(completed[col], errors="coerce")
    completed = completed.dropna(subset=["prob_home", "prob_draw", "prob_away", "actual_home_score", "actual_away_score"])
    if completed.empty:
        return {"completed": 0, "accuracy": 0.0, "log_loss": 0.0, "brier": 0.0}
    probs = completed[["prob_away", "prob_draw", "prob_home"]].to_numpy(dtype=float)
    probs = probs / probs.sum(axis=1, keepdims=True)
    y = []
    for _, row in completed.iterrows():
        if row["actual_home_score"] > row["actual_away_score"]:
            y.append(2)
        elif row["actual_home_score"] == row["actual_away_score"]:
            y.append(1)
        else:
            y.append(0)
    y = pd.Series(y, dtype=int).to_numpy()
    pred = probs.argmax(axis=1)
    target = pd.get_dummies(pd.Series(y)).reindex(columns=[0, 1, 2], fill_value=0).to_numpy(dtype=float)
    ll = -float(pd.Series([max(probs[i, y[i]], 1e-12) for i in range(len(y))]).map(__import__("math").log).mean())
    brier = float(((probs - target) ** 2).sum(axis=1).mean())
    return {"completed": len(y), "accuracy": float((pred == y).mean()), "log_loss": ll, "brier": brier}


def auto_fill_results_from_matches(
    journal: pd.DataFrame,
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Fill open journal rows from completed match results without changing predictions."""
    if journal.empty or matches.empty:
        return journal.copy(), 0
    out = journal.copy()
    match_frame = matches.copy()
    match_frame["date_key"] = pd.to_datetime(match_frame["date"], errors="coerce").dt.date.astype(str)
    updated = 0
    for idx, row in out[out["status"].astype(str) == "open"].iterrows():
        candidates = match_frame[
            (match_frame["date_key"] == str(row["match_date"]))
            & (match_frame["home_team"].astype(str) == str(row["home_team"]))
            & (match_frame["away_team"].astype(str) == str(row["away_team"]))
        ]
        reverse = False
        if candidates.empty:
            candidates = match_frame[
                (match_frame["date_key"] == str(row["match_date"]))
                & (match_frame["home_team"].astype(str) == str(row["away_team"]))
                & (match_frame["away_team"].astype(str) == str(row["home_team"]))
            ]
            reverse = not candidates.empty
        if candidates.empty:
            continue
        match = candidates.iloc[-1]
        hs, as_ = int(match["home_score"]), int(match["away_score"])
        if reverse:
            hs, as_ = as_, hs
        out.loc[idx, "actual_home_score"] = hs
        out.loc[idx, "actual_away_score"] = as_
        out.loc[idx, "status"] = "completed"
        updated += 1
    return out, updated


class GitHubJournalStore:
    """Optional persistent CSV storage through the GitHub Contents API.

    Use a fine-grained token restricted to one repository and Contents: read/write.
    Never commit the token; configure it in Streamlit secrets.
    """

    def __init__(self, token: str, repo: str, path: str = "data/prediction_log.csv", branch: str = "main") -> None:
        self.token = token
        self.repo = repo
        self.path = path
        self.branch = branch
        self.api_url = f"https://api.github.com/repos/{repo}/contents/{path}"

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def read(self) -> tuple[pd.DataFrame, str | None]:
        response = requests.get(self.api_url, headers=self.headers, params={"ref": self.branch}, timeout=30)
        if response.status_code == 404:
            return pd.DataFrame(columns=LOG_COLUMNS), None
        response.raise_for_status()
        payload = response.json()
        content = base64.b64decode(payload["content"]).decode("utf-8")
        from io import StringIO
        try:
            df = pd.read_csv(StringIO(content))
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=LOG_COLUMNS)
        for col in LOG_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[LOG_COLUMNS], payload.get("sha")

    def write(self, df: pd.DataFrame, sha: str | None, message: str) -> None:
        content = base64.b64encode(df[LOG_COLUMNS].to_csv(index=False).encode("utf-8")).decode("ascii")
        payload: dict[str, Any] = {
            "message": message,
            "content": content,
            "branch": self.branch,
        }
        if sha:
            payload["sha"] = sha
        response = requests.put(self.api_url, headers=self.headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()

    def append(self, prediction: dict[str, Any]) -> pd.DataFrame:
        df, sha = self.read()
        row = prediction_to_row(prediction)
        if row["prediction_id"] not in set(df.get("prediction_id", [])):
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self.write(df, sha, f"Log prediction {prediction['home']} vs {prediction['away']}")
        return df

    def update_result(self, prediction_id: str, home_score: int, away_score: int) -> pd.DataFrame:
        df, sha = self.read()
        mask = df["prediction_id"].astype(str) == str(prediction_id)
        if not mask.any():
            raise KeyError("Прогноз с таким ID не найден.")
        df.loc[mask, "actual_home_score"] = int(home_score)
        df.loc[mask, "actual_away_score"] = int(away_score)
        df.loc[mask, "status"] = "completed"
        self.write(df, sha, f"Record result for prediction {prediction_id}")
        return df
