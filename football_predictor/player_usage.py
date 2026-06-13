from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PLAYER_POOL_PATH
from .data_loader import normalize_team_name


def rebuild_player_pool_from_lineups(
    lineups: pd.DataFrame,
    path: str | Path = PLAYER_POOL_PATH,
) -> pd.DataFrame:
    """Build a transparent national-team usage index.

    Open sources used by this project do not provide reliable current club
    minutes for every international player.  Therefore no fabricated club
    minutes are written.  ``rating`` is a 50–100 usage index derived only from
    national-team appearances, starts and recorded minutes.
    """
    columns = ["team", "player", "position", "rating", "club_minutes_90d", "national_caps", "available"]
    if lineups is None or lineups.empty:
        existing_path = Path(path)
        if existing_path.exists() and existing_path.stat().st_size > 0:
            try:
                return pd.read_csv(existing_path)
            except (pd.errors.EmptyDataError, OSError):
                pass
        return pd.DataFrame(columns=columns)

    history = lineups.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["team"] = history["team"].map(normalize_team_name)
    history["player"] = history["player"].fillna("").astype(str).str.strip()
    history["starter"] = history["starter"].fillna(False).astype(bool)
    history["minutes"] = pd.to_numeric(history["minutes"], errors="coerce").fillna(0).clip(0, 130)
    history = history[history["date"].notna() & (history["team"] != "") & (history["player"] != "")]
    history = history.drop_duplicates(["date", "team", "player"], keep="last")

    latest = history["date"].max()
    age_days = (latest - history["date"]).dt.days.clip(lower=0)
    history["recency"] = np.exp(-age_days / 540.0)
    history["appearance_value"] = history["recency"] * (
        0.75 + 1.25 * history["starter"].astype(float) + np.clip(history["minutes"], 0, 120) / 90.0
    )

    usage = history.groupby(["team", "player"], as_index=False).agg(
        usage_value=("appearance_value", "sum"),
        appearances=("date", "count"),
        starts=("starter", "sum"),
        national_minutes=("minutes", "sum"),
    )

    rank_pct = usage.groupby("team")["usage_value"].rank(method="average", pct=True)
    usage["rating"] = 50.0 + 50.0 * rank_pct

    usage["position"] = ""
    usage["club_minutes_90d"] = np.nan
    usage["national_caps"] = usage["appearances"]
    usage["available"] = True
    pool = usage[columns].sort_values(["team", "rating"], ascending=[True, False]).reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(path, index=False)
    return pool
