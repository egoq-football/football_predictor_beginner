from __future__ import annotations

import os
from datetime import timedelta

import pandas as pd

from football_predictor.data_loader import download_results
from football_predictor.fifa_rankings import download_current_ranking
from football_predictor.football_data_enrichment import update_football_data_history
from football_predictor.statsbomb_open import update_statsbomb_open_data
from football_predictor.world_cup_live import (
    append_lineup_snapshot,
    fetch_match_lineups,
    get_world_cup_fixtures,
    row_to_fixture,
)


def _safe_step(label: str, callback):
    try:
        result = callback()
        print(f"{label}: {result}")
        return result
    except Exception as exc:
        print(f"{label} skipped: {exc}")
        return None


def main() -> None:
    # Every source is independent. One temporary network/API failure must not stop
    # the remaining collectors or erase the latest good local snapshot.
    _safe_step("Results update", download_results)
    _safe_step("FIFA update", download_current_ranking)

    api_key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    fixtures = _safe_step(
        "World Cup fixtures",
        lambda: get_world_cup_fixtures(api_key=api_key, persist=True),
    )

    if api_key:
        _safe_step(
            "football-data.org historical enrichment",
            lambda: update_football_data_history(api_key, max_details=70),
        )

    if isinstance(fixtures, pd.DataFrame) and not fixtures.empty:
        now = pd.Timestamp.now(tz="UTC")
        window = fixtures[
            (fixtures["kickoff_utc"] >= now - timedelta(hours=6))
            & (fixtures["kickoff_utc"] <= now + timedelta(hours=3))
        ]
        for _, row in window.iterrows():
            fixture = row_to_fixture(row)
            snapshot = fetch_match_lineups(fixture.source_match_id, api_key=api_key, fixture=fixture)
            append_lineup_snapshot(fixture, snapshot)

    _safe_step(
        "StatsBomb open-data update",
        lambda: update_statsbomb_open_data(max_new_matches=420),
    )


if __name__ == "__main__":
    main()
