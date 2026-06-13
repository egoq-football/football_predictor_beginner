from __future__ import annotations

import os

import pandas as pd

from football_predictor.world_cup_live import (
    append_lineup_snapshot,
    fetch_match_lineups,
    get_world_cup_fixtures,
    row_to_fixture,
)


def main() -> None:
    key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    if not key:
        print("FOOTBALL_DATA_API_KEY is not configured; live refresh skipped.")
        return
    fixtures = get_world_cup_fixtures(api_key=key, persist=True)
    if fixtures.empty:
        return
    now = pd.Timestamp.now(tz="UTC")
    # Check games from 100 minutes before kickoff until 20 minutes after kickoff.
    window = fixtures[
        (fixtures["kickoff_utc"] >= now - pd.Timedelta(minutes=20))
        & (fixtures["kickoff_utc"] <= now + pd.Timedelta(minutes=100))
    ]
    for _, row in window.iterrows():
        fixture = row_to_fixture(row)
        snapshot = fetch_match_lineups(fixture.source_match_id, api_key=key)
        append_lineup_snapshot(fixture, snapshot)
        print(fixture.label(), snapshot.message)


if __name__ == "__main__":
    main()
