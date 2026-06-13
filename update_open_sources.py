from __future__ import annotations

import os
from datetime import timedelta

import pandas as pd

from football_predictor.data_loader import download_results
from football_predictor.fifa_rankings import download_current_ranking
from football_predictor.statsbomb_open import update_statsbomb_open_data
from football_predictor.world_cup_live import (
    append_lineup_snapshot,
    fetch_match_lineups,
    get_world_cup_fixtures,
    row_to_fixture,
)


def main() -> None:
    download_results()
    download_current_ranking()
    api_key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    fixtures = get_world_cup_fixtures(api_key=api_key, persist=True)
    now = pd.Timestamp.now(tz="UTC")
    if api_key and not fixtures.empty:
        window = fixtures[(fixtures["kickoff_utc"] >= now - timedelta(days=1)) & (fixtures["kickoff_utc"] <= now + timedelta(days=2))]
        for _, row in window.iterrows():
            fixture = row_to_fixture(row)
            snapshot = fetch_match_lineups(fixture.source_match_id, api_key=api_key)
            append_lineup_snapshot(fixture, snapshot)
    try:
        result = update_statsbomb_open_data()
        print("StatsBomb open-data update:", result)
    except Exception as exc:
        print("StatsBomb update skipped:", exc)


if __name__ == "__main__":
    main()
