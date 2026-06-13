from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from football_predictor.data_loader import clean_optional_stats_frame, merge_optional_stats
from football_predictor.lineups import assess_squad
from football_predictor.statsbomb_open import _parse_events, _team_name


class DataIntegrityTests(unittest.TestCase):
    def test_statsbomb_match_team_schema(self):
        self.assertEqual(_team_name({"home_team_name": "Czechia"}, side="home"), "Czech Republic")
        self.assertEqual(_team_name({"away_team_name": "Korea Republic"}, side="away"), "South Korea")

    def test_statsbomb_event_parser_rejects_blank_teams(self):
        match = {"match_id": 1, "match_date": "2022-01-01", "home_team": {}, "away_team": {}}
        self.assertIsNone(_parse_events(match, []))

    def test_optional_cleanup_removes_broken_row_and_duplicates(self):
        frame = pd.DataFrame([
            {"date": "2022-01-01", "home_team": "", "away_team": "", "source": "StatsBomb Open Data", "home_shots": 0, "away_shots": 0},
            {"date": "2022-01-02", "home_team": "USA", "away_team": "Czechia", "source": "a", "home_xg": 1.1},
            {"date": "2022-01-02", "home_team": "United States", "away_team": "Czech Republic", "source": "b", "away_xg": 0.8},
        ])
        cleaned, audit = clean_optional_stats_frame(frame)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(audit["invalid_rows"], 1)
        self.assertEqual(audit["duplicate_rows"], 1)
        self.assertAlmostEqual(float(cleaned.iloc[0]["home_xg"]), 1.1)
        self.assertAlmostEqual(float(cleaned.iloc[0]["away_xg"]), 0.8)

    def test_duplicate_results_do_not_break_merge(self):
        results = pd.DataFrame([
            {"date": pd.Timestamp("2022-01-01"), "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0},
            {"date": pd.Timestamp("2022-01-01"), "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0},
        ])
        optional = pd.DataFrame([
            {"date": pd.Timestamp("2022-01-01"), "home_team": "A", "away_team": "B", "home_xg": 1.2, "away_xg": 0.7}
        ])
        merged = merge_optional_stats(results, optional)
        self.assertEqual(len(merged), 1)

    def test_no_lineup_means_neutral_adjustment(self):
        pool = pd.DataFrame([
            {"team": "A", "player": f"P{i}", "rating": 90 - i, "club_minutes_90d": np.nan, "national_caps": 10, "available": True}
            for i in range(15)
        ])
        assessment = assess_squad(pool, "A", selected_players=None)
        self.assertFalse(assessment.available)
        self.assertEqual(assessment.relative_strength, 1.0)


if __name__ == "__main__":
    unittest.main()
