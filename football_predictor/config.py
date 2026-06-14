from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_PATH = DATA_DIR / "results.csv"
FIFA_HISTORY_PATH = DATA_DIR / "fifa_rankings_history.csv"
FIFA_CURRENT_PATH = DATA_DIR / "fifa_rankings_current.csv"
ENRICHED_STATS_PATH = DATA_DIR / "enriched_match_stats.csv"
PLAYER_POOL_PATH = DATA_DIR / "player_pool.csv"
MATCH_LINEUPS_PATH = DATA_DIR / "match_lineups.csv"
PREDICTION_LOG_PATH = DATA_DIR / "prediction_log.csv"
GROUPS_PATH = DATA_DIR / "world_cup_2026_groups.csv"
FIXTURES_PATH = DATA_DIR / "world_cup_2026_fixtures.csv"
MODEL_BUNDLE_PATH = MODELS_DIR / "world_cup_2026_bundle.joblib"
MODEL_META_PATH = MODELS_DIR / "model_metadata.json"
BACKTEST_PATH = MODELS_DIR / "backtest_metrics.csv"
WORLD_CUP_BACKTEST_PATH = MODELS_DIR / "world_cup_backtest.csv"
DATA_AUDIT_PATH = MODELS_DIR / "data_audit.json"

DATA_START_DATE = "2010-01-01"
MODEL_VERSION = "4.6.0-world-cup-2026"

WORLD_CUP_TEAMS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Bosnia and Herzegovina", "Qatar"],
    "C": ["Brazil", "Scotland", "Morocco", "Haiti"],
    "D": ["United States", "Australia", "Paraguay", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Norway", "Senegal", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

HOST_TEAMS = {"Canada", "Mexico", "United States"}

# Tournament importance used in Elo updates and features. These are not final
# ensemble weights: they describe the sporting importance of individual matches.
TOURNAMENT_IMPORTANCE = {
    "FIFA World Cup": 1.00,
    "FIFA World Cup qualification": 0.90,
    "UEFA Euro": 0.90,
    "UEFA Euro qualification": 0.78,
    "Copa América": 0.90,
    "African Cup of Nations": 0.88,
    "AFC Asian Cup": 0.86,
    "CONCACAF Gold Cup": 0.82,
    "UEFA Nations League": 0.72,
    "CONCACAF Nations League": 0.70,
    "Friendly": 0.42,
}

OPTIONAL_STATS_COLUMNS = [
    "date", "home_team", "away_team",
    "home_xg", "away_xg",
    "home_shots", "away_shots",
    "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners",
    "home_yellow_cards", "away_yellow_cards",
    "home_red_cards", "away_red_cards",
    "home_possession", "away_possession",
    "home_ppda", "away_ppda",
    "home_ht_score", "away_ht_score",
    "referee", "source", "source_match_id",
]
