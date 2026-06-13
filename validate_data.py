from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from football_predictor.config import ENRICHED_STATS_PATH, MATCH_LINEUPS_PATH, MODELS_DIR, PLAYER_POOL_PATH
from football_predictor.data_loader import (
    MATCH_KEYS,
    clean_optional_stats_frame,
    data_coverage,
    load_match_lineups,
    load_results,
)
from football_predictor.player_usage import rebuild_player_pool_from_lineups


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка и очистка футбольных данных")
    parser.add_argument("--fix", action="store_true", help="Перезаписать очищенные файлы")
    args = parser.parse_args()

    results = load_results()
    optional_raw = _read(Path(ENRICHED_STATS_PATH))
    optional_clean, audit = clean_optional_stats_frame(optional_raw)
    lineups = load_match_lineups()

    if args.fix:
        Path(ENRICHED_STATS_PATH).parent.mkdir(parents=True, exist_ok=True)
        optional_clean.to_csv(ENRICHED_STATS_PATH, index=False)
        lineups.to_csv(MATCH_LINEUPS_PATH, index=False)
        player_pool = rebuild_player_pool_from_lineups(lineups, PLAYER_POOL_PATH)
    else:
        player_pool = _read(Path(PLAYER_POOL_PATH))

    coverage = data_coverage(results, optional_clean, player_pool)
    report = {
        "results_rows": len(results),
        "results_duplicate_keys": int(results.duplicated(MATCH_KEYS).sum()),
        "optional_raw_rows": audit["raw_rows"],
        "optional_valid_rows": len(optional_clean),
        "optional_invalid_rows_removed": audit["invalid_rows"],
        "optional_duplicate_rows_removed": audit["duplicate_rows"],
        "optional_matched_rows": int(coverage["optional_matches"]),
        "optional_unmatched_rows": int(coverage["optional_unmatched_rows"]),
        "halftime_rows": int(coverage["halftime_rows"]),
        "xg_rows": int(coverage["xg_rows"]),
        "corners_rows": int(coverage["corners_rows"]),
        "cards_rows": int(coverage["cards_rows"]),
        "lineup_rows": len(lineups),
        "players": len(player_pool),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if report["results_duplicate_keys"]:
        raise SystemExit("После очистки в results остались повторные ключи матчей")


if __name__ == "__main__":
    main()
