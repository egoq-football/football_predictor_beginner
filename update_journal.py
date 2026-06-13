from __future__ import annotations

from football_predictor.config import PREDICTION_LOG_PATH
from football_predictor.data_loader import load_results
from football_predictor.journal import auto_fill_results_from_matches, load_local_journal


def main() -> None:
    journal = load_local_journal(PREDICTION_LOG_PATH)
    matches = load_results()
    updated, count = auto_fill_results_from_matches(journal, matches)
    updated.to_csv(PREDICTION_LOG_PATH, index=False)
    print(f"Обновлено фактических результатов: {count}")


if __name__ == "__main__":
    main()
