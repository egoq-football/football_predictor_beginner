from __future__ import annotations

from football_predictor.backtest import backtest_world_cups
from football_predictor.config import WORLD_CUP_BACKTEST_PATH
from football_predictor.training import build_dataset


def main() -> None:
    matches, table, fifa = build_dataset(download=False)
    report = backtest_world_cups(table, matches, fifa)
    WORLD_CUP_BACKTEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(WORLD_CUP_BACKTEST_PATH, index=False)
    print(report.to_string(index=False))
    print(f"Сохранено: {WORLD_CUP_BACKTEST_PATH}")


if __name__ == "__main__":
    main()
