from __future__ import annotations

import argparse

from football_predictor.training import train_and_maybe_promote


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение кандидата World Cup 2026 Predictor")
    parser.add_argument("--download", action="store_true", help="Сначала обновить открытый архив матчей и рейтинг FIFA")
    parser.add_argument("--force", action="store_true", help="Принять кандидата независимо от сравнения")
    parser.add_argument("--min-improvement", type=float, default=0.002, help="Минимальное улучшение Log Loss")
    args = parser.parse_args()
    bundle, promoted, reason = train_and_maybe_promote(
        download=args.download,
        force=args.force,
        min_improvement=args.min_improvement,
    )
    print("Модель принята:" if promoted else "Текущая модель сохранена:", reason)
    print(f"Версия: {bundle.version}; матчей: {bundle.training_matches}; данные по: {bundle.train_end_date}")
    for row in bundle.metrics:
        print(row)


if __name__ == "__main__":
    main()
