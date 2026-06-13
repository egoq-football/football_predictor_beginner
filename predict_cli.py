from __future__ import annotations

import argparse
from datetime import date

from football_predictor.context import MatchContext
from football_predictor.prediction import predict_world_cup_match
from football_predictor.training import load_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогноз матча ЧМ-2026")
    parser.add_argument("home", help="Первая команда, например United States")
    parser.add_argument("away", help="Вторая команда, например Paraguay")
    parser.add_argument("--date", default=date.today().isoformat(), help="Дата YYYY-MM-DD")
    parser.add_argument("--knockout", action="store_true", help="Матч плей-офф")
    parser.add_argument("--not-neutral", action="store_true", help="Первая команда играет дома")
    args = parser.parse_args()

    _, fifa, bundle, builder = load_runtime()
    context = MatchContext(stage="knockout" if args.knockout else "group", extra_time_possible=args.knockout)
    result = predict_world_cup_match(
        bundle=bundle,
        builder=builder,
        fifa=fifa,
        home=args.home,
        away=args.away,
        match_date=args.date,
        neutral=not args.not_neutral,
        context=context,
        manual_home_strength=1.0,
        manual_away_strength=1.0,
    )
    print(f"{args.home}: {result['prob_home_win']:.1%}")
    print(f"Ничья: {result['prob_draw']:.1%}")
    print(f"{args.away}: {result['prob_away_win']:.1%}")
    print(f"Ожидаемые голы: {result['expected_goals_home']:.2f} — {result['expected_goals_away']:.2f}")
    print("Вероятные счета:")
    for score, prob in result["markets"]["top_scorelines"][:5]:
        print(f"  {score}: {prob:.1%}")


if __name__ == "__main__":
    main()
