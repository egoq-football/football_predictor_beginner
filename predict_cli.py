from football_predictor.fifa_ranking import load_fifa_rankings, ranking_lookup
from football_predictor.predict import prepare_model, predict_match

print("Готовлю модель...")
df, model, states, h2h, teams = prepare_model()
fifa_lookup = ranking_lookup(load_fifa_rankings())

home = input("Первая команда, например Mexico: ").strip()
away = input("Вторая команда, например South Africa: ").strip()
neutral_text = input("Нейтральное поле? да/нет: ").strip().lower()
neutral = neutral_text in {"да", "yes", "y", "true", "1"}

result = predict_match(home, away, neutral, model, states, h2h, fifa_lookup=fifa_lookup, df=df)

print("\n=== ПРОГНОЗ ===")
print(f"{home} — {away}")
print(f"Победа {home}: {result['prob_home_win'] * 100:.1f}%")
print(f"Ничья: {result['prob_draw'] * 100:.1f}%")
print(f"Победа {away}: {result['prob_away_win'] * 100:.1f}%")
print(f"Ожидаемые голы: {home} {result['expected_goals_home']:.2f} — {result['expected_goals_away']:.2f} {away}")
print("Вероятные счета:")
for score, prob in result["top_scorelines"][:8]:
    print(f"  {score}: {prob * 100:.1f}%")
print("\nНаиболее вероятные исходы:")
for row in result["most_likely_outcomes"]:
    print(f"  {row['Категория']}: {row['Наиболее вероятный исход']} — {row['Вероятность'] * 100:.1f}%")
