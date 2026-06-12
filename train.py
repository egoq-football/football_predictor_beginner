from football_predictor.data_loader import download_results, load_results
from football_predictor.features import build_training_table
from football_predictor.model import save_model, train_with_chronological_test

print("Скачиваю данные...")
download_results("data/results.csv")

print("Загружаю данные...")
df = load_results("data/results.csv")

print("Создаю признаки для обучения...")
table = build_training_table(df, min_year=2010)

print("Обучаю модель...")
model = train_with_chronological_test(table)
save_model(model, "models/football_predictor.joblib")

print("Готово!")
print("Метрики:")
for key, value in model.metrics_.items():
    print(f"{key}: {value}")
