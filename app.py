from __future__ import annotations

import streamlit as st

from football_predictor.data_loader import download_results
from football_predictor.predict import prepare_model, predict_match

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="centered")

st.title("⚽ Прогноз футбольных матчей")
st.write("Простая первая версия: открытые данные сборных, Elo, форма, машинное обучение, нечеткие правила и ожидаемые голы.")

with st.sidebar:
    st.header("Данные и обучение")
    if st.button("1. Скачать/обновить данные"):
        with st.spinner("Скачиваю открытый датасет матчей сборных..."):
            path = download_results("data/results.csv")
        st.success(f"Данные сохранены: {path}")

    force_retrain = st.checkbox("Переобучить модель", value=False)

try:
    with st.spinner("Готовлю модель. Первый запуск может занять 1–3 минуты..."):
        df, model, states, h2h, teams = prepare_model(force_retrain=force_retrain)
except Exception as exc:
    st.error("Не получилось подготовить модель.")
    st.write("Сначала нажми слева кнопку **Скачать/обновить данные**. Если ошибка повторяется, пришли мне её текст.")
    st.exception(exc)
    st.stop()

st.subheader("Выбор матча")
col1, col2 = st.columns(2)
with col1:
    home = st.selectbox("Первая команда", teams, index=teams.index("Mexico") if "Mexico" in teams else 0)
with col2:
    away = st.selectbox("Вторая команда", teams, index=teams.index("South Africa") if "South Africa" in teams else 1)

neutral = st.checkbox("Нейтральное поле", value=True)

if st.button("Сделать прогноз", type="primary"):
    try:
        result = predict_match(home, away, neutral, model, states, h2h)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.subheader(f"Прогноз: {home} — {away}")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Победа {home}", f"{result['prob_home_win'] * 100:.1f}%")
    c2.metric("Ничья", f"{result['prob_draw'] * 100:.1f}%")
    c3.metric(f"Победа {away}", f"{result['prob_away_win'] * 100:.1f}%")

    st.write("**Ожидаемые голы:**")
    st.write(f"{home}: **{result['expected_goals_home']:.2f}**, {away}: **{result['expected_goals_away']:.2f}**")

    st.write("**Наиболее вероятные счета:**")
    score_text = ", ".join([f"{score} — {prob * 100:.1f}%" for score, prob in result["top_scorelines"][:5]])
    st.write(score_text)

    st.write("**Почему модель так решила:**")
    for item in result["explanations"]:
        st.write(f"- {item}")

    with st.expander("Показать технические признаки"):
        st.json(result["features"])

with st.expander("Качество модели на историческом тесте"):
    metrics = getattr(model, "metrics_", {})
    if metrics:
        st.write(f"Всего матчей в таблице обучения: {int(metrics['matches_total'])}")
        st.write(f"Тестовая точность: {metrics['accuracy'] * 100:.1f}%")
        st.write(f"Log loss: {metrics['log_loss']:.3f}")
    else:
        st.write("Модель была загружена из файла. Для пересчета метрик поставь галочку «Переобучить модель».")
