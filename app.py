from __future__ import annotations

import pandas as pd
import streamlit as st

from football_predictor.data_loader import download_results
from football_predictor.predict import prepare_model, predict_match
from football_predictor.stats import form_chart_data

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")

st.title("⚽ Football Predictor")
st.caption("Версия 2: больше статистики, усиленная модель, вероятности счета, тоталов и формы команд.")

with st.sidebar:
    st.header("Данные и обучение")
    if st.button("1. Скачать/обновить данные"):
        with st.spinner("Скачиваю открытый датасет матчей сборных..."):
            path = download_results("data/results.csv")
        st.success(f"Данные сохранены: {path}")

    force_retrain = st.checkbox("Переобучить модель", value=False)
    st.info("Если после обновления сайта видишь ошибку модели, включи «Переобучить модель» один раз.")

try:
    with st.spinner("Готовлю модель. Первый запуск может занять 1–3 минуты..."):
        df, model, states, h2h, teams = prepare_model(force_retrain=force_retrain)
except Exception as exc:
    st.error("Не получилось подготовить модель.")
    st.write("Сначала нажми слева кнопку **Скачать/обновить данные**. Если ошибка повторяется, пришли мне её текст.")
    st.exception(exc)
    st.stop()

st.subheader("Выбор матча")
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    home = st.selectbox("Первая команда", teams, index=teams.index("Mexico") if "Mexico" in teams else 0)
with col2:
    away = st.selectbox("Вторая команда", teams, index=teams.index("South Africa") if "South Africa" in teams else 1)
with col3:
    neutral = st.checkbox("Нейтральное поле", value=True)

if st.button("Сделать прогноз", type="primary", use_container_width=True):
    try:
        result = predict_match(home, away, neutral, model, states, h2h, df=df)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.divider()
    st.subheader(f"Прогноз: {home} — {away}")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Победа {home}", f"{result['prob_home_win'] * 100:.1f}%")
    c2.metric("Ничья", f"{result['prob_draw'] * 100:.1f}%")
    c3.metric(f"Победа {away}", f"{result['prob_away_win'] * 100:.1f}%")

    st.progress(result["prob_home_win"], text=f"Шанс победы {home}: {result['prob_home_win'] * 100:.1f}%")
    st.progress(result["prob_draw"], text=f"Шанс ничьей: {result['prob_draw'] * 100:.1f}%")
    st.progress(result["prob_away_win"], text=f"Шанс победы {away}: {result['prob_away_win'] * 100:.1f}%")

    xg1, xg2, total_xg = st.columns(3)
    xg1.metric(f"Ожидаемые голы {home}", f"{result['expected_goals_home']:.2f}")
    xg2.metric(f"Ожидаемые голы {away}", f"{result['expected_goals_away']:.2f}")
    xg3_value = result["expected_goals_home"] + result["expected_goals_away"]
    total_xg.metric("Ожидаемый тотал", f"{xg3_value:.2f}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Итог",
        "Статистика команд",
        "Счета и тоталы",
        "Форма",
        "Очные встречи",
        "Техника модели",
    ])

    with tab1:
        st.write("### Почему модель так решила")
        for item in result["explanations"]:
            st.write(f"- {item}")

        st.write("### Вероятности по разным слоям модели")
        layer_df = pd.DataFrame([
            {"Модель": "Машинное обучение", home: result["ml_probs"]["home_win"], "Ничья": result["ml_probs"]["draw"], away: result["ml_probs"]["away_win"]},
            {"Модель": "Нечеткие правила", home: result["fuzzy_probs"]["home_win"], "Ничья": result["fuzzy_probs"]["draw"], away: result["fuzzy_probs"]["away_win"]},
            {"Модель": "Пуассон по голам", home: result["poisson_probs"]["home_win"], "Ничья": result["poisson_probs"]["draw"], away: result["poisson_probs"]["away_win"]},
            {"Модель": "Итог", home: result["prob_home_win"], "Ничья": result["prob_draw"], away: result["prob_away_win"]},
        ])
        st.dataframe(layer_df.style.format({home: "{:.1%}", "Ничья": "{:.1%}", away: "{:.1%}"}), use_container_width=True, hide_index=True)

    with tab2:
        st.write("### Сравнение команд")
        summary_df = pd.DataFrame(result["team_summary"])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.write("### Ключевые числовые признаки")
        key_features = {
            "Разница Elo с учетом поля": result["features"]["elo_diff"],
            "Разница очков за 5 матчей": result["features"]["form_points_diff_5"],
            "Разница голов за 5 матчей": result["features"]["goal_diff_form_diff_5"],
            "Разница атаки за 10 матчей": result["features"]["attack_diff_10"],
            "Разница обороны за 10 матчей": result["features"]["defense_diff_10"],
            "Очные встречи: разница голов": result["features"]["h2h_goal_diff"],
        }
        st.dataframe(pd.DataFrame([key_features]).T.rename(columns={0: "Значение"}), use_container_width=True)

    with tab3:
        st.write("### Наиболее вероятные счета")
        score_df = pd.DataFrame(result["top_scorelines"], columns=["Счет", "Вероятность"])
        st.dataframe(score_df.style.format({"Вероятность": "{:.1%}"}), use_container_width=True, hide_index=True)

        st.write("### Дополнительные вероятности")
        m = result["markets"]
        market_rows = [
            {"Показатель": "Тотал больше 1.5", "Вероятность": m["over_1_5"]},
            {"Показатель": "Тотал меньше 1.5", "Вероятность": m["under_1_5"]},
            {"Показатель": "Тотал больше 2.5", "Вероятность": m["over_2_5"]},
            {"Показатель": "Тотал меньше 2.5", "Вероятность": m["under_2_5"]},
            {"Показатель": "Тотал больше 3.5", "Вероятность": m["over_3_5"]},
            {"Показатель": "Обе забьют — да", "Вероятность": m["btts_yes"]},
            {"Показатель": f"{home} не пропустит", "Вероятность": m["home_clean_sheet"]},
            {"Показатель": f"{away} не пропустит", "Вероятность": m["away_clean_sheet"]},
            {"Показатель": f"{home} или ничья", "Вероятность": m["double_chance_home_or_draw"]},
            {"Показатель": f"{away} или ничья", "Вероятность": m["double_chance_away_or_draw"]},
        ]
        market_df = pd.DataFrame(market_rows)
        st.dataframe(market_df.style.format({"Вероятность": "{:.1%}"}), use_container_width=True, hide_index=True)

    with tab4:
        st.write(f"### Последние матчи: {home}")
        st.dataframe(result["recent_home"], use_container_width=True, hide_index=True)
        st.write(f"### Последние матчи: {away}")
        st.dataframe(result["recent_away"], use_container_width=True, hide_index=True)

        chart = form_chart_data(df, home, away, n=10)
        if not chart.empty:
            st.write("### Накопленные очки за последние 10 матчей")
            pivot = chart.pivot(index="Матч №", columns="Команда", values="Накопленные очки")
            st.line_chart(pivot)

    with tab5:
        h2h_table = result["h2h_table"]
        if h2h_table.empty:
            st.info("В базе не найдено очных встреч этих команд.")
        else:
            st.write("### Последние очные встречи")
            st.dataframe(h2h_table, use_container_width=True, hide_index=True)

    with tab6:
        with st.expander("Показать технические признаки"):
            st.json(result["features"])

        st.write("### Качество модели на историческом тесте")
        metrics = getattr(model, "metrics_", {})
        if metrics:
            metrics_df = pd.DataFrame([
                {"Метрика": "Всего матчей в таблице обучения", "Значение": int(metrics["matches_total"])},
                {"Метрика": "Матчей в тесте", "Значение": int(metrics["test_matches"])},
                {"Метрика": "Тестовая точность", "Значение": f"{metrics['accuracy'] * 100:.1f}%"},
                {"Метрика": "Базовая точность: всегда победа первой команды", "Значение": f"{metrics['baseline_home_accuracy'] * 100:.1f}%"},
                {"Метрика": "Log loss", "Значение": f"{metrics['log_loss']:.3f}"},
                {"Метрика": "Точность класса: победа второй", "Значение": f"{metrics['away_win_accuracy'] * 100:.1f}%"},
                {"Метрика": "Точность класса: ничья", "Значение": f"{metrics['draw_accuracy'] * 100:.1f}%"},
                {"Метрика": "Точность класса: победа первой", "Значение": f"{metrics['home_win_accuracy'] * 100:.1f}%"},
            ])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            st.write("Модель была загружена из файла. Для пересчета метрик поставь галочку «Переобучить модель».")
