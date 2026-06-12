from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from football_predictor.data_loader import download_results, load_results, list_teams
from football_predictor.fifa_ranking import download_fifa_rankings, load_fifa_rankings, ranking_lookup
from football_predictor.predict import prepare_model, predict_match
from football_predictor.stats import form_chart_data

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stButton"] > button {min-height: 2.45rem;}
    .small-note {font-size: 0.90rem; color: #666;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ Football Predictor")
st.caption(
    "Версия 3.1: матчи с 2010 года; 50% прогноза дают последние 5 матчей и рейтинг FIFA, "
    "ещё 50% — математические модели."
)


@st.cache_data(show_spinner=False)
def get_data_for_team_list():
    return load_results("data/results.csv")


@st.cache_resource(show_spinner=False)
def get_prepared_model(force_retrain: bool = False):
    return prepare_model(force_retrain=force_retrain)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def get_fifa_data(force_refresh: bool = False):
    return load_fifa_rankings(force_refresh=force_refresh)


with st.sidebar:
    st.header("Данные")
    if st.button("Обновить матчи"):
        with st.spinner("Скачиваю открытый датасет матчей сборных..."):
            path = download_results("data/results.csv")
            model_path = Path("models/football_predictor.joblib")
            if model_path.exists():
                model_path.unlink()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success(f"Данные обновлены: {path}")
        st.rerun()

    if st.button("Обновить рейтинг FIFA"):
        with st.spinner("Загружаю рейтинг FIFA..."):
            try:
                download_fifa_rankings("data/fifa_rankings.csv")
            except Exception as exc:
                st.error(f"Не получилось обновить рейтинг FIFA: {exc}")
            else:
                st.cache_data.clear()
                st.success("Рейтинг FIFA обновлён.")
                st.rerun()

    force_retrain = st.checkbox("Переобучить модель", value=False)
    if st.button("Очистить кэш"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.info(
        "Все результаты, форма и расчётный Elo берутся только из завершённых матчей "
        "с 1 января 2010 года."
    )

try:
    df_for_teams = get_data_for_team_list()
    teams = list_teams(df_for_teams)
except Exception as exc:
    st.error("Не получилось загрузить список команд.")
    st.write("Нажми слева **Обновить матчи**. Если ошибка повторяется, пришли её текст.")
    st.exception(exc)
    st.stop()

fifa_df = get_fifa_data(False)
fifa_lookup = ranking_lookup(fifa_df)

max_date = pd.to_datetime(df_for_teams["date"]).max().date().isoformat()
st.caption(
    f"В расчёте: {len(df_for_teams)} завершённых матчей с 01.01.2010 по {max_date}; "
    f"команд в выборе: {len(teams)}."
)
if not fifa_lookup:
    st.warning(
        "Рейтинг FIFA пока не загрузился. Прогноз будет работать без него. "
        "Попробуй кнопку «Обновить рейтинг FIFA» в боковой панели."
    )

st.subheader("Выбор матча")
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    home = st.selectbox("Первая команда", teams, index=teams.index("Mexico") if "Mexico" in teams else 0)
with col2:
    default_away = teams.index("South Africa") if "South Africa" in teams else min(1, len(teams) - 1)
    away = st.selectbox("Вторая команда", teams, index=default_away)
with col3:
    neutral = st.checkbox("Нейтральное поле", value=True)

# A natural-width button stays compact on both desktop and mobile.
run_prediction = st.button("Сделать прогноз", type="primary")

if run_prediction:
    try:
        with st.spinner("Считаю прогноз..."):
            df, model, states, h2h, _ = get_prepared_model(force_retrain=force_retrain)
            result = predict_match(
                home,
                away,
                neutral,
                model,
                states,
                h2h,
                fifa_lookup=fifa_lookup,
                df=df,
            )
    except Exception as exc:
        st.error(str(exc))
        st.exception(exc)
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
    total_xg.metric("Ожидаемый тотал", f"{result['expected_goals_home'] + result['expected_goals_away']:.2f}")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Итог",
        "Наиболее вероятные исходы",
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

        st.write("### Вероятности по слоям модели")
        layer_df = pd.DataFrame([
            {"Модель": "Последние 5 матчей", home: result["recent_probs"]["home_win"], "Ничья": result["recent_probs"]["draw"], away: result["recent_probs"]["away_win"]},
            {"Модель": "Рейтинг FIFA", home: result["fifa_probs"]["home_win"], "Ничья": result["fifa_probs"]["draw"], away: result["fifa_probs"]["away_win"]},
            {"Модель": "Машинное обучение", home: result["ml_probs"]["home_win"], "Ничья": result["ml_probs"]["draw"], away: result["ml_probs"]["away_win"]},
            {"Модель": "Пуассон по голам", home: result["poisson_probs"]["home_win"], "Ничья": result["poisson_probs"]["draw"], away: result["poisson_probs"]["away_win"]},
            {"Модель": "Нечёткие правила", home: result["fuzzy_probs"]["home_win"], "Ничья": result["fuzzy_probs"]["draw"], away: result["fuzzy_probs"]["away_win"]},
            {"Модель": "Итог", home: result["prob_home_win"], "Ничья": result["prob_draw"], away: result["prob_away_win"]},
        ])
        st.dataframe(
            layer_df.style.format({home: "{:.1%}", "Ничья": "{:.1%}", away: "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        st.write("### Наиболее вероятный вариант в каждой группе исходов")
        likely_df = pd.DataFrame(result["most_likely_outcomes"])
        st.dataframe(
            likely_df.style.format({"Вероятность": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.warning(
            "Эти вероятности не являются независимыми: их нельзя перемножать и считать готовым экспрессом."
        )
        st.info(
            "Угловые и жёлтые карточки пока не рассчитываются: используемый открытый датасет сборных "
            "не содержит их исторической статистики. Добавлять выдуманные оценки было бы неправильно."
        )

    with tab3:
        st.write("### Сравнение команд")
        summary_df = pd.DataFrame(result["team_summary"])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.write("### Ключевые числовые признаки")
        key_features = {
            "Разница очков FIFA с учётом поля": result["fifa_probs"]["points_diff"],
            "Разница Elo с учётом поля": result["features"]["elo_diff"],
            "Разница очков за 5 матчей": result["features"]["form_points_diff_5"],
            "Разница голов за 5 матчей": result["features"]["goal_diff_form_diff_5"],
            "Разница атаки за 10 матчей": result["features"]["attack_diff_10"],
            "Разница обороны за 10 матчей": result["features"]["defense_diff_10"],
            "Очные встречи: разница голов": result["features"]["h2h_goal_diff"],
        }
        st.dataframe(pd.DataFrame([key_features]).T.rename(columns={0: "Значение"}), use_container_width=True)

    with tab4:
        st.write("### Наиболее вероятные счета")
        score_df = pd.DataFrame(result["top_scorelines"], columns=["Счёт", "Вероятность"])
        st.dataframe(score_df.style.format({"Вероятность": "{:.1%}"}), use_container_width=True, hide_index=True)

        st.write("### Основные тоталы и дополнительные вероятности")
        m = result["markets"]
        market_rows = [
            {"Показатель": "Тотал больше 0,5", "Вероятность": m["over_0_5"]},
            {"Показатель": "Тотал больше 1,5", "Вероятность": m["over_1_5"]},
            {"Показатель": "Тотал меньше 1,5", "Вероятность": m["under_1_5"]},
            {"Показатель": "Тотал больше 2,5", "Вероятность": m["over_2_5"]},
            {"Показатель": "Тотал меньше 2,5", "Вероятность": m["under_2_5"]},
            {"Показатель": "Тотал больше 3,5", "Вероятность": m["over_3_5"]},
            {"Показатель": "Обе забьют — да", "Вероятность": m["btts_yes"]},
            {"Показатель": "Обе забьют — нет", "Вероятность": m["btts_no"]},
            {"Показатель": f"{home} не пропустит", "Вероятность": m["home_clean_sheet"]},
            {"Показатель": f"{away} не пропустит", "Вероятность": m["away_clean_sheet"]},
            {"Показатель": f"{home} или ничья", "Вероятность": result["prob_home_win"] + result["prob_draw"]},
            {"Показатель": f"{away} или ничья", "Вероятность": result["prob_away_win"] + result["prob_draw"]},
        ]
        market_df = pd.DataFrame(market_rows)
        st.dataframe(market_df.style.format({"Вероятность": "{:.1%}"}), use_container_width=True, hide_index=True)

    with tab5:
        st.write(f"### Последние матчи: {home}")
        st.dataframe(result["recent_home"], use_container_width=True, hide_index=True)
        st.write(f"### Последние матчи: {away}")
        st.dataframe(result["recent_away"], use_container_width=True, hide_index=True)

        chart = form_chart_data(df, home, away, n=10)
        if not chart.empty:
            st.write("### Накопленные очки за последние 10 матчей")
            pivot = chart.pivot(index="Матч №", columns="Команда", values="Накопленные очки")
            st.line_chart(pivot)

    with tab6:
        h2h_table = result["h2h_table"]
        if h2h_table.empty:
            st.info("С 1 января 2010 года в базе нет очных встреч этих команд.")
        else:
            st.write("### Очные встречи с 2010 года")
            st.dataframe(h2h_table, use_container_width=True, hide_index=True)

    with tab7:
        st.write("### Баланс двух основных блоков")
        info_weight = result["model_weights"]["recent5"] + result["model_weights"]["fifa"]
        math_weight = (
            result["model_weights"]["ml"]
            + result["model_weights"]["poisson"]
            + result["model_weights"]["fuzzy"]
        )
        group_weights_df = pd.DataFrame([
            {"Блок": "Последние 5 матчей + рейтинг FIFA", "Вес": info_weight},
            {"Блок": "Математические модели", "Вес": math_weight},
        ])
        st.dataframe(
            group_weights_df.style.format({"Вес": "{:.0%}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.write("### Детальные веса итогового прогноза")
        labels = {
            "recent5": "Последние 5 матчей",
            "fifa": "Рейтинг FIFA",
            "ml": "Машинное обучение",
            "poisson": "Модель Пуассона",
            "fuzzy": "Нечёткие правила",
        }
        weights_df = pd.DataFrame([
            {"Компонент": labels.get(key, key), "Вес": value}
            for key, value in result["model_weights"].items()
        ])
        st.dataframe(weights_df.style.format({"Вес": "{:.0%}"}), use_container_width=True, hide_index=True)

        with st.expander("Показать технические признаки"):
            st.json(result["features"])

        st.write("### Качество модели на историческом тесте с 2010 года")
        metrics = getattr(model, "metrics_", {})
        if metrics:
            metrics_df = pd.DataFrame([
                {"Метрика": "Всего матчей", "Значение": int(metrics["matches_total"])},
                {"Метрика": "Матчей в тесте", "Значение": int(metrics["test_matches"])},
                {"Метрика": "Тестовая точность", "Значение": f"{metrics['accuracy'] * 100:.1f}%"},
                {"Метрика": "Базовая точность: всегда победа первой", "Значение": f"{metrics['baseline_home_accuracy'] * 100:.1f}%"},
                {"Метрика": "Log loss", "Значение": f"{metrics['log_loss']:.3f}"},
                {"Метрика": "Точность: победа второй", "Значение": f"{metrics['away_win_accuracy'] * 100:.1f}%"},
                {"Метрика": "Точность: ничья", "Значение": f"{metrics['draw_accuracy'] * 100:.1f}%"},
                {"Метрика": "Точность: победа первой", "Значение": f"{metrics['home_win_accuracy'] * 100:.1f}%"},
            ])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
