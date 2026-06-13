from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from football_predictor.config import (
    BACKTEST_PATH,
    WORLD_CUP_BACKTEST_PATH,
    ENRICHED_STATS_PATH,
    FIFA_CURRENT_PATH,
    MATCH_LINEUPS_PATH,
    PLAYER_POOL_PATH,
    WORLD_CUP_TEAMS,
)
from football_predictor.context import MatchContext
from football_predictor.data_loader import (
    data_coverage,
    download_results,
    load_optional_stats,
    load_player_pool,
    world_cup_team_list,
)
from football_predictor.fifa_rankings import download_current_ranking
from football_predictor.journal import (
    GitHubJournalStore,
    append_local_prediction,
    journal_metrics,
    load_local_journal,
    update_actual_result,
)
from football_predictor.prediction import predict_world_cup_match
from football_predictor.training import load_runtime, train_and_maybe_promote

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")
st.markdown(
    """
    <style>
      div[data-testid="stButton"] > button {min-height: 2.1rem; padding: .3rem .8rem; width: auto;}
      .muted {color:#6b7280;font-size:.9rem}
      .status-box {padding:.7rem 1rem;border:1px solid #d1d5db;border-radius:.5rem;margin:.25rem 0}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ Прогноз матчей чемпионата мира 2026")


@st.cache_resource(show_spinner="Загружаю модели и историю матчей…")
def runtime():
    return load_runtime(retrain_if_missing=True)


@st.cache_data(show_spinner=False)
def cached_optional_stats():
    return load_optional_stats()


@st.cache_data(show_spinner=False)
def cached_player_pool():
    return load_player_pool()


def clear_all_caches() -> None:
    st.cache_resource.clear()
    st.cache_data.clear()


def model_component_importance(bundle) -> pd.DataFrame:
    try:
        estimator = bundle.meta_model.named_steps["model"]
        coefs = np.mean(np.abs(estimator.coef_), axis=0)
        groups = [
            ("Машинное обучение", 0, 3),
            ("Dixon–Coles", 3, 6),
            ("Elo", 6, 9),
            ("FIFA", 9, 12),
            ("Последние матчи", 12, 15),
            ("Турнирный контекст и качество данных", 15, len(coefs)),
        ]
        rows = []
        for label, start, end in groups:
            rows.append({"Компонент": label, "Относительное влияние": float(coefs[start:end].sum())})
        frame = pd.DataFrame(rows)
        total = frame["Относительное влияние"].sum()
        if total > 0:
            frame["Относительное влияние"] /= total
        return frame
    except Exception:
        return pd.DataFrame(columns=["Компонент", "Относительное влияние"])


def get_github_store() -> GitHubJournalStore | None:
    try:
        cfg = st.secrets.get("github_journal", {})
        token = cfg.get("token")
        repo = cfg.get("repo")
        path = cfg.get("path", "data/prediction_log.csv")
        branch = cfg.get("branch", "main")
        if token and repo:
            return GitHubJournalStore(token=token, repo=repo, path=path, branch=branch)
    except Exception:
        return None
    return None


with st.sidebar:
    st.header("Управление")
    if st.button("Обновить рейтинг FIFA"):
        with st.spinner("Загружаю текущий рейтинг FIFA…"):
            frame = download_current_ranking(FIFA_CURRENT_PATH)
        clear_all_caches()
        if frame.empty:
            st.warning("Текущий рейтинг не загрузился. Программа продолжит использовать последний исторический снимок.")
        else:
            st.success(f"Загружено сборных: {len(frame)}")
        st.rerun()

    if st.button("Обновить матчи и обучить кандидата"):
        with st.spinner("Обновляю данные, обучаю и сравниваю новую модель со старой…"):
            try:
                _, promoted, reason = train_and_maybe_promote(download=True, force=False)
                clear_all_caches()
                st.success(("Новая модель принята. " if promoted else "Старая модель оставлена. ") + reason)
            except Exception as exc:
                st.error(f"Не удалось выполнить переобучение: {exc}")
        st.rerun()

    if st.button("Очистить кэш"):
        clear_all_caches()
        st.rerun()

    st.caption("Автоматические веса не задаются вручную: итоговый ансамбль обучает метамодель на прошлых матчах.")

try:
    matches, fifa_history, bundle, builder = runtime()
except Exception as exc:
    st.error("Программа не смогла загрузить модель.")
    st.exception(exc)
    st.stop()

optional_stats = cached_optional_stats()
player_pool = cached_player_pool()
coverage = data_coverage(matches, optional_stats, player_pool)

with st.sidebar:
    st.divider()
    st.write("**Состояние модели**")
    st.write(f"Версия: `{bundle.version}`")
    st.write(f"Обучена по: **{bundle.train_end_date}**")
    st.write(f"Матчей обучения: **{bundle.training_matches:,}**")
    latest_fifa_dates = fifa_history.frame["date"] if not fifa_history.frame.empty else pd.Series(dtype="datetime64[ns]")
    if not latest_fifa_dates.empty:
        latest_fifa_date = pd.to_datetime(latest_fifa_dates.max()).date()
        st.write(f"Последний рейтинг FIFA: **{latest_fifa_date}**")
        if (date.today() - latest_fifa_date).days > 90:
            st.warning("Рейтинг FIFA устарел. Нажми «Обновить рейтинг FIFA» перед прогнозом.")

st.subheader("1. Выбери матч и турнирную ситуацию")
mode = st.radio("Стадия", ["Групповой этап", "Плей-офф"], horizontal=True)

if mode == "Групповой этап":
    group_name = st.selectbox("Группа", sorted(WORLD_CUP_TEAMS.keys()))
    group_teams = WORLD_CUP_TEAMS[group_name]
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Первая команда", group_teams, index=0, key="group_home")
    with col2:
        away_options = [x for x in group_teams if x != home]
        away = st.selectbox("Вторая команда", away_options, index=0, key="group_away")
    group_round = st.selectbox("Тур группы", [1, 2, 3], index=0)
else:
    group_name = ""
    all_teams = world_cup_team_list()
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Первая команда", all_teams, index=0, key="ko_home")
    with col2:
        away = st.selectbox("Вторая команда", [x for x in all_teams if x != home], index=0, key="ko_away")
    group_round = 3

with st.expander("Турнирная ситуация, отдых и ротация", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        match_date = st.date_input("Дата матча", value=date.today())
        neutral = st.checkbox("Нейтральное поле", value=True)
    with c2:
        home_points = st.number_input(f"Очки {home} перед матчем", min_value=0, max_value=9, value=0, disabled=mode != "Групповой этап")
        away_points = st.number_input(f"Очки {away} перед матчем", min_value=0, max_value=9, value=0, disabled=mode != "Групповой этап")
        home_gd = st.number_input(f"Разница мячей {home}", min_value=-20, max_value=20, value=0, disabled=mode != "Групповой этап")
        away_gd = st.number_input(f"Разница мячей {away}", min_value=-20, max_value=20, value=0, disabled=mode != "Групповой этап")
    with c3:
        home_rest = st.number_input(f"Дней отдыха {home}", min_value=1, max_value=14, value=5)
        away_rest = st.number_input(f"Дней отдыха {away}", min_value=1, max_value=14, value=5)
        home_rotation = st.slider(f"Риск ротации {home}", 0, 100, 0, step=5) / 100
        away_rotation = st.slider(f"Риск ротации {away}", 0, 100, 0, step=5) / 100

    m1, m2 = st.columns(2)
    with m1:
        home_must_win = st.checkbox(f"{home}: победа необходима")
        home_draw_enough = st.checkbox(f"{home}: ничьей достаточно")
    with m2:
        away_must_win = st.checkbox(f"{away}: победа необходима")
        away_draw_enough = st.checkbox(f"{away}: ничьей достаточно")

with st.expander("Составы", expanded=False):
    lineups_known = st.checkbox("Стартовые составы уже известны")
    if player_pool.empty:
        st.info("Файл player_pool.csv пока пуст. До загрузки данных игроков можно использовать ручную оценку состава.")
        s1, s2 = st.columns(2)
        with s1:
            home_squad_percent = st.slider(f"Сила состава {home}, % от оптимального", 50, 105, 100)
        with s2:
            away_squad_percent = st.slider(f"Сила состава {away}, % от оптимального", 50, 105, 100)
        selected_home_players = None
        selected_away_players = None
    else:
        home_players = player_pool[player_pool["team"] == home]["player"].dropna().astype(str).tolist()
        away_players = player_pool[player_pool["team"] == away]["player"].dropna().astype(str).tolist()
        s1, s2 = st.columns(2)
        with s1:
            selected_home_players = st.multiselect(f"Игроки стартового состава {home}", home_players, default=home_players[:11])
        with s2:
            selected_away_players = st.multiselect(f"Игроки стартового состава {away}", away_players, default=away_players[:11])
        home_squad_percent = 100
        away_squad_percent = 100

context = MatchContext(
    stage="group" if mode == "Групповой этап" else "knockout",
    group_name=group_name,
    group_round=group_round,
    home_points=int(home_points) if mode == "Групповой этап" else 0,
    away_points=int(away_points) if mode == "Групповой этап" else 0,
    home_goal_difference=int(home_gd) if mode == "Групповой этап" else 0,
    away_goal_difference=int(away_gd) if mode == "Групповой этап" else 0,
    home_must_win=home_must_win,
    away_must_win=away_must_win,
    home_draw_enough=home_draw_enough,
    away_draw_enough=away_draw_enough,
    home_rotation_risk=home_rotation,
    away_rotation_risk=away_rotation,
    home_days_rest=int(home_rest),
    away_days_rest=int(away_rest),
    lineups_known=lineups_known,
    extra_time_possible=mode == "Плей-офф",
)

run = st.button("Сделать прогноз", type="primary")
if run:
    with st.spinner("Считаю отдельные модели, применяю метамодель и калибровку…"):
        try:
            st.session_state["prediction"] = predict_world_cup_match(
                bundle=bundle,
                builder=builder,
                fifa=fifa_history,
                home=home,
                away=away,
                match_date=match_date,
                neutral=neutral,
                context=context,
                selected_home_players=selected_home_players,
                selected_away_players=selected_away_players,
                player_pool=player_pool,
                manual_home_strength=home_squad_percent / 100,
                manual_away_strength=away_squad_percent / 100,
            )
        except Exception as exc:
            st.error(f"Ошибка прогноза: {exc}")
            st.exception(exc)

result = st.session_state.get("prediction")
if result:
    st.divider()
    st.subheader(f"2. Прогноз: {result['home']} — {result['away']}")
    p1, px, p2 = st.columns(3)
    p1.metric(f"Победа {result['home']}", f"{result['prob_home_win'] * 100:.1f}%")
    px.metric("Ничья", f"{result['prob_draw'] * 100:.1f}%")
    p2.metric(f"Победа {result['away']}", f"{result['prob_away_win'] * 100:.1f}%")

    g1, g2, gt = st.columns(3)
    g1.metric(f"Ожидаемые голы {result['home']}", f"{result['expected_goals_home']:.2f}")
    g2.metric(f"Ожидаемые голы {result['away']}", f"{result['expected_goals_away']:.2f}")
    gt.metric("Ожидаемый тотал", f"{result['expected_goals_home'] + result['expected_goals_away']:.2f}")

    tabs = st.tabs([
        "Обоснование",
        "Наиболее вероятные исходы",
        "Голы и точный счёт",
        "Таймы, угловые, карточки",
        "Составы и данные",
        "Модели",
        "Тестирование",
        "Журнал",
    ])

    with tabs[0]:
        st.write("### Почему получен такой прогноз")
        for explanation in result["explanations"]:
            st.write(f"- {explanation}")
        if result.get("progression"):
            pr = result["progression"]
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Проход {result['home']}", f"{pr['home_advance'] * 100:.1f}%")
            c2.metric(f"Проход {result['away']}", f"{pr['away_advance'] * 100:.1f}%")
            c3.metric("Дополнительное время", f"{pr['extra_time_probability'] * 100:.1f}%")

    with tabs[1]:
        outcomes = pd.DataFrame(result["outcomes"])
        st.dataframe(outcomes.style.format({"Вероятность": "{:.1%}"}), use_container_width=True, hide_index=True)

    with tabs[2]:
        st.write("### Наиболее вероятные счета")
        score_df = pd.DataFrame(result["markets"]["top_scorelines"], columns=["Счёт", "Вероятность"])
        st.dataframe(score_df.style.format({"Вероятность": "{:.1%}"}), use_container_width=True, hide_index=True)
        market_rows = [
            ("Тотал больше 1,5", result["markets"]["over_1_5"]),
            ("Тотал меньше 1,5", result["markets"]["under_1_5"]),
            ("Тотал больше 2,5", result["markets"]["over_2_5"]),
            ("Тотал меньше 2,5", result["markets"]["under_2_5"]),
            ("Тотал больше 3,5", result["markets"]["over_3_5"]),
            ("Тотал меньше 3,5", result["markets"]["under_3_5"]),
            ("Обе забьют — да", result["markets"]["btts_yes"]),
            ("Обе забьют — нет", result["markets"]["btts_no"]),
        ]
        market_df = pd.DataFrame(market_rows, columns=["Рынок", "Вероятность"])
        st.dataframe(market_df.style.format({"Вероятность": "{:.1%}"}), use_container_width=True, hide_index=True)

    with tabs[3]:
        sections = [
            ("Первый тайм", result["halftime"]),
            ("Второй тайм", result["second_half"]),
            ("Угловые", result["corners"]),
            ("Жёлтые карточки", result["cards"]),
        ]
        for title, section in sections:
            st.write(f"### {title}")
            if section.get("available"):
                rows = [{"Показатель": key, "Значение": value} for key, value in section.items() if key != "available"]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info(section.get("reason", "Модель пока недоступна."))

    with tabs[4]:
        st.write("### Составы")
        squad_df = pd.DataFrame([
            {
                "Команда": result["home"],
                "Относительная сила": result["home_squad"]["relative_strength"],
                "Ключевых потерь": result["home_squad"]["missing_key_players"],
                "Пояснение": result["home_squad"]["explanation"],
            },
            {
                "Команда": result["away"],
                "Относительная сила": result["away_squad"]["relative_strength"],
                "Ключевых потерь": result["away_squad"]["missing_key_players"],
                "Пояснение": result["away_squad"]["explanation"],
            },
        ])
        st.dataframe(squad_df.style.format({"Относительная сила": "{:.1%}"}), use_container_width=True, hide_index=True)
        st.write("### Покрытие данных")
        coverage_df = pd.DataFrame([
            ("Матчи с результатами", f"{coverage['results_matches']:,}"),
            ("Матчи с расширенной статистикой", f"{coverage['optional_matches']:,}"),
            ("Покрытие xG", f"{coverage['xg_coverage']:.1%}"),
            ("Покрытие угловых", f"{coverage['corners_coverage']:.1%}"),
            ("Покрытие карточек", f"{coverage['cards_coverage']:.1%}"),
            ("Покрытие таймов", f"{coverage['halftime_coverage']:.1%}"),
            ("Игроков в базе", f"{coverage['players']:,}"),
        ], columns=["Показатель", "Значение"])
        st.dataframe(coverage_df, use_container_width=True, hide_index=True)

    with tabs[5]:
        st.write("### Вероятности отдельных моделей")
        layer_rows = []
        labels = {
            "ml": "Машинное обучение",
            "dixon_coles": "Dixon–Coles",
            "elo": "Elo",
            "fifa": "FIFA",
            "recent_form": "Последние матчи с учётом соперников",
        }
        for key, probs in result["components"].items():
            layer_rows.append({
                "Модель": labels.get(key, key),
                result["home"]: probs["home"],
                "Ничья": probs["draw"],
                result["away"]: probs["away"],
            })
        layer_rows.append({
            "Модель": "Итог после метамодели и калибровки",
            result["home"]: result["prob_home_win"],
            "Ничья": result["prob_draw"],
            result["away"]: result["prob_away_win"],
        })
        layer_df = pd.DataFrame(layer_rows)
        st.dataframe(layer_df.style.format({result["home"]: "{:.1%}", "Ничья": "{:.1%}", result["away"]: "{:.1%}"}), use_container_width=True, hide_index=True)
        st.write("### Влияние компонентов, изученное метамоделью")
        importance_df = model_component_importance(bundle)
        st.dataframe(importance_df.style.format({"Относительное влияние": "{:.1%}"}), use_container_width=True, hide_index=True)
        st.caption("Это не вручную назначенные коэффициенты. Значения рассчитаны по обученным коэффициентам метамодели.")

    with tabs[6]:
        st.write("### Честный хронологический тест")
        metrics_df = pd.DataFrame(bundle.metrics)
        st.dataframe(metrics_df.style.format({"accuracy": "{:.1%}", "log_loss": "{:.4f}", "brier": "{:.4f}"}), use_container_width=True, hide_index=True)
        st.write(f"Температура калибровки: **{bundle.calibrator.temperature_:.3f}**")
        st.caption("Обучение, подбор ансамбля, калибровка и итоговый тест разделены по времени. Будущие матчи не используются при обучении прошлых прогнозов.")
        if Path(WORLD_CUP_BACKTEST_PATH).exists():
            try:
                wc_bt = pd.read_csv(WORLD_CUP_BACKTEST_PATH)
                if not wc_bt.empty:
                    st.write("### Проверка на прошлых чемпионатах мира")
                    st.dataframe(wc_bt.style.format({"accuracy": "{:.1%}", "log_loss": "{:.4f}", "brier": "{:.4f}"}), use_container_width=True, hide_index=True)
            except Exception:
                pass
        if Path(BACKTEST_PATH).exists():
            try:
                bt = pd.read_csv(BACKTEST_PATH)
                if not bt.empty:
                    st.write("### Последний отчёт обучения")
                    st.dataframe(bt.style.format({"accuracy": "{:.1%}", "log_loss": "{:.4f}", "brier": "{:.4f}"}), use_container_width=True, hide_index=True)
            except Exception:
                pass

    with tabs[7]:
        store = get_github_store()
        st.write("### Сохранение прогноза")
        if st.button("Сохранить этот прогноз в журнал"):
            try:
                if store:
                    journal_df = store.append(result)
                    st.success("Прогноз сохранён в GitHub-журнале.")
                else:
                    journal_df = append_local_prediction(result)
                    st.success("Прогноз сохранён локально. На Streamlit Cloud локальный файл может исчезнуть после перезапуска; настрой GitHub-журнал для постоянного хранения.")
                st.session_state["journal"] = journal_df
            except Exception as exc:
                st.error(f"Не удалось сохранить прогноз: {exc}")

        try:
            if store:
                journal_df, _ = store.read()
            else:
                journal_df = load_local_journal()
        except Exception:
            journal_df = load_local_journal()
        st.dataframe(journal_df, use_container_width=True, hide_index=True)
        st.download_button("Скачать журнал CSV", journal_df.to_csv(index=False).encode("utf-8-sig"), file_name="prediction_log.csv", mime="text/csv")

        open_rows = journal_df[journal_df["status"].astype(str) == "open"] if not journal_df.empty else journal_df
        if not open_rows.empty:
            st.write("### Записать фактический результат")
            selected_id = st.selectbox("Прогноз", open_rows["prediction_id"].astype(str).tolist(), format_func=lambda x: f"{open_rows.loc[open_rows['prediction_id'].astype(str)==x, 'home_team'].iloc[0]} — {open_rows.loc[open_rows['prediction_id'].astype(str)==x, 'away_team'].iloc[0]}")
            a, b = st.columns(2)
            with a:
                actual_home = st.number_input("Голы первой команды", 0, 20, 0)
            with b:
                actual_away = st.number_input("Голы второй команды", 0, 20, 0)
            if st.button("Сохранить результат"):
                try:
                    updated = store.update_result(selected_id, int(actual_home), int(actual_away)) if store else update_actual_result(selected_id, int(actual_home), int(actual_away))
                    st.success("Результат сохранён.")
                    st.dataframe(updated, use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.error(f"Не удалось сохранить результат: {exc}")
        jm = journal_metrics(journal_df)
        if jm["completed"]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Завершённых прогнозов", int(jm["completed"]))
            c2.metric("Точность", f"{jm['accuracy'] * 100:.1f}%")
            c3.metric("Log Loss", f"{jm['log_loss']:.3f}")
            c4.metric("Brier", f"{jm['brier']:.3f}")

st.divider()
with st.expander("Загрузка расширенной статистики и данных игроков"):
    st.write("Постоянное обновление удобнее делать через GitHub: замени CSV-файлы в папке `data`, затем запусти переобучение.")
    for label, path in [
        ("Шаблон расширенной статистики", ENRICHED_STATS_PATH),
        ("Шаблон игроков", PLAYER_POOL_PATH),
        ("Шаблон составов матчей", MATCH_LINEUPS_PATH),
    ]:
        file_path = Path(path)
        if file_path.exists():
            st.download_button(label, file_path.read_bytes(), file_name=file_path.name, mime="text/csv", key=label)
    st.write("### Ручная загрузка текущего рейтинга FIFA")
    st.caption("CSV должен содержать колонки: team, points, rank, date. Это запасной вариант, если автоматическое обновление недоступно.")
    uploaded_fifa = st.file_uploader("Выбери CSV рейтинга FIFA", type=["csv"], key="fifa_upload")
    if uploaded_fifa is not None and st.button("Сохранить загруженный рейтинг FIFA"):
        try:
            frame = pd.read_csv(uploaded_fifa)
            required = {"team", "points", "rank", "date"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"Не хватает колонок: {sorted(missing)}")
            frame.to_csv(FIFA_CURRENT_PATH, index=False)
            clear_all_caches()
            st.success("Рейтинг сохранён. Страница будет перезагружена.")
            st.rerun()
        except Exception as exc:
            st.error(f"Не удалось сохранить рейтинг: {exc}")
