from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from football_predictor.config import (
    BACKTEST_PATH,
    FIFA_CURRENT_PATH,
    WORLD_CUP_BACKTEST_PATH,
)
from football_predictor.data_loader import (
    data_coverage,
    load_match_lineups,
    load_optional_stats,
    load_player_pool,
)
from football_predictor.fifa_rankings import refresh_current_ranking_if_stale
from football_predictor.journal import (
    GitHubJournalStore,
    append_local_prediction,
    journal_metrics,
    load_local_journal,
    update_actual_result,
)
from football_predictor.prediction import predict_world_cup_match
from football_predictor.training import load_runtime, train_and_maybe_promote
from football_predictor.world_cup_live import (
    append_lineup_snapshot,
    automatic_match_context,
    fetch_match_lineups,
    get_world_cup_fixtures,
    row_to_fixture,
    selectable_fixtures,
)

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")
st.markdown(
    """
    <style>
      div[data-testid="stButton"] > button {min-height:2.1rem;padding:.3rem .8rem;width:auto;}
      .muted {color:#6b7280;font-size:.9rem}
      .status-box {padding:.7rem 1rem;border:1px solid #d1d5db;border-radius:.5rem;margin:.25rem 0}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ Прогноз матчей чемпионата мира 2026")


def _data_source_key() -> str:
    try:
        cfg = st.secrets.get("data_sources", {})
        return str(cfg.get("football_data_api_key", "")).strip()
    except Exception:
        return ""


def _github_store() -> GitHubJournalStore | None:
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


def _file_stamp(path: str | Path) -> float:
    p = Path(path)
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner="Загружаю модели и историю матчей…")
def runtime(fifa_stamp: float):
    return load_runtime(retrain_if_missing=True)


@st.cache_data(ttl=300, show_spinner=False)
def cached_fixtures(api_key: str) -> pd.DataFrame:
    return get_world_cup_fixtures(api_key=api_key, persist=False)


@st.cache_data(ttl=900, show_spinner=False)
def cached_optional_stats() -> pd.DataFrame:
    return load_optional_stats()


@st.cache_data(ttl=900, show_spinner=False)
def cached_player_pool() -> pd.DataFrame:
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
        rows = [{"Компонент": label, "Относительное влияние": float(coefs[start:end].sum())} for label, start, end in groups]
        frame = pd.DataFrame(rows)
        total = frame["Относительное влияние"].sum()
        if total > 0:
            frame["Относительное влияние"] /= total
        return frame
    except Exception:
        return pd.DataFrame(columns=["Компонент", "Относительное влияние"])


# FIFA refresh is automatic. A failed request leaves the last good file in place.
try:
    refresh_current_ranking_if_stale(max_age_hours=12)
except Exception:
    pass

api_key = _data_source_key()
fixtures = cached_fixtures(api_key)
if fixtures.empty:
    st.error("Не удалось загрузить календарь ЧМ-2026.")
    st.stop()

try:
    matches, fifa_history, bundle, builder = runtime(_file_stamp(FIFA_CURRENT_PATH))
except Exception as exc:
    st.error("Программа не смогла загрузить модель.")
    st.exception(exc)
    st.stop()

optional_stats = cached_optional_stats()
player_pool = cached_player_pool()
lineup_history = load_match_lineups()
coverage = data_coverage(matches, optional_stats, player_pool)

with st.sidebar:
    st.header("Состояние системы")
    st.write(f"Версия модели: `{bundle.version}`")
    st.write(f"Обучена по: **{bundle.train_end_date}**")
    st.write(f"Матчей обучения: **{bundle.training_matches:,}**")
    latest_fifa_dates = fifa_history.frame["date"] if not fifa_history.frame.empty else pd.Series(dtype="datetime64[ns]")
    if not latest_fifa_dates.empty:
        st.write(f"Рейтинг FIFA: **{pd.to_datetime(latest_fifa_dates.max()).date()}**")
    st.divider()
    if api_key:
        st.success("Календарь, результаты, таблицы и составы обновляются через football-data.org.")
    else:
        st.warning("Используется встроенный календарь. Для автоматических результатов, таблиц и составов добавь бесплатный ключ football-data.org в Secrets.")
    st.caption("Рейтинг FIFA обновляется автоматически. Ручная загрузка и ручные коэффициенты не используются.")

st.subheader("1. Выбери матч")
selectable = selectable_fixtures(fixtures)
options = list(selectable.index)
selected_idx = st.selectbox(
    "Матч ЧМ-2026",
    options,
    format_func=lambda idx: row_to_fixture(selectable.loc[idx]).label(),
)
fixture = row_to_fixture(selectable.loc[selected_idx])

context_preview = automatic_match_context(fixtures, fixture, matches, lineups_known=False)
info1, info2, info3 = st.columns(3)
info1.metric("Стадия", f"Группа {fixture.group_name}" if fixture.stage == "group" else "Плей-офф")
info2.metric("Начало", fixture.kickoff_utc.strftime("%d.%m.%Y %H:%M UTC"))
info3.metric("Источник календаря", fixture.source)

if fixture.stage == "group" and not context_preview.standings.empty:
    with st.expander("Таблица группы перед матчем", expanded=False):
        table = context_preview.standings.rename(columns={
            "team": "Команда", "played": "И", "points": "О", "gf": "ЗМ", "ga": "ПМ", "gd": "РМ"
        })
        st.dataframe(table, use_container_width=True, hide_index=True)

st.caption("Турнирный контекст, положение в группе, дни отдыха, мотивация и риск ротации определяются автоматически.")

run = st.button("Сделать прогноз", type="primary")
if run:
    with st.spinner("Обновляю контекст матча, проверяю составы и считаю модели…"):
        try:
            lineup_snapshot = fetch_match_lineups(fixture.source_match_id, api_key=api_key)
            if lineup_snapshot.available:
                append_lineup_snapshot(fixture, lineup_snapshot)
                lineup_history = load_match_lineups()
            auto = automatic_match_context(fixtures, fixture, matches, lineups_known=lineup_snapshot.available)
            source_notes = list(auto.source_notes)
            source_notes.append(lineup_snapshot.message)
            source_notes.append(
                f"Расширенная статистика доступна для {coverage['optional_matches']:,} матчей; "
                "она используется только там, где открытый источник реально предоставляет значения."
            )
            prediction = predict_world_cup_match(
                bundle=bundle,
                builder=builder,
                fifa=fifa_history,
                home=fixture.home_team,
                away=fixture.away_team,
                match_date=fixture.kickoff_utc,
                neutral=auto.neutral,
                context=auto.context,
                selected_home_players=lineup_snapshot.home_players if lineup_snapshot.available else None,
                selected_away_players=lineup_snapshot.away_players if lineup_snapshot.available else None,
                player_pool=player_pool,
                lineup_history=lineup_history,
                data_source_notes=source_notes,
            )
            prediction["fixture"] = fixture.as_dict()
            prediction["lineup_source_message"] = lineup_snapshot.message
            st.session_state["prediction"] = prediction
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
        "Обоснование", "Наиболее вероятные исходы", "Голы и точный счёт",
        "Таймы, угловые, карточки", "Составы и данные", "Модели", "Тестирование", "Журнал",
    ])

    with tabs[0]:
        st.write("### Итог")
        st.info(result.get("summary", result["explanations"][0]))
        st.write("### Факторы прогноза")
        explanation_df = pd.DataFrame(result.get("explanation_rows", []))
        if not explanation_df.empty:
            st.dataframe(explanation_df, use_container_width=True, hide_index=True)
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
            ("Первый тайм", result["halftime"]), ("Второй тайм", result["second_half"]),
            ("Угловые", result["corners"]), ("Жёлтые карточки", result["cards"]),
        ]
        for title, section in sections:
            st.write(f"### {title}")
            if section.get("available"):
                rows = [{"Показатель": key, "Значение": value} for key, value in section.items() if key != "available"]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info(section.get("reason", "Модель пока недоступна."))

    with tabs[4]:
        st.write("### Стартовые составы")
        squad_df = pd.DataFrame([
            {"Команда": result["home"], "Относительная сила": result["home_squad"]["relative_strength"],
             "Ключевых потерь": result["home_squad"]["missing_key_players"], "Пояснение": result["home_squad"]["explanation"]},
            {"Команда": result["away"], "Относительная сила": result["away_squad"]["relative_strength"],
             "Ключевых потерь": result["away_squad"]["missing_key_players"], "Пояснение": result["away_squad"]["explanation"]},
        ])
        st.dataframe(squad_df.style.format({"Относительная сила": "{:.1%}"}), use_container_width=True, hide_index=True)
        st.caption(result.get("lineup_source_message", ""))
        st.write("### Покрытие автоматически собранных данных")
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
        labels = {"ml": "Машинное обучение", "dixon_coles": "Dixon–Coles", "elo": "Elo", "fifa": "FIFA", "recent_form": "Последние матчи с учётом соперников"}
        layer_rows = [{
            "Модель": labels.get(key, key), result["home"]: probs["home"], "Ничья": probs["draw"], result["away"]: probs["away"]
        } for key, probs in result["components"].items()]
        layer_rows.append({"Модель": "Итог после метамодели и калибровки", result["home"]: result["prob_home_win"], "Ничья": result["prob_draw"], result["away"]: result["prob_away_win"]})
        layer_df = pd.DataFrame(layer_rows)
        st.dataframe(layer_df.style.format({result["home"]: "{:.1%}", "Ничья": "{:.1%}", result["away"]: "{:.1%}"}), use_container_width=True, hide_index=True)
        st.write("### Влияние компонентов, изученное метамоделью")
        importance_df = model_component_importance(bundle)
        st.dataframe(importance_df.style.format({"Относительное влияние": "{:.1%}"}), use_container_width=True, hide_index=True)

    with tabs[6]:
        st.write("### Хронологическое тестирование")
        metrics_df = pd.DataFrame(bundle.metrics)
        st.dataframe(metrics_df.style.format({"accuracy": "{:.1%}", "log_loss": "{:.4f}", "brier": "{:.4f}"}), use_container_width=True, hide_index=True)
        st.write(f"Температура калибровки: **{bundle.calibrator.temperature_:.3f}**")
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
        store = _github_store()
        st.write("### Сохранение прогноза")
        if st.button("Сохранить этот прогноз в журнал"):
            try:
                journal_df = store.append(result) if store else append_local_prediction(result)
                st.success("Прогноз сохранён.")
                st.session_state["journal"] = journal_df
            except Exception as exc:
                st.error(f"Не удалось сохранить прогноз: {exc}")
        try:
            journal_df, _ = store.read() if store else (load_local_journal(), None)
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
