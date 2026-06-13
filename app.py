from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from football_predictor.config import (
    BACKTEST_PATH,
    ENRICHED_STATS_PATH,
    FIFA_CURRENT_PATH,
    MATCH_LINEUPS_PATH,
    MODEL_BUNDLE_PATH,
    PLAYER_POOL_PATH,
    WORLD_CUP_BACKTEST_PATH,
)
from football_predictor.data_loader import data_coverage, load_match_lineups, load_optional_stats, load_player_pool
from football_predictor.fifa_rankings import refresh_current_ranking_if_stale
from football_predictor.journal import (
    GitHubJournalStore,
    append_local_prediction,
    journal_metrics,
    load_local_journal,
    update_actual_result,
)
from football_predictor.odds_provider import fetch_market_snapshot
from football_predictor.prediction import predict_world_cup_match
from football_predictor.training import load_runtime
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
      .best-card {padding:1rem 1.2rem;border:1px solid #9ca3af;border-radius:.75rem;background:#f8fafc;margin:.4rem 0 1rem 0;}
      .best-card h3 {margin:0 0 .25rem 0;}
      .muted {color:#6b7280;font-size:.9rem}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ Прогноз матчей чемпионата мира 2026")


def _secret(section: str, name: str) -> str:
    try:
        return str(st.secrets.get(section, {}).get(name, "")).strip()
    except Exception:
        return ""


def _github_store() -> GitHubJournalStore | None:
    try:
        cfg = st.secrets.get("github_journal", {})
        if cfg.get("token") and cfg.get("repo"):
            return GitHubJournalStore(
                token=cfg["token"], repo=cfg["repo"],
                path=cfg.get("path", "data/prediction_log.csv"), branch=cfg.get("branch", "main"),
            )
    except Exception:
        pass
    return None


def _file_stamp(path: str | Path) -> float:
    p = Path(path)
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner="Загружаю модели и историю матчей…")
def runtime(model_stamp: float, fifa_stamp: float, stats_stamp: float):
    return load_runtime(retrain_if_missing=True)


@st.cache_data(ttl=180, show_spinner=False)
def cached_fixtures(api_key: str) -> pd.DataFrame:
    return get_world_cup_fixtures(api_key=api_key, persist=False)


@st.cache_data(ttl=300, show_spinner=False)
def cached_optional_stats() -> pd.DataFrame:
    return load_optional_stats()


@st.cache_data(ttl=300, show_spinner=False)
def cached_player_pool() -> pd.DataFrame:
    return load_player_pool()


@st.cache_data(ttl=300, show_spinner=False)
def cached_market_snapshot(home: str, away: str, kickoff: str, key: str):
    return fetch_market_snapshot(home, away, kickoff, key)


def model_component_importance(bundle) -> pd.DataFrame:
    try:
        estimator = bundle.meta_model.named_steps["model"]
        coefs = np.mean(np.abs(estimator.coef_), axis=0)
        groups = [
            ("Машинное обучение", 0, 3), ("Dixon–Coles", 3, 6), ("Elo", 6, 9),
            ("FIFA", 9, 12), ("Последние матчи", 12, 15),
            ("Турнирный контекст и качество данных", 15, len(coefs)),
        ]
        frame = pd.DataFrame([
            {"Компонент": label, "Относительное влияние": float(coefs[start:end].sum())}
            for label, start, end in groups
        ])
        total = frame["Относительное влияние"].sum()
        if total > 0:
            frame["Относительное влияние"] /= total
        return frame
    except Exception:
        return pd.DataFrame(columns=["Компонент", "Относительное влияние"])


def _status_frame(rows: list[dict]) -> pd.DataFrame:
    output = []
    for row in rows:
        output.append({
            "Модель": row.get("name", ""),
            "Статус": "Активна" if row.get("active") else "Не обучена",
            "Матчей": int(row.get("rows") or 0),
            "Проверка": int(row.get("validation_rows") or 0),
            "Период": f"{row.get('data_from') or '—'} — {row.get('data_to') or '—'}",
            "Алгоритм": row.get("selected_algorithm") or "—",
            "MAE": row.get("mae"),
            "MAE среднего": row.get("baseline_mae"),
            "Изменение качества": row.get("improvement"),
            "Пояснение": row.get("reason", ""),
        })
    return pd.DataFrame(output)


def _section_probability_rows(title: str, section: dict, home: str, away: str) -> pd.DataFrame:
    if title == "Первый тайм":
        return pd.DataFrame([
            (f"Победа {home}", section["home_win"]), ("Ничья", section["draw"]), (f"Победа {away}", section["away_win"]),
            ("Тотал больше 0,5", section["over_0_5"]), ("Тотал больше 1,5", section["over_1_5"]),
        ], columns=["Исход", "Вероятность"])
    if title == "Второй тайм":
        return pd.DataFrame([
            (f"Победа {home}", section["home_win"]), ("Ничья", section["draw"]), (f"Победа {away}", section["away_win"]),
            ("Тотал больше 0,5", section["over_0_5"]), ("Тотал больше 1,5", section["over_1_5"]),
        ], columns=["Исход", "Вероятность"])
    lines = (7.5, 8.5, 9.5, 10.5) if title == "Угловые" else (2.5, 3.5, 4.5, 5.5)
    prefix = "Тотал угловых" if title == "Угловые" else "Тотал жёлтых карточек"
    rows = []
    for line in lines:
        suffix = str(line).replace(".", "_")
        rows.extend([
            (f"{prefix} больше {str(line).replace('.', ',')}", section[f"over_{suffix}"]),
            (f"{prefix} меньше {str(line).replace('.', ',')}", section[f"under_{suffix}"]),
        ])
    return pd.DataFrame(rows, columns=["Исход", "Вероятность"])


try:
    refresh_current_ranking_if_stale(max_age_hours=12)
except Exception:
    pass

football_data_key = _secret("data_sources", "football_data_api_key")
odds_key = _secret("data_sources", "odds_api_key")
fixtures = cached_fixtures(football_data_key)
if fixtures.empty:
    st.error("Не удалось загрузить календарь ЧМ-2026.")
    st.stop()

try:
    matches, fifa_history, bundle, builder = runtime(
        _file_stamp(MODEL_BUNDLE_PATH), _file_stamp(FIFA_CURRENT_PATH), _file_stamp(ENRICHED_STATS_PATH)
    )
except Exception as exc:
    st.error("Программа не смогла загрузить или переобучить модель.")
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
    latest_fifa = fifa_history.frame["date"] if not fifa_history.frame.empty else pd.Series(dtype="datetime64[ns]")
    if not latest_fifa.empty:
        st.write(f"Рейтинг FIFA: **{pd.to_datetime(latest_fifa.max()).date()}**")
    st.divider()
    if football_data_key:
        st.success("Календарь, результаты и составы подключены.")
    else:
        st.warning("Без ключа football-data.org составы и текущие результаты могут поступать с задержкой.")
    st.caption("FIFA и открытая статистика обновляются автоматически. Коэффициенты нигде не показываются.")

st.subheader("1. Выбери матч")
selectable = selectable_fixtures(fixtures)
selected_idx = st.selectbox(
    "Матч ЧМ-2026", list(selectable.index),
    format_func=lambda idx: row_to_fixture(selectable.loc[idx]).label(),
)
fixture = row_to_fixture(selectable.loc[selected_idx])
context_preview = automatic_match_context(fixtures, fixture, matches, lineups_known=False)
info1, info2, info3 = st.columns(3)
info1.metric("Стадия", f"Группа {fixture.group_name}" if fixture.stage == "group" else "Плей-офф")
info2.metric("Начало", fixture.kickoff_utc.strftime("%d.%m.%Y %H:%M UTC"))
info3.metric("Источник", fixture.source)

if fixture.stage == "group" and not context_preview.standings.empty:
    with st.expander("Таблица группы перед матчем", expanded=False):
        table = context_preview.standings.rename(columns={"team": "Команда", "played": "И", "points": "О", "gf": "ЗМ", "ga": "ПМ", "gd": "РМ"})
        st.dataframe(table, use_container_width=True, hide_index=True)

run = st.button("Сделать прогноз", type="primary")
if run:
    with st.spinner("Проверяю составы, турнирный контекст, открытые данные и модели…"):
        try:
            lineup_snapshot = fetch_match_lineups(fixture.source_match_id, api_key=football_data_key)
            if lineup_snapshot.available:
                append_lineup_snapshot(fixture, lineup_snapshot)
                lineup_history = load_match_lineups()
            auto = automatic_match_context(fixtures, fixture, matches, lineups_known=lineup_snapshot.available)
            market_snapshot = cached_market_snapshot(
                fixture.home_team, fixture.away_team, fixture.kickoff_utc.isoformat(), odds_key
            )
            source_notes = list(auto.source_notes) + [lineup_snapshot.message]
            source_notes.append(
                f"Расширенная статистика: {coverage['optional_matches']:,} матчей; "
                f"таймы {coverage['halftime_rows']:,}, угловые {coverage['corners_rows']:,}, карточки {coverage['cards_rows']:,}."
            )
            prediction = predict_world_cup_match(
                bundle=bundle, builder=builder, fifa=fifa_history,
                home=fixture.home_team, away=fixture.away_team,
                match_date=fixture.kickoff_utc, neutral=auto.neutral, context=auto.context,
                selected_home_players=lineup_snapshot.home_players if lineup_snapshot.available else None,
                selected_away_players=lineup_snapshot.away_players if lineup_snapshot.available else None,
                player_pool=player_pool, lineup_history=lineup_history,
                data_source_notes=source_notes, market_snapshot=market_snapshot,
            )
            prediction["fixture"] = fixture.as_dict()
            prediction["lineup_source_message"] = lineup_snapshot.message
            st.session_state["prediction"] = prediction
        except Exception as exc:
            st.error(f"Ошибка прогноза: {exc}")
            st.exception(exc)

result = st.session_state.get("prediction")
if not result:
    st.stop()

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
    st.info(result["summary"])
    explanation = pd.DataFrame(result.get("explanation_rows", []))
    if not explanation.empty:
        st.dataframe(explanation, use_container_width=True, hide_index=True)
    if result.get("progression"):
        pr = result["progression"]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Проход {result['home']}", f"{pr['home_advance'] * 100:.1f}%")
        c2.metric(f"Проход {result['away']}", f"{pr['away_advance'] * 100:.1f}%")
        c3.metric("Дополнительное время", f"{pr['extra_time_probability'] * 100:.1f}%")

with tabs[1]:
    selection = result.get("non_obvious_selection", {})
    st.write("### Лучший неочевидный исход")
    if selection.get("found"):
        best = selection["best"]
        st.markdown(
            f"""<div class="best-card"><h3>{best['Исход']}</h3>
            <b>Вероятность модели: {best['Вероятность'] * 100:.1f}%</b><br>
            Уверенность: {best['Уверенность']}<br><span class="muted">{best['Основание']}</span></div>""",
            unsafe_allow_html=True,
        )
        alternatives = pd.DataFrame(selection.get("alternatives", []))
        if not alternatives.empty:
            st.write("### Альтернативы")
            st.dataframe(alternatives.style.format({"Вероятность": "{:.1%}"}), use_container_width=True, hide_index=True)
    else:
        st.info(selection.get("message", "Подходящий неочевидный исход не найден."))
    st.caption(selection.get("market_message", ""))
    st.write("### Лучший исход в каждой категории")
    outcomes = pd.DataFrame(result["outcomes"])
    st.dataframe(outcomes.style.format({"Вероятность": "{:.1%}"}), use_container_width=True, hide_index=True)

with tabs[2]:
    st.write("### Наиболее вероятные счета")
    score_df = pd.DataFrame(result["markets"]["top_scorelines"], columns=["Счёт", "Вероятность"])
    st.dataframe(score_df.style.format({"Вероятность": "{:.1%}"}), use_container_width=True, hide_index=True)
    market_rows = []
    for line in (1.5, 2.5, 3.5, 4.5):
        suffix = str(line).replace(".", "_")
        market_rows += [
            (f"Тотал больше {str(line).replace('.', ',')}", result["markets"][f"over_{suffix}"]),
            (f"Тотал меньше {str(line).replace('.', ',')}", result["markets"][f"under_{suffix}"]),
        ]
    market_rows += [("Обе забьют — да", result["markets"]["btts_yes"]), ("Обе забьют — нет", result["markets"]["btts_no"])]
    st.dataframe(pd.DataFrame(market_rows, columns=["Рынок", "Вероятность"]).style.format({"Вероятность": "{:.1%}"}), use_container_width=True, hide_index=True)

with tabs[3]:
    st.write("### Статус специализированных моделей")
    status_df = _status_frame(result.get("optional_model_status", []))
    st.dataframe(status_df.style.format({"MAE": "{:.3f}", "MAE среднего": "{:.3f}", "Изменение качества": "{:.1%}"}, na_rep="—"), use_container_width=True, hide_index=True)
    for title, section in (("Первый тайм", result["halftime"]), ("Второй тайм", result["second_half"]), ("Угловые", result["corners"]), ("Жёлтые карточки", result["cards"])):
        st.write(f"### {title}")
        if section.get("available"):
            table = _section_probability_rows(title, section, result["home"], result["away"])
            st.dataframe(table.style.format({"Вероятность": "{:.1%}"}), use_container_width=True, hide_index=True)
        else:
            st.info(section.get("reason", "Модель пока не прошла проверку и не используется."))

with tabs[4]:
    st.write("### Стартовые составы")
    squad_df = pd.DataFrame([
        {"Команда": result["home"], "Относительная сила": result["home_squad"]["relative_strength"], "Ключевых потерь": result["home_squad"]["missing_key_players"], "Пояснение": result["home_squad"]["explanation"]},
        {"Команда": result["away"], "Относительная сила": result["away_squad"]["relative_strength"], "Ключевых потерь": result["away_squad"]["missing_key_players"], "Пояснение": result["away_squad"]["explanation"]},
    ])
    st.dataframe(squad_df.style.format({"Относительная сила": "{:.1%}"}), use_container_width=True, hide_index=True)
    st.caption(result.get("lineup_source_message", ""))
    st.write("### Автоматически собранные данные")
    coverage_df = pd.DataFrame([
        ("Матчи с результатами", f"{coverage['results_matches']:,}"),
        ("Матчи с расширенной статистикой", f"{coverage['optional_matches']:,}"),
        ("Матчи со счётом тайма", f"{coverage['halftime_rows']:,}"),
        ("Матчи с xG", f"{coverage['xg_rows']:,}"),
        ("Матчи с угловыми", f"{coverage['corners_rows']:,}"),
        ("Матчи с карточками", f"{coverage['cards_rows']:,}"),
        ("Игроков в базе", f"{coverage['players']:,}"),
        ("Последняя дата расширенных данных", coverage.get("optional_last_date") or "—"),
        ("Источники", coverage.get("sources") or "Автоматический сбор ещё не дал данных"),
    ], columns=["Показатель", "Значение"])
    st.dataframe(coverage_df, use_container_width=True, hide_index=True)

with tabs[5]:
    st.write("### Вероятности отдельных моделей")
    labels = {"ml": "Машинное обучение", "dixon_coles": "Dixon–Coles", "elo": "Elo", "fifa": "FIFA", "recent_form": "Последние матчи с учётом соперников"}
    layer_rows = [{"Модель": labels.get(key, key), result["home"]: probs["home"], "Ничья": probs["draw"], result["away"]: probs["away"]} for key, probs in result["components"].items()]
    layer_rows.append({"Модель": "Итог после метамодели и калибровки", result["home"]: result["prob_home_win"], "Ничья": result["prob_draw"], result["away"]: result["prob_away_win"]})
    st.dataframe(pd.DataFrame(layer_rows).style.format({result["home"]: "{:.1%}", "Ничья": "{:.1%}", result["away"]: "{:.1%}"}), use_container_width=True, hide_index=True)
    st.write("### Влияние компонентов, изученное метамоделью")
    st.dataframe(model_component_importance(bundle).style.format({"Относительное влияние": "{:.1%}"}), use_container_width=True, hide_index=True)

with tabs[6]:
    st.write("### Хронологическое тестирование")
    st.dataframe(pd.DataFrame(bundle.metrics).style.format({"accuracy": "{:.1%}", "log_loss": "{:.4f}", "brier": "{:.4f}"}), use_container_width=True, hide_index=True)
    st.write(f"Температура калибровки: **{bundle.calibrator.temperature_:.3f}**")
    selector_status = result.get("non_obvious_selection", {}).get("selector_status", {})
    if selector_status:
        st.write("### Проверка селектора неочевидного исхода")
        selector_table = pd.DataFrame([{
            "Статус": "Активен" if selector_status.get("active") else "Не обучен",
            "Обучение": selector_status.get("training_rows", 0),
            "Калибровка": selector_status.get("calibration_rows", 0),
            "Проверка": selector_status.get("validation_rows", 0),
            "Log Loss": selector_status.get("log_loss"),
            "Brier": selector_status.get("brier"),
            "AUC": selector_status.get("auc"),
            "Пояснение": selector_status.get("reason", ""),
        }])
        st.dataframe(
            selector_table.style.format({"Log Loss": "{:.4f}", "Brier": "{:.4f}", "AUC": "{:.4f}"}, na_rep="—"),
            use_container_width=True, hide_index=True,
        )
    for path, title in ((WORLD_CUP_BACKTEST_PATH, "Проверка на прошлых чемпионатах мира"), (BACKTEST_PATH, "Последний отчёт обучения")):
        if Path(path).exists():
            try:
                table = pd.read_csv(path)
                if not table.empty:
                    st.write(f"### {title}")
                    st.dataframe(table.style.format({"accuracy": "{:.1%}", "log_loss": "{:.4f}", "brier": "{:.4f}"}), use_container_width=True, hide_index=True)
            except Exception:
                pass

with tabs[7]:
    store = _github_store()
    st.write("### Сохранение прогноза")
    if st.button("Сохранить этот прогноз в журнал"):
        try:
            journal_df = store.append(result) if store else append_local_prediction(result)
            st.success("Прогноз сохранён.")
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
        actual_home = a.number_input("Голы первой команды", 0, 20, 0)
        actual_away = b.number_input("Голы второй команды", 0, 20, 0)
        if st.button("Сохранить результат"):
            try:
                updated = store.update_result(selected_id, int(actual_home), int(actual_away)) if store else update_actual_result(selected_id, int(actual_home), int(actual_away))
                st.success("Результат сохранён.")
                st.dataframe(updated, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Не удалось сохранить результат: {exc}")
    metrics = journal_metrics(journal_df)
    if metrics["completed"]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Завершённых прогнозов", int(metrics["completed"]))
        c2.metric("Точность", f"{metrics['accuracy'] * 100:.1f}%")
        c3.metric("Log Loss", f"{metrics['log_loss']:.3f}")
        c4.metric("Brier", f"{metrics['brier']:.3f}")
