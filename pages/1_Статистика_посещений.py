from __future__ import annotations

import hmac
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from football_predictor.site_analytics import SiteAnalyticsClient, SiteAnalyticsError


st.set_page_config(page_title="Статистика посещений", page_icon="📊", layout="wide")
st.title("📊 Статистика посещений")
st.caption("Закрытая страница владельца сайта. IP-адреса и персональные данные не сохраняются.")


def _cfg() -> dict:
    try:
        return dict(st.secrets.get("site_analytics", {}))
    except Exception:
        return {}


cfg = _cfg()
client = SiteAnalyticsClient(
    url=cfg.get("supabase_url", ""),
    publishable_key=cfg.get("publishable_key", ""),
    secret_key=cfg.get("secret_key", ""),
)
admin_password = str(cfg.get("admin_password", ""))
timezone_name = str(cfg.get("timezone", "Europe/Berlin"))

if not client.public_enabled or not client.admin_enabled or not admin_password:
    st.error(
        "Статистика ещё не настроена. Добавьте раздел [site_analytics] в Streamlit Secrets "
        "по инструкции из файла ANALYTICS_SETUP.md."
    )
    st.stop()

if "analytics_admin_authenticated" not in st.session_state:
    st.session_state["analytics_admin_authenticated"] = False

if not st.session_state["analytics_admin_authenticated"]:
    with st.form("analytics_login"):
        entered = st.text_input("Пароль владельца", type="password")
        submitted = st.form_submit_button("Открыть статистику", type="primary")
    if submitted:
        if hmac.compare_digest(entered, admin_password):
            st.session_state["analytics_admin_authenticated"] = True
            st.rerun()
        else:
            st.error("Неверный пароль.")
    st.stop()

logout_col, refresh_col = st.columns([1, 5])
with logout_col:
    if st.button("Выйти"):
        st.session_state["analytics_admin_authenticated"] = False
        st.rerun()
with refresh_col:
    st.caption("Страница показывает активные браузерные сеансы, а не подтверждённых уникальных людей.")

try:
    summary = client.public_summary()
    raw_rows = client.fetch_sessions()
except SiteAnalyticsError as exc:
    st.error(f"Не удалось получить статистику: {exc}")
    st.stop()

try:
    local_tz = ZoneInfo(timezone_name)
except Exception:
    local_tz = timezone.utc
    timezone_name = "UTC"

frame = pd.DataFrame(raw_rows)
if frame.empty:
    st.info("Посещений пока нет.")
    st.stop()

for column in ("first_seen", "last_seen"):
    frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
frame = frame.dropna(subset=["first_seen", "last_seen"]).copy()
frame["first_seen_local"] = frame["first_seen"].dt.tz_convert(local_tz)
frame["last_seen_local"] = frame["last_seen"].dt.tz_convert(local_tz)
frame["duration_minutes"] = (
    (frame["last_seen"] - frame["first_seen"]).dt.total_seconds().clip(lower=0) / 60
)
now_utc = pd.Timestamp.now(tz="UTC")
frame["online"] = frame["last_seen"] >= now_utc - pd.Timedelta(minutes=2)

now_local = datetime.now(local_tz)
today_local = now_local.date()
last_7_cutoff = pd.Timestamp(now_local - pd.Timedelta(days=7))
visits_today = int((frame["first_seen_local"].dt.date == today_local).sum())
visits_7_days = int((frame["first_seen_local"] >= last_7_cutoff).sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Сейчас онлайн", summary.online_now)
m2.metric("Посещений сегодня", visits_today)
m3.metric("За последние 7 дней", visits_7_days)
m4.metric("Всего посещений", len(frame))

st.write("### Посещения по дням")
daily = (
    frame.assign(Дата=frame["first_seen_local"].dt.date)
    .groupby("Дата", as_index=False)
    .agg(Посещения=("session_id", "count"))
    .sort_values("Дата")
)
st.line_chart(daily.set_index("Дата")["Посещения"], height=320)
st.dataframe(daily.sort_values("Дата", ascending=False), use_container_width=True, hide_index=True)

st.write("### Последние посещения")
visible = frame.sort_values("first_seen", ascending=False).head(500).copy()
visible["Первый вход"] = visible["first_seen_local"].dt.strftime("%d.%m.%Y %H:%M:%S")
visible["Последняя активность"] = visible["last_seen_local"].dt.strftime("%d.%m.%Y %H:%M:%S")
visible["Длительность, мин"] = visible["duration_minutes"].round(1)
visible["Статус"] = visible["online"].map({True: "Онлайн", False: "Завершён"})
visible["Первая страница"] = visible["first_page"].fillna("main")
visible["Последняя страница"] = visible["last_page"].fillna("main")
st.dataframe(
    visible[[
        "Первый вход", "Последняя активность", "Длительность, мин",
        "Статус", "Первая страница", "Последняя страница",
    ]],
    use_container_width=True,
    hide_index=True,
)

csv_export = visible[[
    "session_id", "first_seen", "last_seen", "duration_minutes",
    "online", "first_page", "last_page",
]].to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Скачать историю CSV",
    data=csv_export,
    file_name="site_visit_history.csv",
    mime="text/csv",
)

st.caption(
    f"Часовой пояс отчёта: {timezone_name}. Один новый сеанс Streamlit считается одним посещением; "
    "обновление страницы, новый браузер или новая вкладка иногда создают новый сеанс."
)
