from __future__ import annotations

import pandas as pd

from .features import TeamState
from .fifa_ranking import FifaRankingInfo


def _result_label(gf: int, ga: int) -> str:
    if gf > ga:
        return "Победа"
    if gf == ga:
        return "Ничья"
    return "Поражение"


def _points(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def team_recent_matches(df: pd.DataFrame, team: str, n: int = 10) -> pd.DataFrame:
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    part = df.loc[mask].sort_values("date", ascending=False).head(n).copy()
    rows = []
    for _, m in part.iterrows():
        is_home = m["home_team"] == team
        opponent = m["away_team"] if is_home else m["home_team"]
        gf = int(m["home_score"] if is_home else m["away_score"])
        ga = int(m["away_score"] if is_home else m["home_score"])
        rows.append({
            "Дата": pd.to_datetime(m["date"]).date().isoformat(),
            "Турнир": str(m["tournament"]),
            "Соперник": opponent,
            "Где": "дом" if is_home and not bool(m["neutral"]) else ("гости" if not is_home and not bool(m["neutral"]) else "нейтр."),
            "Счет": f"{gf}:{ga}",
            "Исход": _result_label(gf, ga),
            "Очки": _points(gf, ga),
        })
    return pd.DataFrame(rows)


def h2h_matches(df: pd.DataFrame, team1: str, team2: str, n: int = 10) -> pd.DataFrame:
    mask = ((df["home_team"] == team1) & (df["away_team"] == team2)) | ((df["home_team"] == team2) & (df["away_team"] == team1))
    part = df.loc[mask].sort_values("date", ascending=False).head(n).copy()
    rows = []
    for _, m in part.iterrows():
        rows.append({
            "Дата": pd.to_datetime(m["date"]).date().isoformat(),
            "Турнир": str(m["tournament"]),
            "Матч": f"{m['home_team']} — {m['away_team']}",
            "Счет": f"{int(m['home_score'])}:{int(m['away_score'])}",
            "Поле": "нейтральное" if bool(m["neutral"]) else str(m["country"]),
        })
    return pd.DataFrame(rows)


def team_summary(
    team: str,
    state: TeamState,
    fifa: FifaRankingInfo | None = None,
) -> dict[str, float | str | int]:
    points_5 = list(state.recent_points)[-5:]
    points_10 = list(state.recent_points)[-10:]
    gf_5 = list(state.recent_goals_for)[-5:]
    ga_5 = list(state.recent_goals_against)[-5:]
    gf_10 = list(state.recent_goals_for)[-10:]
    ga_10 = list(state.recent_goals_against)[-10:]
    opponent_elo_5 = list(state.recent_opponent_elo)[-5:]
    strength_points_5 = list(state.recent_strength_points)[-5:]
    performance_5 = list(state.recent_performance)[-5:]

    def avg(values, default=0.0):
        return sum(values) / len(values) if values else default

    def rate(values, value):
        return sum(1 for x in values if x == value) / len(values) if values else 0.0

    return {
        "Команда": team,
        "Рейтинг FIFA": fifa.rank if fifa else "нет данных",
        "Очки FIFA": round(float(fifa.points), 2) if fifa else "нет данных",
        "Внутренний рейтинг силы (Elo)": round(float(state.elo), 1),
        "Матчей с 2010": int(state.matches),
        "Очки/матч за 5": round(avg(points_5, 1.0), 2),
        "Очки/матч за 10": round(avg(points_10, 1.0), 2),
        "Победы за 5": f"{rate(points_5, 3) * 100:.0f}%",
        "Ничьи за 5": f"{rate(points_5, 1) * 100:.0f}%",
        "Голы заб. за 5": round(avg(gf_5, 1.25), 2),
        "Голы проп. за 5": round(avg(ga_5, 1.25), 2),
        "Голы заб. за 10": round(avg(gf_10, 1.25), 2),
        "Голы проп. за 10": round(avg(ga_10, 1.25), 2),
        "Средняя сила соперников за 5": round(avg(opponent_elo_5, 1500.0), 1),
        "Очки/матч за 5 с учётом силы соперников": round(avg(strength_points_5, 1.0), 2),
        "Результат выше/ниже ожидания Elo за 5": round(avg(performance_5, 0.0), 3),
    }


def form_chart_data(df: pd.DataFrame, team1: str, team2: str, n: int = 10) -> pd.DataFrame:
    rows = []
    for team in [team1, team2]:
        recent = team_recent_matches(df, team, n=n).iloc[::-1].reset_index(drop=True)
        total = 0
        for idx, row in recent.iterrows():
            total += int(row["Очки"])
            rows.append({"Матч №": idx + 1, "Команда": team, "Накопленные очки": total})
    if not rows:
        return pd.DataFrame(columns=["Матч №", "Команда", "Накопленные очки"])
    return pd.DataFrame(rows)
