from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import HOST_TEAMS, TOURNAMENT_IMPORTANCE
from .context import MatchContext
from .elo import BASE_ELO, HOME_ADVANTAGE, expected_score, result_score, update_elo
from .fifa_rankings import FifaRankingHistory


def tournament_importance(name: str) -> float:
    name = str(name)
    if name in TOURNAMENT_IMPORTANCE:
        return float(TOURNAMENT_IMPORTANCE[name])
    lowered = name.lower()
    if "world cup" in lowered and "qualification" in lowered:
        return 0.90
    if "world cup" in lowered:
        return 1.00
    if "qualification" in lowered:
        return 0.76
    if "friendly" in lowered:
        return 0.42
    if "nations league" in lowered:
        return 0.70
    return 0.62


@dataclass
class TeamState:
    elo: float = BASE_ELO
    matches: int = 0
    last_date: pd.Timestamp | None = None
    points: deque = field(default_factory=lambda: deque(maxlen=20))
    goal_diff: deque = field(default_factory=lambda: deque(maxlen=20))
    goals_for: deque = field(default_factory=lambda: deque(maxlen=20))
    goals_against: deque = field(default_factory=lambda: deque(maxlen=20))
    opponent_elo: deque = field(default_factory=lambda: deque(maxlen=20))
    performance_vs_expectation: deque = field(default_factory=lambda: deque(maxlen=20))
    adjusted_points: deque = field(default_factory=lambda: deque(maxlen=20))
    adjusted_goal_diff: deque = field(default_factory=lambda: deque(maxlen=20))
    xg_for: deque = field(default_factory=lambda: deque(maxlen=20))
    xg_against: deque = field(default_factory=lambda: deque(maxlen=20))
    shots_for: deque = field(default_factory=lambda: deque(maxlen=20))
    shots_against: deque = field(default_factory=lambda: deque(maxlen=20))
    shots_on_target_for: deque = field(default_factory=lambda: deque(maxlen=20))
    shots_on_target_against: deque = field(default_factory=lambda: deque(maxlen=20))
    corners_for: deque = field(default_factory=lambda: deque(maxlen=20))
    corners_against: deque = field(default_factory=lambda: deque(maxlen=20))
    yellow_cards: deque = field(default_factory=lambda: deque(maxlen=20))
    possession: deque = field(default_factory=lambda: deque(maxlen=20))
    ppda: deque = field(default_factory=lambda: deque(maxlen=20))
    ht_goals_for: deque = field(default_factory=lambda: deque(maxlen=20))
    ht_goals_against: deque = field(default_factory=lambda: deque(maxlen=20))


@dataclass
class WorldCupState:
    appearances: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    points: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    goal_difference: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _avg(values, n: int, default: float = 0.0) -> float:
    items = [float(x) for x in list(values)[-n:] if pd.notna(x)]
    return float(np.mean(items)) if items else float(default)


def _points(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def _result_target(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 2
    if home_score == away_score:
        return 1
    return 0


def _optional_value(row: pd.Series, name: str) -> float:
    value = row.get(name, np.nan)
    return float(value) if pd.notna(value) else np.nan


FEATURE_COLUMNS = [
    "elo_diff", "abs_elo_diff", "home_elo", "away_elo",
    "fifa_points_diff", "abs_fifa_points_diff", "fifa_available",
    "home_matches", "away_matches", "experience_diff",
    "form_points_diff_3", "form_points_diff_5", "form_points_diff_10",
    "adjusted_form_diff_5", "performance_diff_5", "opponent_elo_diff_5",
    "goal_diff_form_diff_5", "goal_diff_form_diff_10", "adjusted_goal_diff_5",
    "home_gf_5", "home_ga_5", "away_gf_5", "away_ga_5",
    "home_gf_10", "home_ga_10", "away_gf_10", "away_ga_10",
    "attack_diff_5", "defense_diff_5", "rest_days_diff",
    "h2h_goal_diff", "h2h_points_diff", "h2h_matches",
    "neutral", "home_host", "away_host", "importance", "is_friendly", "is_world_cup",
    "xg_diff_5", "xga_diff_5", "shots_diff_5", "shots_on_target_diff_5",
    "corners_diff_5", "cards_diff_5", "possession_diff_5", "ppda_edge_5",
    "optional_stats_available", "lineup_strength_diff", "lineup_data_available",
    "context_group_stage", "context_knockout", "context_group_round",
    "context_points_diff", "context_goal_difference_diff", "context_must_win_diff",
    "context_draw_enough_diff", "context_rotation_diff", "context_rest_diff",
    "context_lineups_known", "context_extra_time",
]


class SequentialFeatureBuilder:
    def __init__(self, fifa: FifaRankingHistory | None = None) -> None:
        self.fifa = fifa
        self.states: dict[str, TeamState] = defaultdict(TeamState)
        self.h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))
        self.world_cups: dict[int, WorldCupState] = defaultdict(WorldCupState)

    def _historical_context(self, row: pd.Series) -> MatchContext:
        tournament = str(row["tournament"])
        if tournament != "FIFA World Cup":
            return MatchContext(stage="group", group_round=1)
        year = int(pd.Timestamp(row["date"]).year)
        wc = self.world_cups[year]
        home = str(row["home_team"])
        away = str(row["away_team"])
        # The first three appearances by a team are group-stage games.
        home_apps = wc.appearances[home]
        away_apps = wc.appearances[away]
        group_stage = home_apps < 3 and away_apps < 3
        if group_stage:
            round_no = max(home_apps, away_apps) + 1
            context = MatchContext(
                stage="group",
                group_round=round_no,
                home_points=wc.points[home],
                away_points=wc.points[away],
                home_goal_difference=wc.goal_difference[home],
                away_goal_difference=wc.goal_difference[away],
            )
            if round_no == 3:
                context.home_must_win = wc.points[home] <= 1
                context.away_must_win = wc.points[away] <= 1
                context.home_draw_enough = wc.points[home] >= 4
                context.away_draw_enough = wc.points[away] >= 4
            return context
        return MatchContext(stage="knockout", group_round=3, extra_time_possible=True)

    def make_row(
        self,
        home: str,
        away: str,
        neutral: bool,
        match_date: pd.Timestamp,
        tournament: str,
        context: MatchContext | None = None,
        lineup_strength_home: float | None = None,
        lineup_strength_away: float | None = None,
    ) -> dict[str, float]:
        hs = self.states[home]
        as_ = self.states[away]
        date = pd.Timestamp(match_date)
        home_adv = 0.0 if neutral else HOME_ADVANTAGE
        elo_diff = hs.elo + home_adv - as_.elo

        home_fifa = self.fifa.lookup(home, date) if self.fifa else None
        away_fifa = self.fifa.lookup(away, date) if self.fifa else None
        fifa_available = float(home_fifa is not None and away_fifa is not None)
        fifa_diff = float(home_fifa.points - away_fifa.points) if fifa_available else 0.0

        pair = tuple(sorted((home, away)))
        h2h_records = list(self.h2h[pair])
        h2h_gd: list[float] = []
        h2h_pd: list[float] = []
        for first_team, gd_first, points_first in h2h_records:
            if home == first_team:
                h2h_gd.append(float(gd_first))
                h2h_pd.append(float(points_first - (3 if points_first == 0 else 0 if points_first == 3 else 1)))
            else:
                h2h_gd.append(float(-gd_first))
                h2h_pd.append(float((3 if points_first == 0 else 0 if points_first == 3 else 1) - points_first))

        home_rest = 7.0 if hs.last_date is None else float(np.clip((date - hs.last_date).days, 1, 30))
        away_rest = 7.0 if as_.last_date is None else float(np.clip((date - as_.last_date).days, 1, 30))

        optional_available = float(
            len(hs.xg_for) >= 3 and len(as_.xg_for) >= 3
        )
        lineup_available = float(lineup_strength_home is not None and lineup_strength_away is not None)
        lineup_diff = float((lineup_strength_home or 0.0) - (lineup_strength_away or 0.0))

        ctx = context or MatchContext()
        ctx_features = ctx.feature_dict()

        home_gf5 = _avg(hs.goals_for, 5, 1.2)
        home_ga5 = _avg(hs.goals_against, 5, 1.2)
        away_gf5 = _avg(as_.goals_for, 5, 1.2)
        away_ga5 = _avg(as_.goals_against, 5, 1.2)
        home_gf10 = _avg(hs.goals_for, 10, 1.2)
        home_ga10 = _avg(hs.goals_against, 10, 1.2)
        away_gf10 = _avg(as_.goals_for, 10, 1.2)
        away_ga10 = _avg(as_.goals_against, 10, 1.2)

        values: dict[str, float] = {
            "elo_diff": float(elo_diff),
            "abs_elo_diff": float(abs(elo_diff)),
            "home_elo": float(hs.elo),
            "away_elo": float(as_.elo),
            "fifa_points_diff": fifa_diff,
            "abs_fifa_points_diff": abs(fifa_diff),
            "fifa_available": fifa_available,
            "home_matches": float(hs.matches),
            "away_matches": float(as_.matches),
            "experience_diff": float(hs.matches - as_.matches),
            "form_points_diff_3": _avg(hs.points, 3, 1.0) - _avg(as_.points, 3, 1.0),
            "form_points_diff_5": _avg(hs.points, 5, 1.0) - _avg(as_.points, 5, 1.0),
            "form_points_diff_10": _avg(hs.points, 10, 1.0) - _avg(as_.points, 10, 1.0),
            "adjusted_form_diff_5": _avg(hs.adjusted_points, 5, 1.0) - _avg(as_.adjusted_points, 5, 1.0),
            "performance_diff_5": _avg(hs.performance_vs_expectation, 5, 0.0) - _avg(as_.performance_vs_expectation, 5, 0.0),
            "opponent_elo_diff_5": _avg(hs.opponent_elo, 5, BASE_ELO) - _avg(as_.opponent_elo, 5, BASE_ELO),
            "goal_diff_form_diff_5": _avg(hs.goal_diff, 5, 0.0) - _avg(as_.goal_diff, 5, 0.0),
            "goal_diff_form_diff_10": _avg(hs.goal_diff, 10, 0.0) - _avg(as_.goal_diff, 10, 0.0),
            "adjusted_goal_diff_5": _avg(hs.adjusted_goal_diff, 5, 0.0) - _avg(as_.adjusted_goal_diff, 5, 0.0),
            "home_gf_5": home_gf5,
            "home_ga_5": home_ga5,
            "away_gf_5": away_gf5,
            "away_ga_5": away_ga5,
            "home_gf_10": home_gf10,
            "home_ga_10": home_ga10,
            "away_gf_10": away_gf10,
            "away_ga_10": away_ga10,
            "attack_diff_5": home_gf5 - away_gf5,
            "defense_diff_5": away_ga5 - home_ga5,
            "rest_days_diff": home_rest - away_rest,
            "h2h_goal_diff": float(np.mean(h2h_gd)) if h2h_gd else 0.0,
            "h2h_points_diff": float(np.mean(h2h_pd)) if h2h_pd else 0.0,
            "h2h_matches": float(len(h2h_records)),
            "neutral": float(neutral),
            "home_host": float(home in HOST_TEAMS),
            "away_host": float(away in HOST_TEAMS),
            "importance": tournament_importance(tournament),
            "is_friendly": float(str(tournament).lower() == "friendly"),
            "is_world_cup": float(tournament == "FIFA World Cup"),
            "xg_diff_5": _avg(hs.xg_for, 5, 0.0) - _avg(as_.xg_for, 5, 0.0),
            "xga_diff_5": _avg(as_.xg_against, 5, 0.0) - _avg(hs.xg_against, 5, 0.0),
            "shots_diff_5": _avg(hs.shots_for, 5, 0.0) - _avg(as_.shots_for, 5, 0.0),
            "shots_on_target_diff_5": _avg(hs.shots_on_target_for, 5, 0.0) - _avg(as_.shots_on_target_for, 5, 0.0),
            "corners_diff_5": _avg(hs.corners_for, 5, 0.0) - _avg(as_.corners_for, 5, 0.0),
            "cards_diff_5": _avg(as_.yellow_cards, 5, 0.0) - _avg(hs.yellow_cards, 5, 0.0),
            "possession_diff_5": _avg(hs.possession, 5, 0.0) - _avg(as_.possession, 5, 0.0),
            "ppda_edge_5": _avg(as_.ppda, 5, 0.0) - _avg(hs.ppda, 5, 0.0),
            "optional_stats_available": optional_available,
            "lineup_strength_diff": lineup_diff,
            "lineup_data_available": lineup_available,
        }
        values.update(ctx_features)
        return values

    def update(self, row: pd.Series) -> None:
        home = str(row["home_team"])
        away = str(row["away_team"])
        hs = self.states[home]
        as_ = self.states[away]
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        neutral = bool(row["neutral"])
        importance = tournament_importance(str(row["tournament"]))

        home_expected = expected_score(hs.elo + (0.0 if neutral else HOME_ADVANTAGE), as_.elo)
        away_expected = 1.0 - home_expected
        home_actual = result_score(home_score, away_score)
        away_actual = 1.0 - home_actual

        # Opponent-strength adjustment: performance above expectation receives
        # the strongest signal, while raw points remain available separately.
        home_points = _points(home_score, away_score)
        away_points = _points(away_score, home_score)
        home_adj_points = home_points + (home_actual - home_expected) * 1.25
        away_adj_points = away_points + (away_actual - away_expected) * 1.25
        home_gd = home_score - away_score
        away_gd = -home_gd
        home_opp_factor = np.clip(as_.elo / BASE_ELO, 0.70, 1.35)
        away_opp_factor = np.clip(hs.elo / BASE_ELO, 0.70, 1.35)

        hs.points.append(home_points)
        as_.points.append(away_points)
        hs.goal_diff.append(home_gd)
        as_.goal_diff.append(away_gd)
        hs.goals_for.append(home_score)
        hs.goals_against.append(away_score)
        as_.goals_for.append(away_score)
        as_.goals_against.append(home_score)
        hs.opponent_elo.append(as_.elo)
        as_.opponent_elo.append(hs.elo)
        hs.performance_vs_expectation.append(home_actual - home_expected)
        as_.performance_vs_expectation.append(away_actual - away_expected)
        hs.adjusted_points.append(home_adj_points)
        as_.adjusted_points.append(away_adj_points)
        hs.adjusted_goal_diff.append(home_gd * home_opp_factor)
        as_.adjusted_goal_diff.append(away_gd * away_opp_factor)

        # Optional event statistics.
        optional_pairs = [
            ("xg", hs.xg_for, hs.xg_against, as_.xg_for, as_.xg_against),
            ("shots", hs.shots_for, hs.shots_against, as_.shots_for, as_.shots_against),
            ("shots_on_target", hs.shots_on_target_for, hs.shots_on_target_against, as_.shots_on_target_for, as_.shots_on_target_against),
            ("corners", hs.corners_for, hs.corners_against, as_.corners_for, as_.corners_against),
        ]
        for base, h_for, h_against, a_for, a_against in optional_pairs:
            hv = _optional_value(row, f"home_{base}")
            av = _optional_value(row, f"away_{base}")
            if pd.notna(hv) and pd.notna(av):
                h_for.append(hv); h_against.append(av); a_for.append(av); a_against.append(hv)
        hy = _optional_value(row, "home_yellow_cards")
        ay = _optional_value(row, "away_yellow_cards")
        if pd.notna(hy): hs.yellow_cards.append(hy)
        if pd.notna(ay): as_.yellow_cards.append(ay)
        hp = _optional_value(row, "home_possession")
        ap = _optional_value(row, "away_possession")
        if pd.notna(hp): hs.possession.append(hp)
        if pd.notna(ap): as_.possession.append(ap)
        hppda = _optional_value(row, "home_ppda")
        appda = _optional_value(row, "away_ppda")
        if pd.notna(hppda): hs.ppda.append(hppda)
        if pd.notna(appda): as_.ppda.append(appda)
        hht = _optional_value(row, "home_ht_score")
        aht = _optional_value(row, "away_ht_score")
        if pd.notna(hht) and pd.notna(aht):
            hs.ht_goals_for.append(hht); hs.ht_goals_against.append(aht)
            as_.ht_goals_for.append(aht); as_.ht_goals_against.append(hht)

        first_team = min(home, away)
        if home == first_team:
            first_gd, first_points = home_gd, home_points
        else:
            first_gd, first_points = away_gd, away_points
        self.h2h[tuple(sorted((home, away)))].append((first_team, first_gd, first_points))

        hs.elo, as_.elo = update_elo(
            hs.elo, as_.elo, home_score, away_score, neutral, importance=importance
        )
        hs.matches += 1
        as_.matches += 1
        hs.last_date = pd.Timestamp(row["date"])
        as_.last_date = pd.Timestamp(row["date"])

        if str(row["tournament"]) == "FIFA World Cup":
            year = int(pd.Timestamp(row["date"]).year)
            wc = self.world_cups[year]
            if wc.appearances[home] < 3 and wc.appearances[away] < 3:
                wc.points[home] += home_points
                wc.points[away] += away_points
                wc.goal_difference[home] += home_gd
                wc.goal_difference[away] += away_gd
            wc.appearances[home] += 1
            wc.appearances[away] += 1


def build_training_table(
    matches: pd.DataFrame,
    fifa: FifaRankingHistory | None = None,
) -> tuple[pd.DataFrame, SequentialFeatureBuilder]:
    builder = SequentialFeatureBuilder(fifa=fifa)
    rows: list[dict[str, Any]] = []
    for _, match in matches.sort_values("date").iterrows():
        context = builder._historical_context(match)
        feature_row = builder.make_row(
            str(match["home_team"]), str(match["away_team"]), bool(match["neutral"]),
            pd.Timestamp(match["date"]), str(match["tournament"]), context=context,
        )
        feature_row.update({
            "date": pd.Timestamp(match["date"]),
            "home_team": str(match["home_team"]),
            "away_team": str(match["away_team"]),
            "home_score": int(match["home_score"]),
            "away_score": int(match["away_score"]),
            "target": _result_target(int(match["home_score"]), int(match["away_score"])),
            "tournament": str(match["tournament"]),
        })
        for optional_target in [
            "home_xg", "away_xg", "home_ht_score", "away_ht_score",
            "home_corners", "away_corners",
            "home_yellow_cards", "away_yellow_cards",
        ]:
            feature_row[optional_target] = match.get(optional_target, np.nan)
        hht = match.get("home_ht_score", np.nan)
        aht = match.get("away_ht_score", np.nan)
        feature_row["home_second_half_score"] = int(match["home_score"]) - float(hht) if pd.notna(hht) else np.nan
        feature_row["away_second_half_score"] = int(match["away_score"]) - float(aht) if pd.notna(aht) else np.nan
        rows.append(feature_row)
        builder.update(match)
    table = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in table:
            table[col] = 0.0
        table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0.0)
    return table, builder


def build_current_builder(matches: pd.DataFrame, fifa: FifaRankingHistory | None = None) -> SequentialFeatureBuilder:
    builder = SequentialFeatureBuilder(fifa=fifa)
    for _, match in matches.sort_values("date").iterrows():
        builder.update(match)
    return builder


def current_feature_frame(
    builder: SequentialFeatureBuilder,
    home: str,
    away: str,
    neutral: bool,
    match_date: str | pd.Timestamp,
    context: MatchContext,
    lineup_strength_home: float | None = None,
    lineup_strength_away: float | None = None,
) -> pd.DataFrame:
    row = builder.make_row(
        home, away, neutral, pd.Timestamp(match_date), "FIFA World Cup",
        context=context,
        lineup_strength_home=lineup_strength_home,
        lineup_strength_away=lineup_strength_away,
    )
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def recent_form_probabilities(row: pd.Series) -> np.ndarray:
    edge = (
        float(row["adjusted_form_diff_5"]) * 0.55
        + float(row["performance_diff_5"]) * 2.0
        + float(row["adjusted_goal_diff_5"]) * 0.24
        + float(row["form_points_diff_5"]) * 0.18
        + float(row["opponent_elo_diff_5"]) / 600.0
        + float(row["context_must_win_diff"]) * 0.10
        + float(row["context_rotation_diff"]) * 0.20
    )
    non_draw_home = 1.0 / (1.0 + np.exp(-edge))
    draw = float(np.clip(0.27 - abs(edge) * 0.055, 0.16, 0.31))
    home = (1.0 - draw) * non_draw_home
    away = (1.0 - draw) * (1.0 - non_draw_home)
    return np.array([away, draw, home], dtype=float)


def fifa_probabilities(row: pd.Series) -> np.ndarray:
    if float(row["fifa_available"]) < 0.5:
        return np.array([0.33, 0.34, 0.33], dtype=float)
    edge = float(row["fifa_points_diff"]) / 210.0
    non_draw_home = 1.0 / (1.0 + np.exp(-edge))
    draw = float(np.clip(0.27 - abs(edge) * 0.04, 0.17, 0.30))
    home = (1.0 - draw) * non_draw_home
    away = (1.0 - draw) * (1.0 - non_draw_home)
    return np.array([away, draw, home], dtype=float)
