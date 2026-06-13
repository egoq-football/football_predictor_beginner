from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class MatchContext:
    stage: str = "group"  # group or knockout
    group_name: str = ""
    group_round: int = 1
    home_points: int = 0
    away_points: int = 0
    home_goal_difference: int = 0
    away_goal_difference: int = 0
    home_must_win: bool = False
    away_must_win: bool = False
    home_draw_enough: bool = False
    away_draw_enough: bool = False
    home_rotation_risk: float = 0.0
    away_rotation_risk: float = 0.0
    home_days_rest: int = 5
    away_days_rest: int = 5
    lineups_known: bool = False
    extra_time_possible: bool = False

    def normalized(self) -> "MatchContext":
        self.group_round = int(np.clip(self.group_round, 1, 3))
        self.home_rotation_risk = float(np.clip(self.home_rotation_risk, 0.0, 1.0))
        self.away_rotation_risk = float(np.clip(self.away_rotation_risk, 0.0, 1.0))
        self.home_days_rest = int(np.clip(self.home_days_rest, 1, 14))
        self.away_days_rest = int(np.clip(self.away_days_rest, 1, 14))
        self.extra_time_possible = bool(self.stage == "knockout")
        return self

    def feature_dict(self) -> dict[str, float]:
        self.normalized()
        return {
            "context_group_stage": 1.0 if self.stage == "group" else 0.0,
            "context_knockout": 1.0 if self.stage == "knockout" else 0.0,
            "context_group_round": float(self.group_round),
            "context_points_diff": float(self.home_points - self.away_points),
            "context_goal_difference_diff": float(self.home_goal_difference - self.away_goal_difference),
            "context_must_win_diff": float(self.home_must_win) - float(self.away_must_win),
            "context_draw_enough_diff": float(self.home_draw_enough) - float(self.away_draw_enough),
            "context_rotation_diff": float(self.away_rotation_risk - self.home_rotation_risk),
            "context_rest_diff": float(self.home_days_rest - self.away_days_rest),
            "context_lineups_known": float(self.lineups_known),
            "context_extra_time": float(self.extra_time_possible),
        }

    def as_dict(self) -> dict:
        return asdict(self.normalized())


def infer_group_motivation(context: MatchContext) -> MatchContext:
    """Apply conservative automatic suggestions without overriding explicit choices."""
    context.normalized()
    if context.stage != "group":
        return context
    if context.group_round == 3:
        if context.home_points <= 1 and not context.home_draw_enough:
            context.home_must_win = True
        if context.away_points <= 1 and not context.away_draw_enough:
            context.away_must_win = True
        if context.home_points >= 4 and not context.home_must_win:
            context.home_draw_enough = True
        if context.away_points >= 4 and not context.away_must_win:
            context.away_draw_enough = True
    return context
