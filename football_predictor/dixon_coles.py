from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .config import HOST_TEAMS


def _poisson_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-8)
    return float(np.exp(-lam) * lam**k / factorial(k))


def _tau(home_goals: int, away_goals: int, lam: float, mu: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lam * mu * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lam * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + mu * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


@dataclass
class DixonColesPrediction:
    home_lambda: float
    away_lambda: float
    matrix: np.ndarray

    @property
    def probabilities_1x2(self) -> np.ndarray:
        home = float(np.tril(self.matrix, -1).sum())
        draw = float(np.trace(self.matrix))
        away = float(np.triu(self.matrix, 1).sum())
        probs = np.array([away, draw, home], dtype=float)
        return probs / probs.sum()


class DixonColesModel:
    """Dixon–Coles score model with fast iterative attack/defence fitting."""

    def __init__(self, half_life_days: float = 900.0, shrinkage: float = 16.0) -> None:
        self.half_life_days = float(half_life_days)
        self.shrinkage = float(shrinkage)
        self.rho_ = -0.05
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}
        self.home_base_ = 1.40
        self.away_base_ = 1.05
        self.neutral_base_ = 1.18
        self.fitted_ = False

    def fit(self, matches: pd.DataFrame, max_matches: int = 14000) -> "DixonColesModel":
        data = matches.sort_values("date").copy()
        if max_matches and len(data) > max_matches:
            data = data.iloc[-max_matches:].copy()
        if len(data) < 100:
            raise ValueError("Для Dixon–Coles требуется минимум 100 матчей.")
        latest = pd.Timestamp(data["date"].max())
        ages = (latest - pd.to_datetime(data["date"])).dt.days.to_numpy(dtype=float)
        weights = 0.10 + 0.90 * np.power(0.5, ages / self.half_life_days)
        neutral_mask = data["neutral"].astype(bool).to_numpy()
        hg = data["home_score"].to_numpy(dtype=float)
        ag = data["away_score"].to_numpy(dtype=float)

        nonneutral = ~neutral_mask
        if nonneutral.any():
            self.home_base_ = float(np.average(hg[nonneutral], weights=weights[nonneutral]))
            self.away_base_ = float(np.average(ag[nonneutral], weights=weights[nonneutral]))
        if neutral_mask.any():
            neutral_goals = np.concatenate([hg[neutral_mask], ag[neutral_mask]])
            neutral_weights = np.concatenate([weights[neutral_mask], weights[neutral_mask]])
            self.neutral_base_ = float(np.average(neutral_goals, weights=neutral_weights))
        self.home_base_ = max(self.home_base_, 0.3)
        self.away_base_ = max(self.away_base_, 0.3)
        self.neutral_base_ = max(self.neutral_base_, 0.3)

        teams = sorted(set(data["home_team"]).union(data["away_team"]))
        team_to_idx = {team: idx for idx, team in enumerate(teams)}
        hidx = data["home_team"].map(team_to_idx).to_numpy(dtype=int)
        aidx = data["away_team"].map(team_to_idx).to_numpy(dtype=int)
        nteams = len(teams)
        attack = np.ones(nteams, dtype=float)
        defense = np.ones(nteams, dtype=float)
        home_base = np.where(neutral_mask, self.neutral_base_, self.home_base_)
        away_base = np.where(neutral_mask, self.neutral_base_, self.away_base_)
        exposure = np.bincount(hidx, weights=weights, minlength=nteams) + np.bincount(aidx, weights=weights, minlength=nteams)

        for _ in range(30):
            attack_obs = (
                np.bincount(hidx, weights=weights * hg, minlength=nteams)
                + np.bincount(aidx, weights=weights * ag, minlength=nteams)
            )
            attack_exp = (
                np.bincount(hidx, weights=weights * home_base * defense[aidx], minlength=nteams)
                + np.bincount(aidx, weights=weights * away_base * defense[hidx], minlength=nteams)
            )
            attack = (attack_obs + self.shrinkage) / np.maximum(attack_exp + self.shrinkage, 1e-8)
            attack = np.clip(attack, 0.25, 4.0)
            attack /= np.exp(np.mean(np.log(attack)))

            defense_obs = (
                np.bincount(hidx, weights=weights * ag, minlength=nteams)
                + np.bincount(aidx, weights=weights * hg, minlength=nteams)
            )
            defense_exp = (
                np.bincount(hidx, weights=weights * away_base * attack[aidx], minlength=nteams)
                + np.bincount(aidx, weights=weights * home_base * attack[hidx], minlength=nteams)
            )
            defense = (defense_obs + self.shrinkage) / np.maximum(defense_exp + self.shrinkage, 1e-8)
            defense = np.clip(defense, 0.25, 4.0)

        self.attack_ = {team: float(attack[idx]) for team, idx in team_to_idx.items()}
        self.defense_ = {team: float(defense[idx]) for team, idx in team_to_idx.items()}
        lam = home_base * attack[hidx] * defense[aidx]
        mu = away_base * attack[aidx] * defense[hidx]
        low = (hg <= 1) & (ag <= 1)
        if low.sum() >= 50:
            hgi = hg.astype(int); agi = ag.astype(int)
            def objective(rho: float) -> float:
                values = np.ones(low.sum(), dtype=float)
                h = hgi[low]; a = agi[low]; l = lam[low]; m = mu[low]
                m00 = (h == 0) & (a == 0); m01 = (h == 0) & (a == 1)
                m10 = (h == 1) & (a == 0); m11 = (h == 1) & (a == 1)
                values[m00] = 1.0 - l[m00] * m[m00] * rho
                values[m01] = 1.0 + l[m01] * rho
                values[m10] = 1.0 + m[m10] * rho
                values[m11] = 1.0 - rho
                if np.any(values <= 1e-8):
                    return 1e9
                return float(-np.sum(weights[low] * np.log(values)))
            result = minimize_scalar(objective, bounds=(-0.18, 0.18), method="bounded")
            if result.success:
                self.rho_ = float(result.x)
        self.fitted_ = True
        self.optimization_success_ = True
        self.optimization_message_ = "vectorized weighted Poisson + Dixon–Coles rho"
        return self

    def expected_goals(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        ah = self.attack_.get(home, 1.0); aa = self.attack_.get(away, 1.0)
        dh = self.defense_.get(home, 1.0); da = self.defense_.get(away, 1.0)
        if neutral:
            home_base = away_base = self.neutral_base_
        else:
            home_base = self.home_base_; away_base = self.away_base_
        host_home = 1.04 if home in HOST_TEAMS else 1.0
        host_away = 1.04 if away in HOST_TEAMS else 1.0
        lam = home_base * ah * da * host_home / host_away
        mu = away_base * aa * dh * host_away / host_home
        return float(np.clip(lam, 0.05, 6.0)), float(np.clip(mu, 0.05, 6.0))

    def predict(self, home: str, away: str, neutral: bool = True, max_goals: int = 8) -> DixonColesPrediction:
        lam, mu = self.expected_goals(home, away, neutral)
        matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                matrix[i, j] = _poisson_pmf(i, lam) * _poisson_pmf(j, mu) * _tau(i, j, lam, mu, self.rho_)
        matrix = np.clip(matrix, 0.0, None)
        matrix /= matrix.sum()
        return DixonColesPrediction(lam, mu, matrix)


def score_markets(prediction: DixonColesPrediction) -> dict[str, float | list[tuple[str, float]]]:
    matrix = prediction.matrix
    max_goals = matrix.shape[0] - 1
    total = np.fromfunction(lambda i, j: i + j, matrix.shape)
    btts = np.fromfunction(lambda i, j: (i > 0) & (j > 0), matrix.shape).astype(bool)
    markets: dict[str, float | list[tuple[str, float]]] = {
        "home_win": float(np.tril(matrix, -1).sum()), "draw": float(np.trace(matrix)), "away_win": float(np.triu(matrix, 1).sum()),
        "over_0_5": float(matrix[total > 0.5].sum()),
        "over_1_5": float(matrix[total > 1.5].sum()), "under_1_5": float(matrix[total < 1.5].sum()),
        "over_2_5": float(matrix[total > 2.5].sum()), "under_2_5": float(matrix[total < 2.5].sum()),
        "over_3_5": float(matrix[total > 3.5].sum()), "under_3_5": float(matrix[total < 3.5].sum()),
        "over_4_5": float(matrix[total > 4.5].sum()), "under_4_5": float(matrix[total < 4.5].sum()),
        "btts_yes": float(matrix[btts].sum()), "btts_no": float(matrix[~btts].sum()),
        "home_clean_sheet": float(matrix[:, 0].sum()), "away_clean_sheet": float(matrix[0, :].sum()),
    }
    home_goal_probs = matrix.sum(axis=1)
    away_goal_probs = matrix.sum(axis=0)
    home_goal_values = np.arange(matrix.shape[0], dtype=float)
    away_goal_values = np.arange(matrix.shape[1], dtype=float)
    for line in (0.5, 1.5, 2.5, 3.5):
        suffix = str(line).replace(".", "_")
        markets[f"home_over_{suffix}"] = float(home_goal_probs[home_goal_values > line].sum())
        markets[f"home_under_{suffix}"] = float(home_goal_probs[home_goal_values < line].sum())
        markets[f"away_over_{suffix}"] = float(away_goal_probs[away_goal_values > line].sum())
        markets[f"away_under_{suffix}"] = float(away_goal_probs[away_goal_values < line].sum())
    scores = [(f"{i}:{j}", float(matrix[i, j])) for i in range(max_goals + 1) for j in range(max_goals + 1)]
    markets["top_scorelines"] = sorted(scores, key=lambda x: x[1], reverse=True)[:12]
    return markets
