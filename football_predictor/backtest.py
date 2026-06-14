from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from .features import build_current_builder, current_feature_frame
from .context import MatchContext
from .fifa_rankings import FifaRankingHistory
from .model import class_brier_scores, expected_calibration_error, multiclass_brier, train_bundle


def backtest_world_cups(table: pd.DataFrame, matches: pd.DataFrame, fifa: FifaRankingHistory, years=(2014, 2018, 2022)) -> pd.DataFrame:
    """Expanding-window World Cup tests. This is intentionally run offline."""
    rows = []
    for year in years:
        cup = matches[(matches["tournament"] == "FIFA World Cup") & (matches["date"].dt.year == year)].copy()
        if cup.empty:
            continue
        start = cup["date"].min()
        train_table = table[table["date"] < start].copy()
        if len(train_table) < 800:
            continue
        bundle = train_bundle(train_table)
        historical_matches = matches[matches["date"] < start].copy()
        builder = build_current_builder(historical_matches, fifa=fifa)
        probs = []
        y = []
        appearances = {}
        for _, match in cup.sort_values("date").iterrows():
            home = str(match["home_team"]); away = str(match["away_team"])
            h_apps = appearances.get(home, 0); a_apps = appearances.get(away, 0)
            stage = "group" if h_apps < 3 and a_apps < 3 else "knockout"
            context = MatchContext(stage=stage, group_round=min(max(h_apps, a_apps) + 1, 3), extra_time_possible=stage == "knockout")
            frame = current_feature_frame(builder, home, away, bool(match["neutral"]), match["date"], context)
            p, _ = bundle.predict(frame, home, away, bool(match["neutral"]))
            probs.append(p)
            hs = int(match["home_score"]); as_ = int(match["away_score"])
            y.append(2 if hs > as_ else 1 if hs == as_ else 0)
            builder.update(match)
            appearances[home] = h_apps + 1; appearances[away] = a_apps + 1
        probs_arr = np.vstack(probs); y_arr = np.array(y, dtype=int)
        class_brier = class_brier_scores(y_arr, probs_arr)
        rows.append({
            "tournament": f"FIFA World Cup {year}",
            "matches": len(y_arr),
            "accuracy": accuracy_score(y_arr, probs_arr.argmax(axis=1)),
            "log_loss": log_loss(y_arr, probs_arr, labels=[0, 1, 2]),
            "brier": multiclass_brier(y_arr, probs_arr),
            "ece": expected_calibration_error(y_arr, probs_arr),
            "away_brier": float(class_brier[0]),
            "draw_brier": float(class_brier[1]),
            "home_brier": float(class_brier[2]),
            "actual_draw_share": float(np.mean(y_arr == 1)),
            "predicted_draw_share": float(probs_arr[:, 1].mean()),
        })
    return pd.DataFrame(rows)
