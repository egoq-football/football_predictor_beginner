from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import BACKTEST_PATH, MODEL_BUNDLE_PATH, MODEL_META_PATH
from .data_loader import (
    download_fifa_history,
    download_results,
    load_optional_stats,
    load_results,
    merge_optional_stats,
)
from .features import build_current_builder, build_training_table
from .fifa_rankings import FifaRankingHistory, download_current_ranking
from .model import WorldCupModelBundle, load_bundle, save_bundle, train_bundle


def build_dataset(download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, FifaRankingHistory]:
    if download:
        download_results()
        try:
            download_fifa_history()
        except Exception:
            pass
        try:
            download_current_ranking()
        except Exception:
            pass
    results = load_results()
    optional = load_optional_stats()
    merged = merge_optional_stats(results, optional)
    fifa = FifaRankingHistory()
    table, _ = build_training_table(merged, fifa=fifa)
    return merged, table, fifa


def _primary_log_loss(bundle: WorldCupModelBundle) -> float:
    for row in bundle.metrics:
        if row.get("model") == "Итоговый ансамбль":
            return float(row.get("log_loss", 999.0))
    return 999.0


def train_and_maybe_promote(
    download: bool = False,
    force: bool = False,
    min_improvement: float = 0.002,
) -> tuple[WorldCupModelBundle, bool, str]:
    matches, table, _ = build_dataset(download=download)
    candidate = train_bundle(table)
    promote = force or not Path(MODEL_BUNDLE_PATH).exists()
    reason = "первая модель" if promote else ""

    if not promote:
        try:
            current = load_bundle(MODEL_BUNDLE_PATH)
            current_loss = _primary_log_loss(current)
            candidate_loss = _primary_log_loss(candidate)
            if candidate_loss <= current_loss - min_improvement:
                promote = True
                reason = f"Log Loss улучшился: {current_loss:.4f} → {candidate_loss:.4f}"
            else:
                reason = f"Кандидат не улучшил Log Loss: текущий {current_loss:.4f}, кандидат {candidate_loss:.4f}"
        except Exception:
            promote = True
            reason = "текущая модель несовместима"

    if promote:
        save_bundle(candidate, MODEL_BUNDLE_PATH)
        selected = candidate
    else:
        selected = load_bundle(MODEL_BUNDLE_PATH)

    Path(MODEL_META_PATH).parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "promoted": promote,
        "reason": reason,
        "candidate_train_end": candidate.train_end_date,
        "candidate_training_matches": candidate.training_matches,
        "candidate_metrics": candidate.metrics,
        "selected_version": selected.version,
    }
    Path(MODEL_META_PATH).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(candidate.metrics).to_csv(BACKTEST_PATH, index=False)
    return selected, promote, reason


def load_runtime(retrain_if_missing: bool = True):
    results = load_results()
    optional = load_optional_stats()
    matches = merge_optional_stats(results, optional)
    fifa = FifaRankingHistory()
    if Path(MODEL_BUNDLE_PATH).exists():
        try:
            bundle = load_bundle(MODEL_BUNDLE_PATH)
        except Exception:
            if not retrain_if_missing:
                raise
            _, table, _ = build_dataset(download=False)
            bundle = train_bundle(table)
            save_bundle(bundle, MODEL_BUNDLE_PATH)
    elif retrain_if_missing:
        _, table, _ = build_dataset(download=False)
        bundle = train_bundle(table)
        save_bundle(bundle, MODEL_BUNDLE_PATH)
    else:
        raise FileNotFoundError(MODEL_BUNDLE_PATH)
    builder = build_current_builder(matches, fifa=fifa)
    return matches, fifa, bundle, builder
