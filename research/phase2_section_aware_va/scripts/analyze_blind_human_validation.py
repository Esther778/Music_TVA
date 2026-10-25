#!/usr/bin/env python3
"""Evaluate blind section-level human ratings against frozen pseudo-VA outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEATURES = ROOT / (
    "research/phase2_section_aware_va/data/blind_validation/stage5/"
    "blind_sample_15_section_features_with_types.csv"
)
DEFAULT_CANDIDATES = ROOT / (
    "research/phase2_section_aware_va/data/blind_validation/stage5/"
    "blind_sample_15_climax_candidates.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / (
    "research/phase2_section_aware_va/outputs/human_validation/rater_01_evaluation"
)

LEVEL_MAP = {"low": 0, "mid": 1, "high": 2}
CHANGE_MAP = {"down": -1, "stable": 0, "up": 1, "not_applicable": np.nan}


def finite_spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x.loc[mask].nunique() < 2 or y.loc[mask].nunique() < 2:
        return float("nan")
    return float(spearmanr(x.loc[mask], y.loc[mask]).statistic)


def bootstrap_mean_ci(values: list[float], seed: int = 20260722) -> list[float]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if clean.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(10_000, clean.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def safe_wilcoxon(values: list[float]) -> float:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if clean.size == 0 or np.allclose(clean, 0):
        return float("nan")
    return float(wilcoxon(clean, alternative="two-sided", zero_method="wilcox").pvalue)


def audit(human: pd.DataFrame, features: pd.DataFrame) -> dict:
    required = {
        "sample_index", "song_id", "title", "artist", "human_annotation_id",
        "section_order", "start_time_sec", "end_time_sec", "human_valence_level",
        "human_arousal_level", "human_change_from_prev", "is_climax", "rater_id",
    }
    missing = sorted(required - set(human.columns))
    if missing:
        raise ValueError(f"Missing human annotation columns: {missing}")
    if len(human) != 179 or human["human_annotation_id"].nunique() != 179:
        raise ValueError("Expected 179 unique human annotation rows")

    valid = {
        "human_valence_level": set(LEVEL_MAP),
        "human_arousal_level": set(LEVEL_MAP),
        "human_change_from_prev": set(CHANGE_MAP),
        "is_climax": {0, 1},
    }
    for column, allowed in valid.items():
        observed = set(human[column].dropna())
        invalid = observed - allowed
        if invalid:
            raise ValueError(f"Invalid values in {column}: {sorted(invalid)}")
        if human[column].isna().any():
            raise ValueError(f"Incomplete values in {column}")

    first = human["section_order"].eq(1)
    if not human.loc[first, "human_change_from_prev"].eq("not_applicable").all():
        raise ValueError("Every first section must use not_applicable")
    if human.loc[~first, "human_change_from_prev"].eq("not_applicable").any():
        raise ValueError("Only first sections may use not_applicable")

    joined = human.merge(
        features[["human_annotation_id", "song_id", "section_start_sec", "section_end_sec"]],
        on="human_annotation_id",
        how="left",
        suffixes=("", "_feature"),
        validate="one_to_one",
    )
    if joined["song_id_feature"].isna().any():
        raise ValueError("Some human IDs do not join to frozen section features")
    if not joined["song_id"].eq(joined["song_id_feature"]).all():
        raise ValueError("Song identity mismatch after ID join")
    start_error = float((joined["start_time_sec"] - joined["section_start_sec"]).abs().max())
    end_error = float((joined["end_time_sec"] - joined["section_end_sec"]).abs().max())
    if max(start_error, end_error) > 1e-9:
        raise ValueError("Human annotation boundaries differ from frozen boundaries")

    return {
        "rows": int(len(human)),
        "unique_ids": int(human["human_annotation_id"].nunique()),
        "songs": int(human["song_id"].nunique()),
        "rater_ids": sorted(human["rater_id"].unique().tolist()),
        "max_start_boundary_error_sec": start_error,
        "max_end_boundary_error_sec": end_error,
        "status": "pass",
    }


def evaluate(human: pd.DataFrame, features: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    merged = human.merge(features, on=[
        "sample_index", "song_id", "title", "artist", "human_annotation_id", "section_order"
    ], validate="one_to_one")
    merged["human_valence_ordinal"] = merged["human_valence_level"].map(LEVEL_MAP)
    merged["human_arousal_ordinal"] = merged["human_arousal_level"].map(LEVEL_MAP)
    merged["human_change_ordinal"] = merged["human_change_from_prev"].map(CHANGE_MAP)

    per_song: list[dict] = []
    chorus_verse_rows: list[dict] = []
    for song_id, song in merged.groupby("song_id", sort=False):
        song = song.sort_values("section_order").copy()
        song["model_arousal_delta"] = song["mean_arousal"].diff()
        non_first = song["human_change_ordinal"].notna()
        non_stable = non_first & song["human_change_ordinal"].ne(0)
        sign_matches = int((
            np.sign(song.loc[non_stable, "model_arousal_delta"])
            == song.loc[non_stable, "human_change_ordinal"]
        ).sum())

        human_peaks = song.loc[song["is_climax"].eq(1), "section_order"].astype(int).tolist()
        candidate = candidates.loc[candidates["song_id"].eq(song_id)].iloc[0]
        model_top = int(candidate["climax_section_order_by_mean_arousal"])

        per_song.append({
            "sample_index": int(song["sample_index"].iloc[0]),
            "song_id": song_id,
            "title": song["title"].iloc[0],
            "sections": int(len(song)),
            "valence_spearman": finite_spearman(song["human_valence_ordinal"], song["mean_valence"]),
            "arousal_spearman": finite_spearman(song["human_arousal_ordinal"], song["mean_arousal"]),
            "change_spearman_vs_arousal_delta": finite_spearman(
                song.loc[non_first, "human_change_ordinal"],
                song.loc[non_first, "model_arousal_delta"],
            ),
            "nonstable_change_sign_matches": sign_matches,
            "nonstable_change_count": int(non_stable.sum()),
            "human_peak_count": int(len(human_peaks)),
            "human_peak_sections": "|".join(map(str, human_peaks)),
            "model_top_arousal_section": model_top,
            "model_top_arousal_type": candidate["climax_candidate_section_type"],
            "model_top_exact_human_peak": int(model_top in human_peaks),
            "model_top_adjacent_human_peak": int(any(abs(model_top - peak) <= 1 for peak in human_peaks)),
            "human_song_mean_valence": float(song["human_valence_ordinal"].mean()),
            "human_song_mean_arousal": float(song["human_arousal_ordinal"].mean()),
            "model_song_mean_valence": float(song["mean_valence"].mean()),
            "model_song_mean_arousal": float(song["mean_arousal"].mean()),
        })

        chorus = song[song["section_type"].eq("chorus")]
        verse = song[song["section_type"].eq("verse")]
        if not chorus.empty and not verse.empty:
            chorus_verse_rows.append({
                "song_id": song_id,
                "human_valence_lift": float(
                    chorus["human_valence_ordinal"].mean() - verse["human_valence_ordinal"].mean()
                ),
                "human_arousal_lift": float(
                    chorus["human_arousal_ordinal"].mean() - verse["human_arousal_ordinal"].mean()
                ),
                "model_valence_lift": float(chorus["mean_valence"].mean() - verse["mean_valence"].mean()),
                "model_arousal_lift": float(chorus["mean_arousal"].mean() - verse["mean_arousal"].mean()),
            })

    per_song_df = pd.DataFrame(per_song).sort_values("sample_index")
    cv = pd.DataFrame(chorus_verse_rows)

    for human_col, model_col, suffix in [
        ("human_valence_ordinal", "mean_valence", "valence"),
        ("human_arousal_ordinal", "mean_arousal", "arousal"),
    ]:
        merged[f"human_{suffix}_within_song"] = merged[human_col] - merged.groupby("song_id")[human_col].transform("mean")
        merged[f"model_{suffix}_within_song"] = merged[model_col] - merged.groupby("song_id")[model_col].transform("mean")

    valence_rhos = per_song_df["valence_spearman"].dropna().tolist()
    arousal_rhos = per_song_df["arousal_spearman"].dropna().tolist()
    change_rhos = per_song_df["change_spearman_vs_arousal_delta"].dropna().tolist()

    non_first = merged["human_change_ordinal"].notna()
    merged = merged.sort_values(["sample_index", "section_order"])
    merged["model_arousal_delta"] = merged.groupby("song_id")["mean_arousal"].diff()

    type_counts = (
        merged.loc[merged["is_climax"].eq(1), "section_type"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    human_v_lift = cv["human_valence_lift"].tolist()
    human_a_lift = cv["human_arousal_lift"].tolist()
    summary = {
        "interpretation": "exploratory_one_rater_ordinal_validation",
        "counts": {
            "songs": int(merged["song_id"].nunique()),
            "sections": int(len(merged)),
            "human_peaks": int(merged["is_climax"].sum()),
            "human_peak_section_types": {key: int(value) for key, value in type_counts.items()},
            "songs_with_chorus_and_verse": int(len(cv)),
        },
        "within_song_macro": {
            "valence_spearman_mean": float(np.nanmean(valence_rhos)),
            "valence_spearman_median": float(np.nanmedian(valence_rhos)),
            "valence_spearman_bootstrap_95_ci": bootstrap_mean_ci(valence_rhos),
            "arousal_spearman_mean": float(np.nanmean(arousal_rhos)),
            "arousal_spearman_median": float(np.nanmedian(arousal_rhos)),
            "arousal_spearman_bootstrap_95_ci": bootstrap_mean_ci(arousal_rhos),
            "change_spearman_mean": float(np.nanmean(change_rhos)),
            "change_spearman_bootstrap_95_ci": bootstrap_mean_ci(change_rhos),
        },
        "section_pooled": {
            "valence_spearman": finite_spearman(merged["human_valence_ordinal"], merged["mean_valence"]),
            "arousal_spearman": finite_spearman(merged["human_arousal_ordinal"], merged["mean_arousal"]),
            "within_song_centered_valence_spearman": finite_spearman(
                merged["human_valence_within_song"], merged["model_valence_within_song"]
            ),
            "within_song_centered_arousal_spearman": finite_spearman(
                merged["human_arousal_within_song"], merged["model_arousal_within_song"]
            ),
            "change_spearman_vs_arousal_delta": finite_spearman(
                merged.loc[non_first, "human_change_ordinal"],
                merged.loc[non_first, "model_arousal_delta"],
            ),
        },
        "between_song": {
            "valence_mean_spearman": finite_spearman(
                per_song_df["human_song_mean_valence"], per_song_df["model_song_mean_valence"]
            ),
            "arousal_mean_spearman": finite_spearman(
                per_song_df["human_song_mean_arousal"], per_song_df["model_song_mean_arousal"]
            ),
        },
        "change_direction": {
            "nonstable_sign_matches": int(per_song_df["nonstable_change_sign_matches"].sum()),
            "nonstable_count": int(per_song_df["nonstable_change_count"].sum()),
            "nonstable_sign_accuracy": float(
                per_song_df["nonstable_change_sign_matches"].sum()
                / per_song_df["nonstable_change_count"].sum()
            ),
        },
        "frozen_top_arousal_candidate": {
            "exact_human_peak_hits": int(per_song_df["model_top_exact_human_peak"].sum()),
            "adjacent_human_peak_hits": int(per_song_df["model_top_adjacent_human_peak"].sum()),
            "songs": int(len(per_song_df)),
        },
        "chorus_minus_verse": {
            "songs": int(len(cv)),
            "human_valence_mean_lift_ordinal": float(np.mean(human_v_lift)),
            "human_valence_positive_songs": int((cv["human_valence_lift"] > 0).sum()),
            "human_valence_wilcoxon_p": safe_wilcoxon(human_v_lift),
            "human_arousal_mean_lift_ordinal": float(np.mean(human_a_lift)),
            "human_arousal_positive_songs": int((cv["human_arousal_lift"] > 0).sum()),
            "human_arousal_wilcoxon_p": safe_wilcoxon(human_a_lift),
            "human_model_valence_lift_spearman": finite_spearman(
                cv["human_valence_lift"], cv["model_valence_lift"]
            ),
            "human_model_arousal_lift_spearman": finite_spearman(
                cv["human_arousal_lift"], cv["model_arousal_lift"]
            ),
        },
        "human_level_counts": {
            "valence": {key: int(value) for key, value in merged["human_valence_level"].value_counts().to_dict().items()},
            "arousal": {key: int(value) for key, value in merged["human_arousal_level"].value_counts().to_dict().items()},
            "change": {key: int(value) for key, value in merged["human_change_from_prev"].value_counts().to_dict().items()},
        },
    }
    return per_song_df, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    human = pd.read_csv(args.human_csv)
    features = pd.read_csv(args.features_csv)
    candidates = pd.read_csv(args.candidates_csv)
    audit_result = audit(human, features)
    per_song, summary = evaluate(human, features, candidates)
    payload = {"audit": audit_result, "metrics": summary}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_song.to_csv(args.output_dir / "per_song_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
