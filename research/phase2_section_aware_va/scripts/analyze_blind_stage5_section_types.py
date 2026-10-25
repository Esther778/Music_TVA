#!/usr/bin/env python3
"""Analyze frozen pseudo-VA features against held-out structural section types."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


PHASE2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE4 = PHASE2_ROOT / "data/blind_validation/stage4"
DEFAULT_STAGE5 = PHASE2_ROOT / "data/blind_validation/stage5"
CHINA_TZ = timezone(timedelta(hours=8))
BOOTSTRAP_SEED = 20260721
BOOTSTRAP_SAMPLES = 20000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def paired_stats(comparison: str, dimension: str, values: pd.Series) -> dict[str, object]:
    differences = values.dropna().to_numpy(dtype=float)
    nonzero = differences[~np.isclose(differences, 0.0)]
    if len(nonzero):
        statistic, p_value = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        ranks = rankdata(np.abs(nonzero))
        positive = float(ranks[nonzero > 0].sum())
        negative = float(ranks[nonzero < 0].sum())
        rank_biserial = (positive - negative) / (positive + negative)
    else:
        statistic, p_value, rank_biserial = 0.0, 1.0, 0.0
    ci_low, ci_high = bootstrap_mean_ci(differences)
    return {
        "comparison": comparison,
        "dimension": dimension,
        "n_songs": len(differences),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "bootstrap_mean_ci95_low": ci_low,
        "bootstrap_mean_ci95_high": ci_high,
        "songs_positive": int((differences > 0).sum()),
        "songs_negative": int((differences < 0).sum()),
        "songs_zero": int(np.isclose(differences, 0.0).sum()),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p_two_sided": float(p_value),
        "rank_biserial_effect": float(rank_biserial),
        "exploratory": True,
    }


def add_heldout_types(sections: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_columns = [
        "human_annotation_id",
        "section_type",
        "annotation_kind",
        "is_approved",
        "confidence",
        "updated_at",
    ]
    enriched = sections.drop(columns=["section_type"]).merge(
        labels[label_columns], on="human_annotation_id", how="left", validate="one_to_one"
    )
    if enriched["section_type"].isna().any():
        raise RuntimeError("Held-out section type join is incomplete")
    return enriched.sort_values(["sample_index", "section_order"]).reset_index(drop=True)


def type_summary(sections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for section_type, group in sections.groupby("section_type", sort=True):
        rows.append(
            {
                "section_type": section_type,
                "section_count": len(group),
                "song_count": group["song_id"].nunique(),
                "mean_valence": float(group["mean_valence"].mean()),
                "median_valence": float(group["mean_valence"].median()),
                "std_valence_across_sections": float(group["mean_valence"].std(ddof=0)),
                "mean_arousal": float(group["mean_arousal"].mean()),
                "median_arousal": float(group["mean_arousal"].median()),
                "std_arousal_across_sections": float(group["mean_arousal"].std(ddof=0)),
                "mean_normalized_midpoint": float(group["normalized_midpoint"].mean()),
                "median_normalized_midpoint": float(group["normalized_midpoint"].median()),
                "mean_within_section_valence_std": float(group["std_valence"].mean()),
                "mean_within_section_arousal_std": float(group["std_arousal"].mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_type_comparison(sections: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    song_type = sections.groupby(["song_id", "title", "section_type"])[["mean_valence", "mean_arousal"]].mean()
    song_type = song_type.reset_index()
    pivot = song_type.pivot(index=["song_id", "title"], columns="section_type", values=["mean_valence", "mean_arousal"])
    paired = pd.DataFrame(index=pivot.index).reset_index()
    rows: list[dict[str, object]] = []
    for dimension in ("valence", "arousal"):
        difference = pivot[(f"mean_{dimension}", "chorus")] - pivot[(f"mean_{dimension}", "verse")]
        paired[f"chorus_minus_verse_{dimension}"] = difference.to_numpy()
        rows.append(paired_stats("chorus_minus_verse", dimension, difference))
    return paired, rows


def adjacent_transition_comparisons(sections: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    ordered = sections.sort_values(["sample_index", "section_order"]).copy()
    ordered["prev_section_type"] = ordered.groupby("song_id")["section_type"].shift(1)
    ordered["transition"] = ordered["prev_section_type"].fillna("START") + "_to_" + ordered["section_type"]
    ordered["transition_valence_change"] = ordered["change_from_prev_valence"]
    ordered["transition_arousal_change"] = ordered["change_from_prev_arousal"]
    rows: list[dict[str, object]] = []
    for transition in ("verse_to_chorus", "pre_chorus_to_chorus"):
        subset = ordered[ordered["transition"] == transition]
        per_song = subset.groupby(["song_id", "title"])[
            ["transition_valence_change", "transition_arousal_change"]
        ].mean()
        for dimension in ("valence", "arousal"):
            rows.append(paired_stats(transition, dimension, per_song[f"transition_{dimension}_change"]))
    keep = ordered[
        ordered["transition"].isin(["verse_to_chorus", "pre_chorus_to_chorus"])
    ][
        [
            "song_id",
            "title",
            "section_order",
            "prev_section_type",
            "section_type",
            "transition",
            "transition_valence_change",
            "transition_arousal_change",
        ]
    ]
    return keep, rows


def repeated_consistency(sections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (song_id, title, section_type), group in sections.groupby(["song_id", "title", "section_type"]):
        if len(group) < 2:
            continue
        points = group[["mean_valence", "mean_arousal"]].to_numpy(dtype=float)
        pairs = list(combinations(range(len(points)), 2))
        valence_distances = np.asarray([abs(points[i, 0] - points[j, 0]) for i, j in pairs])
        arousal_distances = np.asarray([abs(points[i, 1] - points[j, 1]) for i, j in pairs])
        va_distances = np.asarray([np.linalg.norm(points[i] - points[j]) for i, j in pairs])
        rows.append(
            {
                "song_id": song_id,
                "title": title,
                "section_type": section_type,
                "occurrences": len(group),
                "pairs": len(pairs),
                "mean_pairwise_valence_distance": float(valence_distances.mean()),
                "mean_pairwise_arousal_distance": float(arousal_distances.mean()),
                "mean_pairwise_va_distance": float(va_distances.mean()),
                "max_pairwise_va_distance": float(va_distances.max()),
            }
        )
    detail = pd.DataFrame(rows)
    summary_rows = []
    for section_type, group in detail.groupby("section_type"):
        summary_rows.append(
            {
                "section_type": section_type,
                "song_count": len(group),
                "mean_song_pairwise_valence_distance": float(group["mean_pairwise_valence_distance"].mean()),
                "mean_song_pairwise_arousal_distance": float(group["mean_pairwise_arousal_distance"].mean()),
                "mean_song_pairwise_va_distance": float(group["mean_pairwise_va_distance"].mean()),
                "median_song_pairwise_va_distance": float(group["mean_pairwise_va_distance"].median()),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def climax_candidates(songs: pd.DataFrame, sections: pd.DataFrame) -> pd.DataFrame:
    candidates = songs.merge(
        sections[["song_id", "section_order", "section_type"]],
        left_on=["song_id", "climax_section_order_by_mean_arousal"],
        right_on=["song_id", "section_order"],
        how="left",
        validate="one_to_one",
    )
    return candidates[
        [
            "sample_index",
            "song_id",
            "title",
            "artist",
            "climax_section_order_by_mean_arousal",
            "section_type",
            "climax_section_mean_arousal",
            "climax_section_time_ratio",
            "frame_peak_arousal",
            "frame_peak_arousal_time_ratio",
        ]
    ].rename(columns={"section_type": "climax_candidate_section_type"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-dir", type=Path, default=DEFAULT_STAGE4)
    parser.add_argument("--stage5-dir", type=Path, default=DEFAULT_STAGE5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    section_path = args.stage4_dir / "blind_sample_15_section_features.csv"
    song_path = args.stage4_dir / "blind_sample_15_song_structure_features.csv"
    stage4_manifest_path = args.stage4_dir / "stage4_manifest.json"
    labels_path = args.stage5_dir / "blind_sample_15_heldout_section_types.json"
    reveal_manifest_path = args.stage5_dir / "stage5_label_reveal_manifest.json"

    stage4 = json.loads(stage4_manifest_path.read_text(encoding="utf-8"))
    reveal = json.loads(reveal_manifest_path.read_text(encoding="utf-8"))
    label_records = json.loads(labels_path.read_text(encoding="utf-8"))
    if stage4.get("status") != "pass" or reveal.get("status") != "pass":
        raise RuntimeError("Stage 4 and held-out label reveal must pass before Stage 5 analysis")
    if canonical_sha256(label_records) != reveal["heldout_labels_canonical_sha256"]:
        raise RuntimeError("Held-out structural-label hash mismatch")

    sections = pd.read_csv(section_path)
    songs = pd.read_csv(song_path)
    labels = pd.DataFrame(label_records)
    enriched = add_heldout_types(sections, labels)
    summary = type_summary(enriched)
    chorus_verse, test_rows = paired_type_comparison(enriched)
    transitions, transition_tests = adjacent_transition_comparisons(enriched)
    test_rows.extend(transition_tests)
    repeat_detail, repeat_summary = repeated_consistency(enriched)
    repeat_pivot = repeat_detail.pivot(
        index=["song_id", "title"],
        columns="section_type",
        values=[
            "mean_pairwise_valence_distance",
            "mean_pairwise_arousal_distance",
            "mean_pairwise_va_distance",
        ],
    )
    for metric, dimension in (
        ("mean_pairwise_valence_distance", "valence_distance"),
        ("mean_pairwise_arousal_distance", "arousal_distance"),
        ("mean_pairwise_va_distance", "va_distance"),
    ):
        difference = repeat_pivot[(metric, "chorus")] - repeat_pivot[(metric, "verse")]
        test_rows.append(paired_stats("repeated_chorus_minus_repeated_verse", dimension, difference))
    tests = pd.DataFrame(test_rows)
    climaxes = climax_candidates(songs, enriched)

    args.stage5_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "section_features_with_types": (args.stage5_dir / "blind_sample_15_section_features_with_types.csv", enriched),
        "section_type_summary": (args.stage5_dir / "blind_sample_15_section_type_summary.csv", summary),
        "chorus_verse_song_pairs": (args.stage5_dir / "blind_sample_15_chorus_verse_song_pairs.csv", chorus_verse),
        "adjacent_transitions": (args.stage5_dir / "blind_sample_15_selected_adjacent_transitions.csv", transitions),
        "paired_structural_tests": (args.stage5_dir / "blind_sample_15_paired_structural_tests.csv", tests),
        "repeated_section_consistency": (args.stage5_dir / "blind_sample_15_repeated_section_consistency.csv", repeat_detail),
        "repeated_section_summary": (args.stage5_dir / "blind_sample_15_repeated_section_summary.csv", repeat_summary),
        "climax_candidates": (args.stage5_dir / "blind_sample_15_climax_candidates.csv", climaxes),
    }
    output_manifest = {}
    for name, (path, frame) in outputs.items():
        frame.to_csv(path, index=False)
        output_manifest[name] = {"path": str(path.resolve()), "rows": len(frame), "sha256": sha256_file(path)}

    errors = []
    if len(enriched) != 179 or enriched["human_annotation_id"].nunique() != 179:
        errors.append("enriched_section_count")
    if enriched["section_type"].isna().any():
        errors.append("missing_section_types")
    if len(climaxes) != 15 or climaxes["climax_candidate_section_type"].isna().any():
        errors.append("climax_candidate_join")
    if chorus_verse[["chorus_minus_verse_valence", "chorus_minus_verse_arousal"]].isna().any().any():
        errors.append("chorus_verse_coverage")

    manifest = {
        "created_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "protocol": "Stage 5 exploratory structural-label analysis; song is the independent unit",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "songs": int(enriched["song_id"].nunique()),
        "sections": len(enriched),
        "section_type_counts": {key: int(value) for key, value in enriched["section_type"].value_counts().items()},
        "approved_sections": int(enriched["is_approved"].fillna(False).sum()),
        "human_va_ground_truth_available": False,
        "human_change_ground_truth_available": False,
        "human_climax_ground_truth_available": False,
        "statistical_unit": "song",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "multiple_comparison_note": "Exploratory tests are reported unadjusted and must not be presented as confirmatory.",
        "stage4_section_features_sha256": sha256_file(section_path),
        "heldout_labels_canonical_sha256": canonical_sha256(label_records),
        "outputs": output_manifest,
    }
    manifest_path = args.stage5_dir / "stage5_analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nPaired structural tests:\n", tests.to_string(index=False))
    print("\nClimax candidate types:\n", climaxes["climax_candidate_section_type"].value_counts().to_string())
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
