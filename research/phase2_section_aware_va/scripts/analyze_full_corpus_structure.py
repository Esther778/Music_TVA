#!/usr/bin/env python3
"""Analyze section-aware VA structure and audit corpus coverage."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_blind_stage5_section_types import (
    CHINA_TZ,
    add_heldout_types,
    adjacent_transition_comparisons,
    canonical_sha256,
    climax_candidates,
    paired_type_comparison,
    repeated_consistency,
    sha256_file,
    type_summary,
)


PHASE2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1 = PHASE2_ROOT / "data/full_corpus/stage1"
DEFAULT_STAGE3 = PHASE2_ROOT / "data/full_corpus/stage3"
DEFAULT_STAGE4 = PHASE2_ROOT / "data/full_corpus/stage4"
# Explicit study-scope exclusions. Outputs are retained but excluded from the
# Chinese-pop primary analysis requested by the researcher.
LANGUAGE_EXCLUSIONS = {
    "ANGEL": "non-Chinese-language song",
    "Easy On Me": "non-Chinese-language song",
}


def bootstrap_ci(values: np.ndarray, statistic: str, seed: int = 20260726, samples: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    stats = np.mean(draws, axis=1) if statistic == "mean" else np.median(draws, axis=1)
    return tuple(float(value) for value in np.quantile(stats, [0.025, 0.975]))


def artist_coverage(songs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    counts = songs.groupby("artist").size().sort_values(ascending=False)
    total = len(songs)
    shares = counts.to_numpy(dtype=float) / total
    effective_artists = float(1.0 / np.square(shares).sum())
    table = counts.rename("song_count").reset_index()
    table["song_share"] = table["song_count"] / total
    return table, {
        "songs": total, "unique_artists": int(len(counts)),
        "max_songs_per_artist": int(counts.max()),
        "top_artist_share": float(counts.iloc[0] / total),
        "top5_artist_share": float(counts.iloc[:5].sum() / total),
        "effective_artist_count_simpson": effective_artists,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--stage4-dir", type=Path, default=DEFAULT_STAGE4)
    args = parser.parse_args()

    section_path = args.stage3_dir / "full_corpus_99_section_features.csv"
    song_path = args.stage3_dir / "full_corpus_99_song_structure_features.csv"
    alignment_manifest_path = args.stage3_dir / "full_corpus_alignment_manifest.json"
    labels_path = args.stage4_dir / "full_corpus_99_frozen_section_types.json"
    reveal_manifest_path = args.stage4_dir / "full_corpus_label_reveal_manifest.json"
    alignment = json.loads(alignment_manifest_path.read_text(encoding="utf-8"))
    reveal = json.loads(reveal_manifest_path.read_text(encoding="utf-8"))
    label_records = json.loads(labels_path.read_text(encoding="utf-8"))
    if alignment.get("status") != "pass" or reveal.get("status") != "pass":
        raise RuntimeError("Alignment and section-type reveal must pass before analysis")
    if canonical_sha256(label_records) != reveal["labels_canonical_sha256"]:
        raise RuntimeError("Frozen section-type hash mismatch")

    sections_all = pd.read_csv(section_path, dtype={"song_id": str})
    songs_all = pd.read_csv(song_path, dtype={"song_id": str})
    labels = pd.DataFrame(label_records)
    labels["human_annotation_id"] = labels["human_annotation_id"].astype(str)
    enriched_all = add_heldout_types(sections_all, labels)
    exclusions = songs_all[songs_all["title"].isin(LANGUAGE_EXCLUSIONS)].copy()
    exclusions["exclusion_reason"] = exclusions["title"].map(LANGUAGE_EXCLUSIONS)
    in_scope_ids = set(songs_all.loc[~songs_all["title"].isin(LANGUAGE_EXCLUSIONS), "song_id"])
    sections = enriched_all[enriched_all["song_id"].isin(in_scope_ids)].copy()
    songs = songs_all[songs_all["song_id"].isin(in_scope_ids)].copy()

    type_table = type_summary(sections)
    chorus_verse, tests = paired_type_comparison(sections)
    transitions, transition_tests = adjacent_transition_comparisons(sections)
    tests.extend(transition_tests)
    repeat_detail, repeat_summary = repeated_consistency(sections)
    climaxes = climax_candidates(songs, sections)
    artist_table, artist_summary = artist_coverage(songs)

    coverage_rows = []
    for section_type, group in sections.groupby("section_type"):
        coverage_rows.append({
            "section_type": section_type, "section_count": len(group),
            "song_count": group["song_id"].nunique(),
            "song_coverage": group["song_id"].nunique() / len(songs),
        })
    coverage = pd.DataFrame(coverage_rows).sort_values("section_count", ascending=False)
    ci_rows = []
    for column in ("global_valence_range", "global_arousal_range", "climax_section_time_ratio", "max_section_contrast"):
        values = songs[column].dropna().to_numpy(dtype=float)
        low, high = bootstrap_ci(values, "mean")
        ci_rows.append({"feature": column, "n_songs": len(values), "mean": float(values.mean()), "ci95_low": low, "ci95_high": high, "ci95_width": high - low})
    song_feature_ci = pd.DataFrame(ci_rows)

    args.stage4_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "all_sections_with_types": (args.stage4_dir / "full_corpus_99_section_features_with_types.csv", enriched_all),
        "scope_exclusions": (args.stage4_dir / "full_corpus_scope_exclusions.csv", exclusions),
        "primary_sections": (args.stage4_dir / "primary_chinese_section_features.csv", sections),
        "primary_songs": (args.stage4_dir / "primary_chinese_song_features.csv", songs),
        "section_type_summary": (args.stage4_dir / "primary_section_type_summary.csv", type_table),
        "section_type_coverage": (args.stage4_dir / "primary_section_type_coverage.csv", coverage),
        "chorus_verse_pairs": (args.stage4_dir / "primary_chorus_verse_song_pairs.csv", chorus_verse),
        "adjacent_transitions": (args.stage4_dir / "primary_adjacent_transitions.csv", transitions),
        "paired_tests": (args.stage4_dir / "primary_paired_structural_tests.csv", pd.DataFrame(tests)),
        "repeated_detail": (args.stage4_dir / "primary_repeated_section_consistency.csv", repeat_detail),
        "repeated_summary": (args.stage4_dir / "primary_repeated_section_summary.csv", repeat_summary),
        "climax_candidates": (args.stage4_dir / "primary_climax_candidates.csv", climaxes),
        "artist_coverage": (args.stage4_dir / "primary_artist_coverage.csv", artist_table),
        "song_feature_ci": (args.stage4_dir / "primary_song_feature_bootstrap_ci.csv", song_feature_ci),
    }
    output_manifest = {}
    for name, (path, frame) in outputs.items():
        frame.to_csv(path, index=False)
        output_manifest[name] = {"path": str(path.resolve()), "rows": len(frame), "sha256": sha256_file(path)}

    errors = []
    if len(enriched_all) != 1097 or enriched_all["human_annotation_id"].nunique() != 1097: errors.append("all_section_count")
    if len(songs_all) != 99 or len(songs) != 97 or len(exclusions) != 2: errors.append("study_scope_count")
    if sections["section_type"].isna().any(): errors.append("missing_section_types")
    if len(climaxes) != len(songs) or climaxes["climax_candidate_section_type"].isna().any(): errors.append("climax_join")
    manifest = {
        "created_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "protocol": "Primary Chinese-pop section-aware exploratory analysis; song is the independent unit",
        "status": "pass" if not errors else "fail", "errors": errors,
        "all_predicted_songs": len(songs_all), "primary_in_scope_songs": len(songs),
        "excluded_songs": len(exclusions), "primary_sections": len(sections),
        "section_type_counts": {key: int(value) for key, value in sections["section_type"].value_counts().items()},
        "artist_coverage": artist_summary,
        "human_va_ground_truth_available": False,
        "statistical_unit": "song",
        "analysis_status": "exploratory",
        "multiple_comparison_note": "Unadjusted exploratory tests; confirmatory claims require preregistered hypotheses or correction.",
        "outputs": output_manifest,
    }
    manifest_path = args.stage4_dir / "full_corpus_structure_analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
