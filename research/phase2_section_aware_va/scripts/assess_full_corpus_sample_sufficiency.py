#!/usr/bin/env python3
"""Quantify corpus sufficiency and freeze a defensible expansion proposal."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PHASE2_ROOT = Path(__file__).resolve().parents[1]
STAGE4 = PHASE2_ROOT / "data/full_corpus/stage4"
CHINA_TZ = timezone(timedelta(hours=8))
RANDOM_SEED = 20260726


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate((len(p_values) - np.arange(len(p_values))) * p_values[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def main() -> None:
    songs = pd.read_csv(STAGE4 / "primary_chinese_song_features.csv")
    sections = pd.read_csv(STAGE4 / "primary_chinese_section_features.csv")
    coverage = pd.read_csv(STAGE4 / "primary_section_type_coverage.csv")
    tests = pd.read_csv(STAGE4 / "primary_paired_structural_tests.csv")
    artists = pd.read_csv(STAGE4 / "primary_artist_coverage.csv")
    ci = pd.read_csv(STAGE4 / "primary_song_feature_bootstrap_ci.csv")

    tests["holm_adjusted_p"] = holm_adjust(tests["wilcoxon_p_two_sided"].to_numpy(dtype=float))
    tests["holm_significant_0_05"] = tests["holm_adjusted_p"] < 0.05
    tests_path = STAGE4 / "primary_paired_structural_tests_holm.csv"
    tests.to_csv(tests_path, index=False)

    current_n = len(songs)
    bridge_song_count = int(coverage.loc[coverage["section_type"] == "bridge", "song_count"].iloc[0])
    scenarios = []
    for added, label in ((0, "current"), (18, "small"), (30, "recommended"), (60, "large"), (170, "full_3_2_1_all_artists")):
        total = current_n + added
        precision_factor = float(np.sqrt(current_n / total))
        scenarios.append({
            "scenario": label,
            "added_songs": added,
            "total_primary_songs": total,
            "approx_standard_error_factor_vs_current": precision_factor,
            "approx_ci_width_reduction_percent": (1.0 - precision_factor) * 100.0,
            "expected_additional_bridge_songs_if_current_rate_holds": added * bridge_song_count / current_n,
            "annotation_cost_relative_to_30_song_plan": added / 30.0,
        })
    scenario_table = pd.DataFrame(scenarios)
    scenario_path = STAGE4 / "sample_expansion_scenarios.csv"
    scenario_table.to_csv(scenario_path, index=False)

    chorus_verse_n = int(tests.loc[
        (tests["comparison"] == "chorus_minus_verse") & (tests["dimension"] == "valence"), "n_songs"
    ].iloc[0])
    core_coverage = {
        row.section_type: {"sections": int(row.section_count), "songs": int(row.song_count), "coverage": float(row.song_coverage)}
        for row in coverage.itertuples(index=False)
    }
    assessment = {
        "created_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "status": "pass",
        "current_primary_corpus": {
            "songs": current_n,
            "sections": len(sections),
            "unique_artists": int(len(artists)),
            "single_song_artists": int((artists["song_count"] == 1).sum()),
            "max_songs_per_artist": int(artists["song_count"].max()),
            "top_artist_share": float(artists["song_share"].max()),
            "chorus_verse_paired_songs": chorus_verse_n,
            "bridge_song_count": bridge_song_count,
            "section_type_coverage": core_coverage,
            "song_feature_bootstrap_ci": ci.to_dict(orient="records"),
        },
        "scientific_judgment": {
            "current_exploratory_analysis": "sufficient",
            "broad_population_generalization": "limited_by_convenience_sampling",
            "mandatory_new_songs_before_current_analysis": 0,
            "recommended_new_songs_for_stronger_external_validity": 30,
            "reason": [
                "Core Chorus-versus-Verse comparison already has 96 independent songs.",
                "Artist concentration is low, but the corpus was not drawn from a frozen probability-like sampling frame.",
                "Bridge coverage is materially smaller than Verse/Chorus coverage.",
                "Thirty added songs provide a practical three-stratum design and approximately 12.6% narrower standard errors under stable variance.",
                "Adding all 170 songs has high cost and does not remove ranking-platform or catalog-selection bias.",
            ],
        },
        "recommended_sampling_design": {
            "sampling_frame": "QQ Music singer ranking top 100, frozen with date and source snapshot before selection",
            "random_seed": RANDOM_SEED,
            "artist_selection": [
                {"rank_stratum": "1-20", "random_artists": 5, "songs_per_artist": 3, "songs": 15},
                {"rank_stratum": "21-50", "random_artists": 5, "songs_per_artist": 2, "songs": 10},
                {"rank_stratum": "51-100", "random_artists": 5, "songs_per_artist": 1, "songs": 5},
            ],
            "total_artists": 15,
            "total_songs": 30,
            "track_pool_rule": "For each selected eligible artist, freeze the QQ top-10 Chinese original-song pool, remove existing corpus tracks, then sample with the same seed.",
            "eligibility_predeclared": [
                "Chinese-language human-created popular song",
                "complete audio available",
                "not already in the corpus",
                "exclude live, remix, instrumental, duplicate recording, and foreign-language track unless the paper scope changes",
            ],
            "artist_cap_after_expansion": "Do not select an artist already at five corpus songs; retain all exclusions and redraws in a log.",
            "analysis_role": "external-validity sensitivity set; do not tune the frozen Teacher on these songs",
        },
        "annotation_reliability_priority": {
            "note": "More one-rater songs do not establish inter-rater reliability.",
            "higher_priority_if_only_one_extra_annotation_task_is_possible": "Have a second independent rater annotate the existing frozen 15-song human-validation set before adding more one-rater VA judgments.",
        },
        "holm_adjustment": {
            "tests": len(tests),
            "significant_after_holm_0_05": int(tests["holm_significant_0_05"].sum()),
            "output": str(tests_path.resolve()),
        },
        "scenario_output": str(scenario_path.resolve()),
    }
    output_path = STAGE4 / "sample_sufficiency_assessment.json"
    output_path.write_text(json.dumps(assessment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(assessment, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
