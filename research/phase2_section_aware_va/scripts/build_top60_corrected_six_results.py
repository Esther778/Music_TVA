#!/usr/bin/env python3
"""Build the six corrected Top60/100-song result tables used by the paper."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "exports/gpt_top60_quota_100_section_va_data_package_20260820"
OUTPUT = ROOT / "exports/top60_quota_100_corrected_six_results_20260820"
RUN_ID = "top60_quota_100_corrected_20260820"
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260820


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm_adjust(values: list[float]) -> list[float]:
    values_array = np.asarray(values, dtype=float)
    order = np.argsort(values_array)
    adjusted = np.empty_like(values_array)
    running = 0.0
    total = len(values_array)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values_array[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def cluster_bootstrap_ci(
    data: pd.DataFrame,
    value_column: str,
    all_clusters: list[str],
    seed: int,
) -> tuple[float, float]:
    grouped = {
        cluster: group[value_column].dropna().to_numpy(dtype=float)
        for cluster, group in data.groupby("quota_artist", sort=False)
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    cluster_count = len(all_clusters)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        arrays = [grouped.get(all_clusters[index], np.empty(0)) for index in sampled]
        arrays = [array for array in arrays if array.size]
        estimates[draw] = np.concatenate(arrays).mean() if arrays else np.nan
    estimates = estimates[np.isfinite(estimates)]
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def direction_consistent(song_low: float, song_high: float, cluster_low: float, cluster_high: float) -> bool:
    positive = song_low > 0 and cluster_low > 0
    negative = song_high < 0 and cluster_high < 0
    return bool(positive or negative)


def json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for raw_record in frame.to_dict(orient="records"):
        record: dict[str, object] = {}
        for key, value in raw_record.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, np.generic):
                record[key] = value.item()
            else:
                record[key] = value
        records.append(record)
    return records


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    section_features_path = PACKAGE / "05_section_level/top60_quota_100_section_features.csv"
    section_summary_path = PACKAGE / "09_derived_tables/va_section_type_summary.csv"
    pair_path = PACKAGE / "09_derived_tables/va_paired_section_tests.csv"
    transition_summary_path = PACKAGE / "09_derived_tables/va_transition_summary.csv"
    adjacent_path = PACKAGE / "09_derived_tables/adjacent_section_transitions_gpt.csv"
    song_type_path = PACKAGE / "09_derived_tables/song_by_section_type_means_with_rank.csv"
    peak_path = PACKAGE / "09_derived_tables/va_arousal_peak_sections.csv"
    peak_summary_path = PACKAGE / "09_derived_tables/va_arousal_peak_summary.csv"
    rank_test_path = PACKAGE / "07_paper_statistics/paper_rank_band_inferential_tests.csv"
    rank_pair_path = PACKAGE / "07_paper_statistics/paper_rank_band_pairwise_tests.csv"
    rank_assoc_path = PACKAGE / "07_paper_statistics/paper_rank_section_type_association.json"

    sections = pd.read_csv(section_features_path)
    section_summary = pd.read_csv(section_summary_path)
    song_types = pd.read_csv(song_type_path)
    adjacent = pd.read_csv(adjacent_path)
    pair_source = pd.read_csv(pair_path)
    transition_source = pd.read_csv(transition_summary_path)
    peaks = pd.read_csv(peak_path)
    peak_summary = pd.read_csv(peak_summary_path)
    rank_tests = pd.read_csv(rank_test_path)
    rank_pairs = pd.read_csv(rank_pair_path)
    rank_assoc = json.loads(rank_assoc_path.read_text(encoding="utf-8"))

    all_clusters = sorted(song_types["quota_artist"].dropna().unique().tolist())
    if len(all_clusters) != 60:
        raise ValueError(f"Expected 60 quota artists, found {len(all_clusters)}")
    if sections["song_id"].nunique() != 100 or len(sections) != 1046:
        raise ValueError("Corrected section snapshot must contain 100 songs and 1046 sections")

    # 1. Seven section types.
    type_order = ["intro", "verse", "pre_chorus", "chorus", "bridge", "interlude", "outro"]
    table_1 = section_summary.copy()
    table_1["section_type"] = pd.Categorical(table_1["section_type"], type_order, ordered=True)
    table_1 = table_1.sort_values("section_type").reset_index(drop=True)
    table_1["section_type"] = table_1["section_type"].astype(str)
    table_1 = table_1[
        [
            "section_type",
            "section_count",
            "song_count",
            "song_coverage",
            "mean_valence",
            "mean_arousal",
            "mean_position",
        ]
    ]
    table_1.to_csv(OUTPUT / "01_section_type_statistics_corrected.csv", index=False)

    # 2. Exact/collapsed sequences and all adjacent transition frequencies.
    raw_counter: Counter[str] = Counter()
    collapsed_counter: Counter[str] = Counter()
    example_by_raw: dict[str, tuple[str, str]] = {}
    example_by_collapsed: dict[str, tuple[str, str]] = {}
    for _, group in sections.sort_values(["song_id", "section_order"]).groupby("song_id", sort=False):
        types = group["section_type"].astype(str).tolist()
        raw_sequence = " > ".join(types)
        collapsed_types = [types[0]]
        for section_type in types[1:]:
            if section_type != collapsed_types[-1]:
                collapsed_types.append(section_type)
        collapsed_sequence = " > ".join(collapsed_types)
        raw_counter[raw_sequence] += 1
        collapsed_counter[collapsed_sequence] += 1
        example = (str(group.iloc[0]["title"]), str(group.iloc[0]["artist"]))
        example_by_raw.setdefault(raw_sequence, example)
        example_by_collapsed.setdefault(collapsed_sequence, example)

    transition_counts = adjacent["transition"].value_counts()
    sequence_records: list[dict[str, object]] = [
        {"record_type": "summary", "metric": "song_count", "value": 100},
        {"record_type": "summary", "metric": "unique_raw_sequence_count", "value": len(raw_counter)},
        {"record_type": "summary", "metric": "unique_collapsed_sequence_count", "value": len(collapsed_counter)},
        {"record_type": "summary", "metric": "adjacent_transition_total", "value": int(transition_counts.sum())},
    ]
    for sequence, count in sorted(raw_counter.items(), key=lambda item: (-item[1], item[0])):
        title, artist = example_by_raw[sequence]
        sequence_records.append(
            {
                "record_type": "raw_sequence",
                "sequence": sequence,
                "count": count,
                "example_title": title,
                "example_artist": artist,
            }
        )
    for sequence, count in sorted(collapsed_counter.items(), key=lambda item: (-item[1], item[0])):
        title, artist = example_by_collapsed[sequence]
        sequence_records.append(
            {
                "record_type": "collapsed_sequence",
                "sequence": sequence,
                "count": count,
                "example_title": title,
                "example_artist": artist,
            }
        )
    for transition, count in transition_counts.items():
        sequence_records.append(
            {"record_type": "transition_frequency", "transition": transition, "count": int(count)}
        )
    table_2 = pd.DataFrame(sequence_records)
    table_2.to_csv(OUTPUT / "02_sequence_and_transition_frequencies_corrected.csv", index=False)

    # 3. Song-paired section-type comparisons, including dz and Holm adjustment.
    pivot_v = song_types.pivot(index="song_id", columns="section_type", values="mean_valence")
    pivot_a = song_types.pivot(index="song_id", columns="section_type", values="mean_arousal")
    song_meta = song_types[["song_id", "quota_artist"]].drop_duplicates("song_id").set_index("song_id")
    comparison_types = {
        "chorus_minus_verse": ("verse", "chorus"),
        "pre_chorus_minus_verse": ("verse", "pre_chorus"),
        "chorus_minus_pre_chorus": ("pre_chorus", "chorus"),
        "outro_minus_intro": ("intro", "outro"),
        "bridge_minus_chorus": ("chorus", "bridge"),
    }
    pair_rows: list[dict[str, object]] = []
    pair_source = pair_source.copy()
    for dimension, pivot in [("valence", pivot_v), ("arousal", pivot_a)]:
        dimension_source = pair_source[pair_source["dimension"] == dimension].copy()
        p_holm = holm_adjust(dimension_source["p_value"].astype(float).tolist())
        holm_by_comparison = dict(zip(dimension_source["comparison"], p_holm))
        for comparison, (left_type, right_type) in comparison_types.items():
            source_row = dimension_source[dimension_source["comparison"] == comparison].iloc[0]
            paired = pivot[[left_type, right_type]].dropna().copy()
            paired["difference"] = paired[right_type] - paired[left_type]
            paired = paired.join(song_meta, how="left")
            dz = float(paired["difference"].mean() / paired["difference"].std(ddof=1))
            cluster_low, cluster_high = cluster_bootstrap_ci(
                paired.reset_index(),
                "difference",
                all_clusters,
                BOOTSTRAP_SEED + len(pair_rows),
            )
            pair_rows.append(
                {
                    **source_row.to_dict(),
                    "cohens_dz": dz,
                    "p_holm_5": holm_by_comparison[comparison],
                    "artist_cluster_ci95_low": cluster_low,
                    "artist_cluster_ci95_high": cluster_high,
                    "song_artist_ci_direction_consistent": direction_consistent(
                        float(source_row["ci95_low"]),
                        float(source_row["ci95_high"]),
                        cluster_low,
                        cluster_high,
                    ),
                }
            )
    table_3 = pd.DataFrame(pair_rows)
    table_3.to_csv(OUTPUT / "03_section_type_paired_tests_corrected.csv", index=False)

    # 4. All adjacent transition VA changes with song- and artist-level intervals.
    transition_song_frames: list[pd.DataFrame] = []
    adjacent_with_artist = adjacent.merge(song_meta.reset_index(), on="song_id", how="left")
    transition_rows: list[dict[str, object]] = []
    for row_number, source_row in transition_source.iterrows():
        transition = str(source_row["transition"])
        dimension = str(source_row["dimension"])
        value_column = "delta_valence" if dimension == "valence" else "delta_arousal"
        per_song = (
            adjacent_with_artist[adjacent_with_artist["transition"] == transition]
            .groupby(["song_id", "quota_artist"], as_index=False)[value_column]
            .mean()
            .rename(columns={value_column: "song_transition_mean"})
        )
        transition_song_frames.append(per_song.assign(transition=transition, dimension=dimension))
        cluster_low, cluster_high = cluster_bootstrap_ci(
            per_song,
            "song_transition_mean",
            all_clusters,
            BOOTSTRAP_SEED + 100 + row_number,
        )
        transition_rows.append(
            {
                **source_row.to_dict(),
                "artist_cluster_ci95_low": cluster_low,
                "artist_cluster_ci95_high": cluster_high,
                "song_artist_ci_direction_consistent": direction_consistent(
                    float(source_row["ci95_low"]),
                    float(source_row["ci95_high"]),
                    cluster_low,
                    cluster_high,
                ),
            }
        )
    table_4 = pd.DataFrame(transition_rows)
    priority = {"intro->verse": 0, "chorus->outro": 1}
    table_4["priority"] = table_4["transition"].map(priority).fillna(2)
    table_4 = table_4.sort_values(["priority", "transition", "dimension"]).drop(columns="priority")
    table_4.to_csv(OUTPUT / "04_adjacent_transition_va_changes_corrected.csv", index=False)

    # 5. Arousal-peak types, last-Chorus peaks, and largest-change target types.
    chorus_last_order = (
        sections[sections["section_type"] == "chorus"].groupby("song_id")["section_order"].max()
    )
    peak_chorus = peaks[peaks["section_type"] == "chorus"].copy()
    peak_chorus["last_chorus_order"] = peak_chorus["song_id"].map(chorus_last_order)
    last_chorus_peak_count = int((peak_chorus["section_order"] == peak_chorus["last_chorus_order"]).sum())

    max_events = adjacent.sort_values(["song_id", "va_distance"], ascending=[True, False]).drop_duplicates("song_id")
    peak_records: list[dict[str, object]] = []
    for _, row in peak_summary.sort_values("n_songs", ascending=False).iterrows():
        peak_records.append(
            {
                "record_type": "arousal_peak_section_type",
                "section_type": row["section_type"],
                "song_count": int(row["n_songs"]),
                "song_fraction": float(row["n_songs"]) / 100.0,
                "mean_position": float(row["mean_position"]),
            }
        )
    peak_records.extend(
        [
            {"record_type": "last_chorus_peak_summary", "metric": "chorus_peak_song_count", "value": len(peak_chorus)},
            {"record_type": "last_chorus_peak_summary", "metric": "last_chorus_peak_song_count", "value": last_chorus_peak_count},
            {
                "record_type": "last_chorus_peak_summary",
                "metric": "last_chorus_share_among_chorus_peaks",
                "value": last_chorus_peak_count / len(peak_chorus),
            },
        ]
    )
    for section_type, count in max_events["to_type"].value_counts().items():
        peak_records.append(
            {
                "record_type": "largest_transition_target_type",
                "section_type": section_type,
                "song_count": int(count),
                "song_fraction": float(count) / 100.0,
            }
        )
    table_5 = pd.DataFrame(peak_records)
    table_5.to_csv(OUTPUT / "05_peak_and_largest_transition_distributions_corrected.csv", index=False)

    # 6. Robustness: artist-cluster intervals and rank-band tests.
    robust_records: list[dict[str, object]] = []
    for _, row in table_3.iterrows():
        robust_records.append(
            {
                "record_type": "paired_artist_cluster_bootstrap",
                "comparison": row["comparison"],
                "dimension": row["dimension"],
                "estimate": row["mean_difference"],
                "song_ci95_low": row["ci95_low"],
                "song_ci95_high": row["ci95_high"],
                "artist_cluster_ci95_low": row["artist_cluster_ci95_low"],
                "artist_cluster_ci95_high": row["artist_cluster_ci95_high"],
                "direction_consistent": row["song_artist_ci_direction_consistent"],
            }
        )
    for _, row in table_4.iterrows():
        robust_records.append(
            {
                "record_type": "transition_artist_cluster_bootstrap",
                "comparison": row["transition"],
                "dimension": row["dimension"],
                "estimate": row["song_mean"],
                "song_ci95_low": row["ci95_low"],
                "song_ci95_high": row["ci95_high"],
                "artist_cluster_ci95_low": row["artist_cluster_ci95_low"],
                "artist_cluster_ci95_high": row["artist_cluster_ci95_high"],
                "direction_consistent": row["song_artist_ci_direction_consistent"],
            }
        )

    rank_tests = rank_tests.copy()
    rank_tests["p_holm_10"] = holm_adjust(rank_tests["p_value"].astype(float).tolist())
    for _, row in rank_tests.iterrows():
        robust_records.append({"record_type": "rank_band_kruskal_wallis", **row.to_dict()})
    for _, row in rank_pairs.iterrows():
        robust_records.append({"record_type": "rank_band_pairwise", **row.to_dict()})
    robust_records.append(
        {
            "record_type": "rank_band_section_type_association",
            "test": "chi-square",
            "statistic": rank_assoc["chi_square"],
            "p_value": rank_assoc["p_value"],
            "degrees_of_freedom": rank_assoc["degrees_of_freedom"],
            "cramers_v": rank_assoc["cramers_v"],
        }
    )
    table_6 = pd.DataFrame(robust_records)
    table_6.to_csv(OUTPUT / "06_robustness_artist_cluster_and_rank_tests_corrected.csv", index=False)

    workbook_data = {
        "section_types": json_records(table_1),
        "sequence_summary": {
            "song_count": 100,
            "unique_raw_sequences": len(raw_counter),
            "unique_collapsed_sequences": len(collapsed_counter),
            "adjacent_transition_total": int(transition_counts.sum()),
        },
        "raw_sequences": [
            {
                "sequence": sequence,
                "song_count": count,
                "example_title": example_by_raw[sequence][0],
                "example_artist": example_by_raw[sequence][1],
            }
            for sequence, count in sorted(raw_counter.items(), key=lambda item: (-item[1], item[0]))
        ],
        "collapsed_sequences": [
            {
                "sequence": sequence,
                "song_count": count,
                "example_title": example_by_collapsed[sequence][0],
                "example_artist": example_by_collapsed[sequence][1],
            }
            for sequence, count in sorted(collapsed_counter.items(), key=lambda item: (-item[1], item[0]))
        ],
        "transition_frequencies": [
            {"transition": transition, "count": int(count)}
            for transition, count in transition_counts.items()
        ],
        "paired_tests": json_records(table_3),
        "transition_va": json_records(table_4),
        "peak_type_distribution": json_records(
            table_5[table_5["record_type"] == "arousal_peak_section_type"]
        ),
        "last_chorus_peak_summary": json_records(
            table_5[table_5["record_type"] == "last_chorus_peak_summary"]
        ),
        "largest_transition_targets": json_records(
            table_5[table_5["record_type"] == "largest_transition_target_type"]
        ),
        "paired_robustness": json_records(
            table_6[table_6["record_type"] == "paired_artist_cluster_bootstrap"]
        ),
        "transition_robustness": json_records(
            table_6[table_6["record_type"] == "transition_artist_cluster_bootstrap"]
        ),
        "rank_band_tests": json_records(
            table_6[table_6["record_type"] == "rank_band_kruskal_wallis"]
        ),
        "rank_band_pairwise": json_records(
            table_6[table_6["record_type"] == "rank_band_pairwise"]
        ),
        "rank_section_association": json_records(
            table_6[table_6["record_type"] == "rank_band_section_type_association"]
        ),
    }
    (OUTPUT / "six_results_workbook_data.json").write_text(
        json.dumps(workbook_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "run_id": RUN_ID,
        "created_at": "2026-08-20",
        "source_package": str(PACKAGE.relative_to(ROOT)),
        "source_commit": "0cb93ad96f076ebe06149bd29458e0193db90da8",
        "correction": {
            "songs": 100,
            "sections": 1046,
            "frames": 51448,
            "intro_sections": 95,
            "intro_songs": 94,
            "outro_sections": 100,
            "outro_songs": 100,
            "human_annotation_ids": [
                "89665e9b-27e3-4c43-8030-43e5940f1018",
                "e9046098-3e28-47f5-877c-bdf361477dd4",
            ],
        },
        "sequence_summary": {
            "unique_raw_sequences": len(raw_counter),
            "unique_collapsed_sequences": len(collapsed_counter),
            "adjacent_transition_total": int(transition_counts.sum()),
        },
        "peak_summary": {
            "chorus_peak_songs": int(len(peak_chorus)),
            "last_chorus_peak_songs": last_chorus_peak_count,
        },
        "bootstrap": {
            "unit": "quota_artist",
            "clusters": 60,
            "draws": BOOTSTRAP_DRAWS,
            "seed_base": BOOTSTRAP_SEED,
            "retain_all_songs_for_sampled_artist": True,
        },
        "files": {},
    }
    for path in sorted(OUTPUT.glob("*.csv")):
        manifest["files"][path.name] = {"sha256": sha256(path), "rows": len(pd.read_csv(path))}
    (OUTPUT / "six_results_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
