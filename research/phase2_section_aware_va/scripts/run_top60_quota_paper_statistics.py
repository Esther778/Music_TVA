#!/usr/bin/env python3
"""Paper-oriented statistics for the reviewed top-60 quota corpus."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PHASE2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PHASE2_ROOT / "data/manifests/top60_quota_100_reviewed_manifest_20260813.csv"
DEFAULT_ANNOTATIONS = PHASE2_ROOT / "data/manifests/top60_quota_100_human_annotations_readable_20260813.csv"
DEFAULT_OUTPUT = PHASE2_ROOT / "analysis/top60_quota_100_20260813/paper_statistics"
CHINA_TZ = timezone(timedelta(hours=8))


def ci_mean(values: pd.Series, z: float = 1.96) -> tuple[float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        value = float(arr[0]) if len(arr) else math.nan
        return value, value
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    return mean - z * se, mean + z * se


def bootstrap_ci(values: pd.Series, statistic: str = "median", seed: int = 20260813, samples: int = 10000) -> tuple[float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        value = float(arr[0]) if len(arr) else math.nan
        return value, value
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(samples, len(arr)), replace=True)
    stats = np.median(draws, axis=1) if statistic == "median" else np.mean(draws, axis=1)
    return tuple(float(value) for value in np.quantile(stats, [0.025, 0.975]))


def describe(values: pd.Series) -> dict[str, float | int]:
    arr = pd.to_numeric(values, errors="coerce").dropna()
    mean_ci = ci_mean(arr)
    median_ci = bootstrap_ci(arr, "median")
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "mean_ci95_low": mean_ci[0],
        "mean_ci95_high": mean_ci[1],
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(arr.median()),
        "median_ci95_low": median_ci[0],
        "median_ci95_high": median_ci[1],
        "q1": float(arr.quantile(0.25)),
        "q3": float(arr.quantile(0.75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def rank_band(rank: int) -> str:
    if rank <= 10:
        return "rank_01_10"
    if rank <= 30:
        return "rank_11_30"
    return "rank_31_60"


def cramers_v(table: pd.DataFrame) -> float:
    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    if total <= 0:
        return 0.0
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    expected = row @ col / total
    mask = expected > 0
    chi2 = float(((observed[mask] - expected[mask]) ** 2 / expected[mask]).sum())
    r, c = observed.shape
    denom = total * max(1, min(r - 1, c - 1))
    return math.sqrt(chi2 / denom) if denom else 0.0


def write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (idx, p_value) in enumerate(indexed, start=1):
        value = min(1.0, (m - rank + 1) * p_value)
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"song_id": str})
    sections = pd.read_csv(args.annotations, dtype={"song_id": str})
    sections["section_type"] = sections["section_type"].str.replace("-", "_").str.lower()
    sections["start_time_sec"] = pd.to_numeric(sections["start_time_sec"], errors="coerce")
    sections["end_time_sec"] = pd.to_numeric(sections["end_time_sec"], errors="coerce")
    sections["duration_sec"] = pd.to_numeric(sections["duration_sec"], errors="coerce")
    manifest["rank_band"] = manifest["rank"].astype(int).map(rank_band)
    sections = sections.merge(manifest[["song_id", "rank", "quota_artist", "rank_band"]], on="song_id", how="left")

    errors: list[str] = []
    if len(manifest) != 100 or manifest["song_id"].nunique() != 100:
        errors.append("manifest_not_100_unique_songs")
    if sections["song_id"].nunique() != 100:
        errors.append("annotations_not_100_songs")
    if sections[["start_time_sec", "end_time_sec", "duration_sec"]].isna().any().any():
        errors.append("missing_numeric_boundary_values")

    song_rows = []
    for song_id, group in sections.sort_values(["song_id", "section_order"]).groupby("song_id"):
        type_counts = Counter(group["section_type"])
        total_duration = float(group["duration_sec"].sum())
        manifest_row = manifest.loc[manifest["song_id"] == song_id].iloc[0]
        song_rows.append(
            {
                "song_id": song_id,
                "title": manifest_row["title"],
                "artist": manifest_row["artist"],
                "quota_artist": manifest_row["quota_artist"],
                "rank": int(manifest_row["rank"]),
                "rank_band": manifest_row["rank_band"],
                "duration_sec": float(manifest_row["duration_sec"]),
                "section_count": int(len(group)),
                "unique_section_types": int(len(type_counts)),
                "chorus_count": int(type_counts.get("chorus", 0)),
                "verse_count": int(type_counts.get("verse", 0)),
                "bridge_count": int(type_counts.get("bridge", 0)),
                "pre_chorus_count": int(type_counts.get("pre_chorus", 0)),
                "intro_present": int(type_counts.get("intro", 0) > 0),
                "outro_present": int(type_counts.get("outro", 0) > 0),
                "bridge_present": int(type_counts.get("bridge", 0) > 0),
                "pre_chorus_present": int(type_counts.get("pre_chorus", 0) > 0),
                "interlude_present": int(type_counts.get("interlude", 0) > 0),
                "chorus_duration_share": float(group.loc[group["section_type"] == "chorus", "duration_sec"].sum() / total_duration),
                "verse_duration_share": float(group.loc[group["section_type"] == "verse", "duration_sec"].sum() / total_duration),
                "bridge_duration_share": float(group.loc[group["section_type"] == "bridge", "duration_sec"].sum() / total_duration),
                "pre_chorus_duration_share": float(group.loc[group["section_type"] == "pre_chorus", "duration_sec"].sum() / total_duration),
                "interlude_duration_share": float(group.loc[group["section_type"] == "interlude", "duration_sec"].sum() / total_duration),
            }
        )
    song_stats = pd.DataFrame(song_rows)

    descriptive_rows = []
    for variable in [
        "duration_sec",
        "section_count",
        "unique_section_types",
        "chorus_count",
        "verse_count",
        "bridge_count",
        "pre_chorus_count",
        "chorus_duration_share",
        "verse_duration_share",
        "bridge_duration_share",
        "pre_chorus_duration_share",
        "interlude_duration_share",
    ]:
        descriptive_rows.append({"variable": variable, **describe(song_stats[variable])})
    descriptive = pd.DataFrame(descriptive_rows)

    type_counts = (
        sections.groupby("section_type")
        .agg(section_count=("section_type", "size"), song_count=("song_id", "nunique"), mean_duration_sec=("duration_sec", "mean"), median_duration_sec=("duration_sec", "median"))
        .reset_index()
        .sort_values("section_count", ascending=False)
    )
    type_counts["section_share"] = type_counts["section_count"] / len(sections)
    type_counts["song_coverage"] = type_counts["song_count"] / len(song_stats)

    rank_rows = []
    for band, group in song_stats.groupby("rank_band", sort=False):
        for variable in ["duration_sec", "section_count", "chorus_count", "verse_count", "unique_section_types", "chorus_duration_share"]:
            rank_rows.append({"rank_band": band, "variable": variable, **describe(group[variable])})
    rank_descriptive = pd.DataFrame(rank_rows)

    inferential_rows = []
    pairwise_rows = []
    bands = ["rank_01_10", "rank_11_30", "rank_31_60"]
    inferential_variables = [
        "duration_sec",
        "section_count",
        "unique_section_types",
        "chorus_count",
        "verse_count",
        "bridge_count",
        "pre_chorus_count",
        "chorus_duration_share",
        "verse_duration_share",
        "interlude_duration_share",
    ]
    for variable in inferential_variables:
        groups = [song_stats.loc[song_stats["rank_band"] == band, variable].dropna() for band in bands]
        kw = stats.kruskal(*groups)
        inferential_rows.append(
            {
                "comparison": "rank_band",
                "variable": variable,
                "test": "Kruskal-Wallis",
                "statistic": float(kw.statistic),
                "p_value": float(kw.pvalue),
                "n_rank_01_10": int(len(groups[0])),
                "n_rank_11_30": int(len(groups[1])),
                "n_rank_31_60": int(len(groups[2])),
            }
        )
        raw_pairwise = []
        for left_idx, left_band in enumerate(bands):
            for right_band in bands[left_idx + 1 :]:
                left = song_stats.loc[song_stats["rank_band"] == left_band, variable].dropna()
                right = song_stats.loc[song_stats["rank_band"] == right_band, variable].dropna()
                test = stats.mannwhitneyu(left, right, alternative="two-sided")
                raw_pairwise.append(
                    {
                        "variable": variable,
                        "left_band": left_band,
                        "right_band": right_band,
                        "test": "Mann-Whitney U",
                        "statistic": float(test.statistic),
                        "p_value": float(test.pvalue),
                        "left_median": float(left.median()),
                        "right_median": float(right.median()),
                    }
                )
        adjusted = holm_adjust([row["p_value"] for row in raw_pairwise])
        for row, adjusted_p in zip(raw_pairwise, adjusted):
            row["p_holm"] = adjusted_p
            pairwise_rows.append(row)
    inferential = pd.DataFrame(inferential_rows)
    pairwise = pd.DataFrame(pairwise_rows)

    transitions = []
    for song_id, group in sections.sort_values(["song_id", "section_order"]).groupby("song_id"):
        types = group["section_type"].tolist()
        for idx, (left, right) in enumerate(zip(types, types[1:]), start=1):
            transitions.append({"song_id": song_id, "transition_order": idx, "from_type": left, "to_type": right, "transition": f"{left}->{right}"})
    transition_frame = pd.DataFrame(transitions)
    transition_summary = transition_frame["transition"].value_counts().rename_axis("transition").reset_index(name="count")
    transition_summary["share"] = transition_summary["count"] / len(transition_frame)

    type_by_band = pd.crosstab(sections["rank_band"], sections["section_type"])
    chi2, chi_p, chi_dof, _ = stats.chi2_contingency(type_by_band)
    rank_section_assoc = {
        "chi_square": float(chi2),
        "degrees_of_freedom": int(chi_dof),
        "p_value": float(chi_p),
        "cramers_v": cramers_v(type_by_band),
        "rank_bands": list(type_by_band.index),
        "section_types": list(type_by_band.columns),
    }

    song_stats_path = args.output_dir / "paper_song_level_features.csv"
    descriptive_path = args.output_dir / "paper_song_level_descriptives.csv"
    type_path = args.output_dir / "paper_section_type_distribution.csv"
    rank_path = args.output_dir / "paper_rank_band_descriptives.csv"
    inferential_path = args.output_dir / "paper_rank_band_inferential_tests.csv"
    pairwise_path = args.output_dir / "paper_rank_band_pairwise_tests.csv"
    transition_path = args.output_dir / "paper_transition_distribution.csv"
    association_path = args.output_dir / "paper_rank_section_type_association.json"
    report_path = args.output_dir / "PAPER_STATISTICS_SUMMARY_ZH.md"
    manifest_path = args.output_dir / "paper_statistics_manifest.json"

    write_table(song_stats_path, song_stats)
    write_table(descriptive_path, descriptive)
    write_table(type_path, type_counts)
    write_table(rank_path, rank_descriptive)
    write_table(inferential_path, inferential)
    write_table(pairwise_path, pairwise)
    write_table(transition_path, transition_summary)
    association_path.write_text(json.dumps(rank_section_assoc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    section_count_desc = descriptive.loc[descriptive["variable"] == "section_count"].iloc[0]
    duration_desc = descriptive.loc[descriptive["variable"] == "duration_sec"].iloc[0]
    chorus_row = type_counts.loc[type_counts["section_type"] == "chorus"].iloc[0]
    verse_row = type_counts.loc[type_counts["section_type"] == "verse"].iloc[0]
    top_transitions = transition_summary.head(8)
    report = [
        "# 100首QQ音乐歌手榜配额语料结构统计（论文用）",
        "",
        f"- 样本：60位歌手、100首歌曲、{len(sections)}个人工质检section。",
        "- 抽样规则：1-10名每人3首，11-30名每人2首，31-60名每人1首；合唱歌按补缺优先分配。",
        "- 统计单位：歌曲为主要独立单位；section级统计作为结构分布描述，不当作独立歌曲样本。",
        "- 数据来源：Supabase `human_annotations_readable` 冻结快照。",
        "",
        "## 主要描述统计",
        "",
        f"- 歌曲时长均值 {duration_desc['mean']:.2f}s，95% CI [{duration_desc['mean_ci95_low']:.2f}, {duration_desc['mean_ci95_high']:.2f}]；中位数 {duration_desc['median']:.2f}s。",
        f"- 每首section数均值 {section_count_desc['mean']:.2f}，95% CI [{section_count_desc['mean_ci95_low']:.2f}, {section_count_desc['mean_ci95_high']:.2f}]；中位数 {section_count_desc['median']:.0f}。",
        f"- Chorus共 {int(chorus_row['section_count'])} 段，覆盖 {int(chorus_row['song_count'])}/100 首；Verse共 {int(verse_row['section_count'])} 段，覆盖 {int(verse_row['song_count'])}/100 首。",
        "",
        "## 高频相邻转移",
        "",
    ]
    for _, row in top_transitions.iterrows():
        report.append(f"- {row['transition']}: {int(row['count'])} 次，占全部相邻转移 {row['share']:.2%}")
    significant = inferential.loc[inferential["p_value"] < 0.05]
    report.extend(["", "## 排名分层差异", ""])
    report.append(
        f"- Section type 与排名层的列联关系很弱：Cramer's V = {rank_section_assoc['cramers_v']:.3f}，"
        f"χ²({rank_section_assoc['degrees_of_freedom']}) = {rank_section_assoc['chi_square']:.2f}，p = {rank_section_assoc['p_value']:.3g}。"
    )
    if len(significant):
        for _, row in significant.iterrows():
            report.append(f"- {row['variable']} 在排名层之间存在探索性差异：H = {row['statistic']:.2f}, p = {row['p_value']:.3g}。")
    else:
        report.append("- 歌曲层结构指标的Kruskal-Wallis检验未发现p<.05的排名层差异。")
    report.extend(
        [
            "",
            "## 写作边界",
            "",
            "- 本报告只统计人工质检的section边界和类型，不包含frame-level VA预测。",
            "- 若论文要报告情绪VA轨迹、Chorus-Valence/Arousal差异，需要另跑新版100首完整VA推断与对齐，不能复用旧99首VA结果。",
        ]
    )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest = {
        "created_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "songs": int(len(song_stats)),
        "sections": int(len(sections)),
        "rank_section_type_cramers_v": rank_section_assoc["cramers_v"],
        "outputs": {
            "song_level_features": str(song_stats_path.resolve()),
            "song_level_descriptives": str(descriptive_path.resolve()),
            "section_type_distribution": str(type_path.resolve()),
            "rank_band_descriptives": str(rank_path.resolve()),
            "rank_band_inferential_tests": str(inferential_path.resolve()),
            "rank_band_pairwise_tests": str(pairwise_path.resolve()),
            "transition_distribution": str(transition_path.resolve()),
            "rank_section_type_association": str(association_path.resolve()),
            "paper_summary_zh": str(report_path.resolve()),
        },
        "note": "Statistics are based on reviewed section annotations only; frame-level VA inference must be run separately for emotion-trajectory claims.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
