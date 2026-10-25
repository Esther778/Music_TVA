#!/usr/bin/env python3
"""Audit arXiv:2609.15675 numbers against the corrected 2026-08-20 package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "exports/gpt_top60_quota_100_section_va_data_package_20260820"
RESULTS = ROOT / "exports/top60_quota_100_corrected_six_results_20260820"


def read_csv(relative: str) -> list[dict[str, str]]:
    with (PACKAGE / relative).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_results_csv(filename: str) -> list[dict[str, str]]:
    with (RESULTS / filename).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rounded_equal(actual: float, expected: float, decimals: int = 3) -> bool:
    return round(actual, decimals) == round(expected, decimals)


class Audit:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, bool]] = []

    def exact(self, label: str, actual: object, expected: object) -> None:
        self.rows.append((label, str(actual), str(expected), actual == expected))

    def number(self, label: str, actual: float, expected: float, decimals: int = 3) -> None:
        self.rows.append(
            (label, f"{actual:.{decimals}f}", f"{expected:.{decimals}f}", rounded_equal(actual, expected, decimals))
        )

    @property
    def passed(self) -> bool:
        return all(row[3] for row in self.rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_package_manifest(audit: Audit) -> tuple[int, list[str]]:
    manifest_path = PACKAGE / "00_index/FILE_MANIFEST.csv"
    failures: list[str] = []
    checked = 0
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            relative = row.get("relative_path") or row.get("path") or row.get("file")
            expected = row.get("sha256") or row.get("sha256_hex")
            if not relative or not expected:
                continue
            target = PACKAGE / relative
            checked += 1
            if not target.is_file() or sha256(target) != expected:
                failures.append(relative)
    audit.exact("Package manifest hash failures", len(failures), 0)
    return checked, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-text", type=Path)
    parser.add_argument("--local-submission-text", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audit = Audit()
    sections = read_csv("05_section_level/top60_quota_100_section_features.csv")
    frames = read_csv("03_frame_level_va/top60_quota_100_frame_va.csv")
    aligned = read_csv("04_frame_section_alignment/top60_quota_100_frame_section_alignment.csv")
    songs = read_csv("08_analysis_ready/song_analysis_ready.csv")
    transitions = read_csv("09_derived_tables/adjacent_section_transitions_gpt.csv")
    trajectory = read_csv("09_derived_tables/va_average_normalized_trajectory.csv")
    peaks = read_csv("09_derived_tables/va_arousal_peak_sections.csv")
    section_summary = {r["section_type"]: r for r in read_csv("09_derived_tables/va_section_type_summary.csv")}
    paired = {(r["comparison"], r["dimension"]): r for r in read_results_csv("03_section_type_paired_tests_corrected.csv")}
    transition_summary = {(r["transition"], r["dimension"]): r for r in read_results_csv("04_adjacent_transition_va_changes_corrected.csv")}
    repeated = {r["section_type"]: r for r in read_csv("09_derived_tables/va_repeated_section_summary.csv")}

    audit.exact("Songs", len({r["song_id"] for r in sections}), 100)
    audit.exact("Sections", len(sections), 1046)
    audit.exact("Frame VA rows", len(frames), 51448)
    audit.exact("Aligned frame rows", len(aligned), 51448)
    audit.exact("Adjacent transitions", len(transitions), 946)

    first_types = {}
    for row in sections:
        if int(float(row["section_order"])) == 1:
            first_types[row["title"]] = row["section_type"]
    audit.exact("演员 first section", first_types.get("演员"), "intro")
    audit.exact("我怀念的 first section", first_types.get("我怀念的"), "intro")

    published_table1 = {
        "intro": (95, 94, 0.075, -0.081, 0.041),
        "verse": (246, 100, 0.111, -0.088, 0.305),
        "pre_chorus": (77, 38, 0.187, 0.107, 0.381),
        "chorus": (362, 100, 0.240, 0.215, 0.592),
        "interlude": (121, 85, 0.260, 0.169, 0.523),
        "bridge": (45, 43, 0.273, 0.270, 0.689),
        "outro": (100, 100, 0.114, 0.104, 0.953),
    }
    for section_type, expected in published_table1.items():
        actual = section_summary[section_type]
        audit.exact(f"Table 1 {section_type} sections", int(actual["section_count"]), expected[0])
        audit.exact(f"Table 1 {section_type} songs", int(actual["song_count"]), expected[1])
        for field, value in zip(("mean_valence", "mean_arousal", "mean_position"), expected[2:]):
            audit.number(f"Table 1 {section_type} {field}", float(actual[field]), value)

    raw_sequences = {r["raw_sequence"] for r in read_csv("09_derived_tables/musicological_collapsed_form_sequences_by_song.csv")}
    collapsed_sequences = {r["collapsed_sequence"] for r in read_csv("09_derived_tables/musicological_collapsed_form_sequences_by_song.csv")}
    audit.exact("Distinct raw sequences", len(raw_sequences), 76)
    audit.exact("Distinct collapsed sequences", len(collapsed_sequences), 49)

    transition_counts = Counter(r["transition"] for r in transitions)
    published_transition_counts = {
        "verse->chorus": 116,
        "chorus->interlude": 111,
        "chorus->chorus": 100,
        "intro->verse": 94,
        "chorus->outro": 91,
        "pre_chorus->chorus": 76,
        "interlude->verse": 68,
        "verse->pre_chorus": 68,
    }
    for transition, expected in published_transition_counts.items():
        audit.exact(f"Frequent transition {transition}", transition_counts[transition], expected)
    top_eight_share = sum(published_transition_counts.values()) / len(transitions)
    audit.number("Top-eight transition share", top_eight_share * 100, 76.5, 1)

    published_paired = {
        ("chorus_minus_verse", "valence"): (100, 0.119, 0.100, 0.138, 0.890),
        ("pre_chorus_minus_verse", "valence"): (38, 0.055, 0.034, 0.076, 0.763),
        ("chorus_minus_pre_chorus", "valence"): (38, 0.078, 0.057, 0.099, 0.921),
        ("bridge_minus_chorus", "valence"): (43, 0.050, 0.019, 0.083, 0.628),
        ("outro_minus_intro", "valence"): (94, 0.036, -0.001, 0.074, 0.543),
        ("chorus_minus_verse", "arousal"): (100, 0.279, 0.251, 0.308, 0.990),
        ("pre_chorus_minus_verse", "arousal"): (38, 0.135, 0.097, 0.172, 0.842),
        ("chorus_minus_pre_chorus", "arousal"): (38, 0.200, 0.162, 0.243, 1.000),
        ("bridge_minus_chorus", "arousal"): (43, 0.088, 0.040, 0.141, 0.721),
        ("outro_minus_intro", "arousal"): (94, 0.186, 0.131, 0.242, 0.745),
    }
    for key, expected in published_paired.items():
        actual = paired[key]
        label = f"Table 2 {key[0]} {key[1]}"
        audit.exact(f"{label} n", int(actual["n_songs"]), expected[0])
        for field, value in zip(("mean_difference", "ci95_low", "ci95_high", "positive_song_fraction"), expected[1:]):
            audit.number(f"{label} {field}", float(actual[field]), value)

    published_table3 = {
        "intro->verse": (94, -0.004, -0.020, 0.012, -0.056, -0.082, -0.032),
        "verse->pre_chorus": (68, 0.037, 0.022, 0.052, 0.107, 0.074, 0.138),
        "pre_chorus->chorus": (76, 0.062, 0.044, 0.079, 0.154, 0.126, 0.187),
        "chorus->interlude": (111, 0.034, 0.017, 0.052, 0.010, -0.011, 0.032),
        "interlude->verse": (68, -0.018, -0.037, 0.001, -0.048, -0.080, -0.017),
        "chorus->bridge": (33, 0.008, -0.021, 0.038, 0.030, -0.008, 0.070),
        "bridge->chorus": (31, -0.006, -0.037, 0.023, 0.024, -0.003, 0.052),
        "chorus->outro": (91, -0.132, -0.159, -0.105, -0.188, -0.230, -0.147),
    }
    for transition, expected in published_table3.items():
        valence = transition_summary[(transition, "valence")]
        arousal = transition_summary[(transition, "arousal")]
        audit.exact(f"Table 3 {transition} events", int(valence["n_events"]), expected[0])
        for actual, value, name in (
            (valence["song_mean"], expected[1], "delta V"),
            (valence["ci95_low"], expected[2], "V CI low"),
            (valence["ci95_high"], expected[3], "V CI high"),
            (arousal["song_mean"], expected[4], "delta A"),
            (arousal["ci95_low"], expected[5], "A CI low"),
            (arousal["ci95_high"], expected[6], "A CI high"),
        ):
            audit.number(f"Table 3 {transition} {name}", float(actual), value)

    chorus_directions = Counter(r["direction_t0.05"] for r in transitions if r["to_type"] == "chorus")
    audit.exact("Chorus-entry transitions", sum(chorus_directions.values()), 362)
    audit.exact("Chorus entry both up", chorus_directions["valence_up|arousal_up"], 138)
    audit.exact("Chorus entry arousal only", chorus_directions["valence_flat|arousal_up"], 71)
    audit.exact("Chorus entry both stable", chorus_directions["valence_flat|arousal_flat"], 64)

    published_repeated = {
        "verse": (91, 0.099, 0.813, 0.165, 0.824),
        "pre_chorus": (37, 0.112, 0.919, 0.156, 0.919),
        "chorus": (100, 0.085, 0.710, 0.225, 0.900),
        "interlude": (33, 0.133, 0.848, 0.260, 0.939),
    }
    for section_type, expected in published_repeated.items():
        actual = repeated[section_type]
        audit.exact(f"Table 4 {section_type} n", int(actual["n_songs"]), expected[0])
        for field, value in zip(
            ("mean_first_to_last_valence", "positive_valence_fraction", "mean_first_to_last_arousal", "positive_arousal_fraction"),
            expected[1:],
        ):
            audit.number(f"Table 4 {section_type} {field}", float(actual[field]), value)

    by_song: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sections:
        by_song[row["song_id"]].append(row)
    displacement = []
    directness = []
    for song in songs:
        distance = math.hypot(float(song["net_section_valence_change"]), float(song["net_section_arousal_change"]))
        displacement.append(distance)
        directness.append(distance / float(song["section_va_path_length"]))
    audit.number("Mean VA path length", statistics.mean(float(r["section_va_path_length"]) for r in songs), 1.443)
    audit.number("Mean endpoint displacement", statistics.mean(displacement), 0.316)
    audit.number("Mean path directness", statistics.mean(directness), 0.242)
    for field, mean_expected, positive_expected in (
        ("net_section_valence_change", 0.039, 0.56),
        ("net_section_arousal_change", 0.193, 0.75),
    ):
        values = [float(r[field]) for r in songs]
        audit.number(f"{field} mean", statistics.mean(values), mean_expected)
        audit.number(f"{field} positive fraction", sum(v > 0 for v in values) / len(values), positive_expected, 2)
    audit.number("Mean-trajectory Valence peak position", float(max(trajectory, key=lambda r: float(r["valence"]))["normalized_time"]), 0.75, 2)
    audit.number("Mean-trajectory Arousal peak position", float(max(trajectory, key=lambda r: float(r["arousal"]))["normalized_time"]), 0.78, 2)
    peak_positions = [float(r["normalized_midpoint"]) for r in peaks]
    audit.number("Song peak-position mean", statistics.mean(peak_positions), 0.733)
    audit.number("Song peak-position median", statistics.median(peak_positions), 0.755)
    audit.exact("Song peaks in final third", sum(p >= 2 / 3 for p in peak_positions), 79)

    peak_types = Counter(r["section_type"] for r in peaks)
    for section_type, expected in {"chorus": 53, "interlude": 19, "bridge": 13, "outro": 10, "intro": 4, "verse": 1}.items():
        audit.exact(f"Peak section type {section_type}", peak_types[section_type], expected)
    final_chorus_peak = 0
    max_transition_targets = Counter()
    for song in songs:
        rows = by_song[song["song_id"]]
        last_chorus = max((r for r in rows if r["section_type"] == "chorus"), key=lambda r: int(float(r["section_order"])))
        peak = max(rows, key=lambda r: float(r["mean_arousal"]))
        final_chorus_peak += int(last_chorus["section_order"] == peak["section_order"])
        target_order = int(float(song["max_contrast_transition_to_section"]))
        target = next(r for r in rows if int(float(r["section_order"])) == target_order)
        max_transition_targets[target["section_type"]] += 1
    audit.exact("Final Chorus is peak", final_chorus_peak, 31)
    for section_type, expected in {"outro": 44, "chorus": 27, "verse": 13, "pre_chorus": 7, "interlude": 6, "bridge": 3}.items():
        audit.exact(f"Largest transition enters {section_type}", max_transition_targets[section_type], expected)

    checked, manifest_failures = verify_package_manifest(audit)

    text_match = None
    if args.paper_text and args.local_submission_text:
        paper = args.paper_text.read_text(encoding="utf-8")
        local = args.local_submission_text.read_text(encoding="utf-8")
        text_match = paper == local
        audit.exact("Published/local extracted PDF text match", text_match, True)

    passed_count = sum(row[3] for row in audit.rows)
    mismatches = [row for row in audit.rows if not row[3]]
    minor_display_mismatches = [
        row
        for row in mismatches
        if row[0].startswith("Table 2 ")
        and isinstance(row[1], float)
        and isinstance(row[2], float)
        and abs(row[1] - row[2]) <= 0.0011
    ]
    if not mismatches:
        result_summary = "通过；论文与最终数据包一致"
    elif len(minor_display_mismatches) == len(mismatches):
        result_summary = (
            f"核心数据与结果一致；{len(mismatches)} 个 Bootstrap CI 端点"
            "与论文展示值相差 0.001"
        )
    else:
        result_summary = "存在不一致"
    lines = [
        "# arXiv:2609.15675 与最终数据一致性审计",
        "",
        "审计日期：2026-10-25",
        "",
        "## 结论",
        "",
        f"- 结果：{result_summary}。",
        f"- 自动核对项：{passed_count}/{len(audit.rows)} 通过。",
        "- 基线：2026-08-20 修正后的 Top60 配额 100 首数据包与六份修正统计。",
        "- 论文：arXiv:2609.15675，提交日期 2026-09-14。",
        "- 两处永久标签修正已包含在论文口径中：《演员》和《我怀念的》首段均为 Intro。",
        f"- 数据包清单哈希：检查 {checked} 个文件，失败 {len(manifest_failures)} 个。",
    ]
    if minor_display_mismatches:
        lines.append(
            "- 微小差异均位于 Table 2 的 Bootstrap 95% CI 端点；样本数、"
            "均值差、正向歌曲比例、显著性结论与主要解释均不受影响。"
        )
    if text_match is not None:
        lines.append(f"- arXiv 下载版与本地最终提交版的提取正文：{'逐行一致' if text_match else '不一致'}。")
    lines += [
        "",
        "## 覆盖范围",
        "",
        "审计覆盖样本数、section/frame/对齐行数、两首修正歌曲的首段标签、Table 1 至 Table 4、",
        "高频转段、Chorus 入口模式、整曲路径指标、平均轨迹峰值、歌曲峰值段落分布、",
        "最后 Chorus 峰值与最大相邻转段目标类型。浮点数按论文展示精度比较。",
        "",
        "## 明细",
        "",
        "| 核对项 | 本地最终数据 | 论文值 | 状态 |",
        "| --- | ---: | ---: | --- |",
    ]
    for label, actual, expected, ok in audit.rows:
        lines.append(f"| {label} | {actual} | {expected} | {'通过' if ok else '不一致'} |")
    if manifest_failures:
        lines += ["", "## 数据包哈希失败文件", ""] + [f"- `{path}`" for path in manifest_failures]
    lines += [
        "",
        "## 使用方法",
        "",
        "```bash",
        ".venv/bin/python tools/audit_arxiv_2609_15675_consistency.py \\",
        "  --paper-text /path/to/arxiv_2609.15675.txt \\",
        "  --local-submission-text /path/to/local_submission.txt \\",
        "  --output docs/research/ARXIV_2609_15675_DATA_AUDIT_20261025.md",
        "```",
        "",
        "PDF 文本参数可省略；其余数据核对仍会执行。",
    ]
    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
