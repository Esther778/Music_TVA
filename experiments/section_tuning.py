#!/usr/bin/env python3
"""Compare section-labeling strategies against one hand annotation.

This is an experiment harness for tuning the MIR pop weak-labeling rules. It
uses the existing MSAF candidate CSV files so parameter sweeps do not need to
rerun the slow segmentation backend.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import segment_music as sm  # noqa: E402


SONG = ROOT / "songs" / "周深-生活总该迎着光亮.mp3"
MSAF_SF = ROOT / "outputs" / "msaf_compare" / "sf_fmc2d.csv"
MSAF_CNMF = ROOT / "outputs" / "msaf_compare" / "cnmf_cnmf.csv"


@dataclass
class ScoredRun:
    name: str
    boundary_mae: float
    label_accuracy: float
    section_count: int
    score: float
    sections: list[sm.Section]


MANUAL = [
    (0.0, 14.0, "A"),
    (14.0, 28.0, "B"),
    (28.0, 43.0, "B"),
    (43.0, 57.0, "C"),
    (57.0, 85.0, "D"),
    (85.0, 100.0, "E"),
    (100.0, 114.0, "B"),
    (114.0, 128.0, "B"),
    (128.0, 143.0, "C"),
    (143.0, 173.0, "D"),
    (173.0, 186.0, "E"),
    (186.0, 200.0, "B"),
    (200.0, 214.0, "D"),
    (214.0, 231.0, "A"),
    (231.0, 242.27, "A"),
]


def parse_seconds(value: str) -> float:
    if ":" not in value:
        return float(value)
    minutes, seconds = value.split(":", 1)
    return int(minutes) * 60 + float(seconds)


def load_sections(path: Path) -> list[sm.Section]:
    rows: list[sm.Section] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                sm.Section(
                    section_id=int(row["section_id"]),
                    start_time=parse_seconds(row["start_time"]),
                    end_time=parse_seconds(row["end_time"]),
                    duration=parse_seconds(row["duration"]),
                    label=row["label"],
                    structure_label=row.get("structure_label", ""),
                )
            )
    return rows


def manual_label_at(time: float) -> str:
    for start, end, label in MANUAL:
        if start <= time < end:
            return label
    return MANUAL[-1][2]


def predicted_label_at(sections: list[sm.Section], time: float) -> str:
    for section in sections:
        if section.start_time <= time < section.end_time:
            return section.structure_label or sm.structure_label_for_semantic_label(section.label) or section.label
    return sections[-1].structure_label or sm.structure_label_for_semantic_label(sections[-1].label)


def boundary_mae(sections: list[sm.Section]) -> float:
    manual_boundaries = [start for start, _end, _label in MANUAL[1:]]
    predicted_boundaries = [section.start_time for section in sections[1:]]
    if not predicted_boundaries:
        return 999.0
    errors = [min(abs(boundary - predicted) for predicted in predicted_boundaries) for boundary in manual_boundaries]
    return sum(errors) / len(errors)


def label_accuracy(sections: list[sm.Section], step: float = 0.5) -> float:
    duration = MANUAL[-1][1]
    correct = 0
    total = 0
    time = 0.0
    while time < duration:
        if manual_label_at(time) == predicted_label_at(sections, time):
            correct += 1
        total += 1
        time += step
    return correct / total


def score_run(name: str, sections: list[sm.Section]) -> ScoredRun:
    mae = boundary_mae(sections)
    acc = label_accuracy(sections)
    count_penalty = abs(len(sections) - len(MANUAL)) * 0.35
    score = mae + count_penalty + (1.0 - acc) * 10.0
    return ScoredRun(name, mae, acc, len(sections), score, sections)


def apply_labels(analysis: dict[str, object], sections: list[sm.Section], lyrics: list[sm.LyricLine] | None = None) -> list[sm.Section]:
    copied = [
        sm.Section(s.section_id, s.start_time, s.end_time, s.duration, s.label, s.structure_label)
        for s in sections
    ]
    labels = sm.semantic_labels_from_mir(analysis, copied, lyrics=lyrics)
    for index, (section, label) in enumerate(zip(copied, labels), start=1):
        section.section_id = index
        section.label = label
        section.structure_label = sm.structure_label_for_semantic_label(label)
    return sm.merge_adjacent_semantic_sections(copied)


def run() -> None:
    analysis = sm.load_analysis_features(SONG)
    sf_sections = load_sections(MSAF_SF)
    cnmf_sections = load_sections(MSAF_CNMF)

    runs = [
        score_run("raw sf/fmc2d labels", sf_sections),
        score_run("raw cnmf/cnmf labels", cnmf_sections),
        score_run("sf + semantic only", apply_labels(analysis, sf_sections)),
        score_run("cnmf + semantic only", apply_labels(analysis, cnmf_sections)),
    ]

    for min_seconds in (3.0, 5.0, 8.0):
        for factor in (1.05, 1.20, 1.35, 1.50):
            refined = sm.refine_boundaries_by_novelty(
                analysis,
                sf_sections,
                min_section_seconds=min_seconds,
                long_section_factor=factor,
            )
            runs.append(score_run(f"sf + mir-pop min={min_seconds:g} factor={factor:g}", apply_labels(analysis, refined)))

    runs.sort(key=lambda item: item.score)
    print("name,boundary_mae,label_accuracy,section_count,score")
    for item in runs:
        print(f"{item.name},{item.boundary_mae:.2f},{item.label_accuracy:.3f},{item.section_count},{item.score:.2f}")

    best = runs[0]
    print()
    print(f"Best: {best.name}")
    for section in best.sections:
        print(
            f"{section.section_id:02d} "
            f"{sm.format_timestamp(section.start_time)}-"
            f"{sm.format_timestamp(section.end_time)} "
            f"{section.label} {section.structure_label}"
        )


if __name__ == "__main__":
    run()
