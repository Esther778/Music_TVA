from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import functional_section_pipeline as pipeline


@dataclass
class EvaluationReport:
    label_accuracy: float
    boundary_mae: float
    section_count: int
    manual_count: int


def parse_manual_csv(path: Path) -> list[tuple[float, float, str]]:
    rows: list[tuple[float, float, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            start = pipeline.base.parse_timestamp(row.get("start_time") or row.get("manual_start") or row["start"])
            end = pipeline.base.parse_timestamp(row.get("end_time") or row.get("manual_end") or row["end"])
            label = row.get("structure_label") or row.get("manual_label") or row.get("label") or row["section_type"]
            rows.append((start, end, label))
    return rows


def evaluate_sections(
    sections: list[pipeline.SectionCandidate],
    manual: list[tuple[float, float, str]],
    step: float = 0.5,
) -> EvaluationReport:
    if not manual:
        return EvaluationReport(0.0, 999.0, len(sections), 0)
    duration = manual[-1][1]
    correct = 0
    total = 0
    time = 0.0
    while time < duration:
        if _manual_label_at(manual, time) == _section_label_at(sections, time):
            correct += 1
        total += 1
        time += step
    manual_boundaries = [start for start, _end, _label in manual[1:]]
    predicted_boundaries = [section.start for section in sections[1:]]
    if predicted_boundaries:
        boundary_mae = sum(min(abs(boundary - predicted) for predicted in predicted_boundaries) for boundary in manual_boundaries)
        boundary_mae /= max(len(manual_boundaries), 1)
    else:
        boundary_mae = 999.0
    return EvaluationReport(round(correct / max(total, 1), 3), round(boundary_mae, 2), len(sections), len(manual))


def _manual_label_at(manual: list[tuple[float, float, str]], time: float) -> str:
    for start, end, label in manual:
        if start <= time < end:
            return label
    return manual[-1][2]


def _section_label_at(sections: list[pipeline.SectionCandidate], time: float) -> str:
    for section in sections:
        if section.start <= time < section.end:
            return pipeline.structure_label(section.section_type)
    return pipeline.structure_label(sections[-1].section_type)

