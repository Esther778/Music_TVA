#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import functional_section_pipeline as pipeline


@dataclass
class AcousticFirstRow:
    song_id: str
    section_index: int
    start_time: str
    end_time: str
    duration: str
    section_type: str
    structure_label: str
    lyric_evidence: str
    vocal_evidence: str
    acoustic_evidence: str
    acoustic_role_score: float
    lyric_role_score: float
    final_confidence: float
    need_human_review: bool


def main() -> None:
    parser = argparse.ArgumentParser(description="Acoustic-first section type experiment.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--input-sections", required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    features = pipeline.load_acoustic_features(Path(args.audio))
    sections = read_sections(Path(args.input_sections))
    rows = classify_sections(args.song_id, sections, features)
    write_rows(Path(args.output), rows)
    for row in rows:
        print(
            f"{row.section_index:02d} {row.start_time}-{row.end_time} "
            f"{row.section_type} acoustic={row.acoustic_role_score:.2f} "
            f"lyric={row.lyric_role_score:.2f} confidence={row.final_confidence:.2f} "
            f"review={str(row.need_human_review).lower()}"
        )


def read_sections(path: Path) -> list[pipeline.SectionCandidate]:
    result: list[pipeline.SectionCandidate] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result.append(
                pipeline.SectionCandidate(
                    start=pipeline.base.parse_timestamp(row["start_time"]),
                    end=pipeline.base.parse_timestamp(row["end_time"]),
                    section_type=row["section_type"],
                    text="",
                    lyric_evidence=row["lyric_evidence"],
                    vocal_evidence=row["vocal_evidence"],
                    acoustic_evidence=row["acoustic_evidence"],
                    boundary_confidence=float(row["boundary_confidence"]),
                    type_confidence=float(row["type_confidence"]),
                    need_human_review=row["need_human_review"].lower() == "true",
                )
            )
    return result


def classify_sections(
    song_id: str,
    sections: list[pipeline.SectionCandidate],
    features: dict[str, object],
) -> list[AcousticFirstRow]:
    duration = float(features["duration"])
    profiles = [profile(section, features) for section in sections]
    rms_values = [item["rms"] for item in profiles]
    onset_values = [item["onset"] for item in profiles]
    novelty_values = [item["novelty"] for item in profiles]
    centroid_values = [item["centroid"] for item in profiles]

    rows: list[AcousticFirstRow] = []
    for index, section in enumerate(sections):
        p = profiles[index]
        position = section.start / max(duration, 1.0)
        energy_rank = rank(p["rms"], rms_values)
        onset_rank = rank(p["onset"], onset_values)
        novelty_rank = rank(p["novelty"], novelty_values)
        centroid_rank = rank(p["centroid"], centroid_values)
        acoustic_intensity = 0.45 * energy_rank + 0.25 * centroid_rank + 0.2 * onset_rank + 0.1 * novelty_rank
        sung = has_complete_vocal(section)
        low_vocal = not sung or has_marker(section, ("主唱退出", "低人声", "弱化", "无完整主唱"))

        old_type = normalize_type(section.section_type)
        lyric_score = lyric_support(old_type, section)
        acoustic_type, acoustic_reason, acoustic_score = choose_acoustic_type(
            section=section,
            index=index,
            position=position,
            acoustic_intensity=acoustic_intensity,
            energy_rank=energy_rank,
            onset_rank=onset_rank,
            novelty_rank=novelty_rank,
            sung=sung,
            low_vocal=low_vocal,
            old_type=old_type,
            previous_type=normalize_type(sections[index - 1].section_type) if index > 0 else "",
            next_type=normalize_type(sections[index + 1].section_type) if index + 1 < len(sections) else "",
            previous_section=sections[index - 1] if index > 0 else None,
            next_section=sections[index + 1] if index + 1 < len(sections) else None,
            total=len(sections),
        )
        final_type, final_reason, final_confidence = reconcile_type(
            old_type=old_type,
            acoustic_type=acoustic_type,
            acoustic_score=acoustic_score,
            lyric_score=lyric_score,
            section=section,
        )
        review = final_confidence < 0.72 or final_type != old_type
        evidence = (
            f"声学先行：energy_rank={energy_rank:.2f}, onset_rank={onset_rank:.2f}, "
            f"novelty_rank={novelty_rank:.2f}, centroid_rank={centroid_rank:.2f}；"
            f"{acoustic_reason}；{final_reason}"
        )
        rows.append(
            AcousticFirstRow(
                song_id=song_id,
                section_index=index + 1,
                start_time=pipeline.base.format_timestamp(section.start),
                end_time=pipeline.base.format_timestamp(section.end),
                duration=pipeline.base.format_timestamp(section.end - section.start),
                section_type=final_type,
                structure_label=structure_label(final_type),
                lyric_evidence=section.lyric_evidence,
                vocal_evidence=section.vocal_evidence,
                acoustic_evidence=evidence,
                acoustic_role_score=round(acoustic_score, 2),
                lyric_role_score=round(lyric_score, 2),
                final_confidence=round(final_confidence, 2),
                need_human_review=review,
            )
        )
    return rows


def profile(section: pipeline.SectionCandidate, features: dict[str, object]) -> dict[str, float]:
    return {
        "rms": pipeline.average_curve(features, "rms", section.start, section.end),
        "onset": pipeline.average_curve(features, "onset", section.start, section.end),
        "novelty": pipeline.average_curve(features, "novelty", section.start, section.end),
        "centroid": pipeline.average_curve(features, "centroid", section.start, section.end),
    }


def rank(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    low = min(values)
    high = max(values)
    if high <= low:
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low)))


def choose_acoustic_type(
    section: pipeline.SectionCandidate,
    index: int,
    position: float,
    acoustic_intensity: float,
    energy_rank: float,
    onset_rank: float,
    novelty_rank: float,
    sung: bool,
    low_vocal: bool,
    old_type: str,
    previous_type: str,
    next_type: str,
    previous_section: pipeline.SectionCandidate | None,
    next_section: pipeline.SectionCandidate | None,
    total: int,
) -> tuple[str, str, float]:
    if not sung and position < 0.18:
        return "intro", "早段低/无人声，按 intro", 0.82
    if not sung and (position > 0.78 or index == total - 1):
        return "outro", "尾段低/无人声，按 outro", 0.82
    if not sung:
        return "bridge", "中段低/无人声，按 bridge", 0.78

    if low_vocal and 0.18 <= position <= 0.78:
        return "bridge", "主唱活动弱且处于中段，按 bridge", 0.72

    if is_pre_chorus_candidate(
        section,
        old_type,
        previous_type,
        next_type,
        previous_section,
        next_section,
        acoustic_intensity,
        energy_rank,
        onset_rank,
        novelty_rank,
        position,
    ):
        return "pre_chorus", "位于进入 chorus 前，歌词/结构有蓄力或连接功能，声学有推进但不是核心 hook", 0.76

    if acoustic_intensity >= 0.62 or energy_rank >= 0.68:
        if old_type == "bridge" and has_marker(section, ("新材料", "转折", "对比", "重新蓄力")) and position >= 0.45:
            return "bridge", "声学强但歌词/结构显示中后段对比材料，保留 lyrical bridge", 0.70
        return "chorus", "声学强度处于高位，优先解释为 chorus/final chorus", 0.78

    if old_type == "pre_chorus" and novelty_rank >= 0.25:
        return "pre_chorus", "保留副歌前蓄力：声学变化不强但结构位置成立", 0.72

    if old_type == "bridge" and position >= 0.35:
        return "bridge", "中后段对比/连接材料，按 bridge", 0.66

    return "verse", "声学强度较低或叙事型稳定段，按 verse", 0.66


def is_pre_chorus_candidate(
    section: pipeline.SectionCandidate,
    old_type: str,
    previous_type: str,
    next_type: str,
    previous_section: pipeline.SectionCandidate | None,
    next_section: pipeline.SectionCandidate | None,
    acoustic_intensity: float,
    energy_rank: float,
    onset_rank: float,
    novelty_rank: float,
    position: float,
) -> bool:
    if next_type != "chorus":
        return False
    if previous_type not in {"verse", "bridge", "intro", "chorus"}:
        return False
    evidence = f"{section.lyric_evidence} {section.vocal_evidence} {section.acoustic_evidence}"
    has_buildup_language = any(
        marker in evidence
        for marker in ("蓄力", "铺垫", "過渡", "过渡", "预备进入", "进入副歌", "转向", "总结", "连接")
    )
    if old_type == "pre_chorus" and has_buildup_language:
        return True
    if old_type == "bridge" and has_buildup_language and position < 0.65:
        return True
    if not previous_section or not next_section:
        return False
    previous_is_core_chorus = normalize_type(previous_section.section_type) == "chorus"
    next_is_core_chorus = normalize_type(next_section.section_type) == "chorus"
    if previous_is_core_chorus and next_is_core_chorus:
        return False
    moderate_to_high_motion = acoustic_intensity >= 0.48 or onset_rank >= 0.55 or novelty_rank >= 0.45
    not_full_chorus_energy = energy_rank < 0.86
    return has_buildup_language and moderate_to_high_motion and not_full_chorus_energy


def reconcile_type(
    old_type: str,
    acoustic_type: str,
    acoustic_score: float,
    lyric_score: float,
    section: pipeline.SectionCandidate,
) -> tuple[str, str, float]:
    if acoustic_type == old_type:
        return acoustic_type, "声学和歌词初稿一致", max(acoustic_score, lyric_score)
    if acoustic_type == "pre_chorus" and acoustic_score >= 0.72:
        return "pre_chorus", "pre_chorus 结构约束成立，保留副歌前蓄力段", max(acoustic_score, lyric_score)
    if acoustic_score >= lyric_score + 0.12:
        return acoustic_type, "声学置信度明显高于歌词初稿，采用声学类型", acoustic_score
    if acoustic_type == "chorus" and old_type in {"bridge", "pre_chorus"} and acoustic_score >= 0.74:
        return "chorus", "高能量/高密度声学证据压过原类型，改为 chorus", acoustic_score
    if old_type == "bridge" and has_marker(section, ("新材料", "转折", "对比", "重新蓄力")):
        return "bridge", "歌词/结构明确为对比转折，保留 bridge", max(lyric_score, 0.68)
    return old_type, "声学证据不足以推翻歌词初稿", max(lyric_score, acoustic_score - 0.08)


def lyric_support(section_type: str, section: pipeline.SectionCandidate) -> float:
    evidence = f"{section.lyric_evidence} {section.vocal_evidence} {section.acoustic_evidence}"
    if section_type == "chorus" and has_marker(section, ("hook", "副歌", "final", "主题")):
        return 0.78
    if section_type == "bridge" and has_marker(section, ("转折", "连接", "新材料", "对比", "无歌词")):
        return 0.76
    if section_type == "pre_chorus" and has_marker(section, ("蓄力", "铺垫", "进入副歌")):
        return 0.72
    if section_type == "verse" and has_marker(section, ("叙事", "背景", "回忆")):
        return 0.72
    if section_type in {"intro", "outro"} and has_marker(section, ("无歌词", "主唱退出", "低人声")):
        return 0.76
    return 0.6


def normalize_type(value: str) -> str:
    return value.replace("-", "_")


def has_complete_vocal(section: pipeline.SectionCandidate) -> bool:
    evidence = f"{section.lyric_evidence} {section.vocal_evidence} {section.acoustic_evidence}"
    if any(marker in evidence for marker in ("无歌词", "主唱退出", "低人声", "弱化", "无完整主唱")):
        return False
    return "完整主唱" in evidence or "主唱" in evidence


def has_marker(section: pipeline.SectionCandidate, markers: tuple[str, ...]) -> bool:
    evidence = f"{section.lyric_evidence} {section.vocal_evidence} {section.acoustic_evidence}"
    return any(marker in evidence for marker in markers)


def structure_label(section_type: str) -> str:
    return {
        "intro": "A",
        "verse": "B",
        "pre_chorus": "C",
        "chorus": "D",
        "bridge": "E",
        "outro": "A'",
    }.get(section_type, "X")


def write_rows(path: Path, rows: list[AcousticFirstRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(AcousticFirstRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


if __name__ == "__main__":
    main()
