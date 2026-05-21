from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import functional_section_pipeline as pipeline


@dataclass
class SentenceFeature:
    line_index: int
    start: float
    end: float
    text: str
    role: str
    rms: float
    onset: float
    centroid: float
    novelty: float
    chroma_peak: int
    chroma_strength: float


@dataclass
class GapDecision:
    start: float
    end: float
    duration: float
    decision: str
    assigned_to: str
    reason: str
    previous_similarity: float
    next_similarity: float


@dataclass
class AcousticValidation:
    boundary_time: float
    left_type: str
    right_type: str
    semantic_score: float
    acoustic_score: float
    novelty_score: float
    timbre_score: float
    chroma_score: float
    onset_score: float
    rms_score: float
    vocal_activity_score: float
    override_by_acoustic: bool
    decision: str
    reason: str


def build_sentence_features(
    lines: list[pipeline.LyricLine],
    roles: list[str],
    features: dict[str, object],
) -> list[SentenceFeature]:
    result: list[SentenceFeature] = []
    for index, line in enumerate(lines):
        result.append(
            SentenceFeature(
                line_index=index,
                start=line.start,
                end=line.end,
                text=line.text,
                role=roles[index] if index < len(roles) else "verse",
                rms=pipeline.average_curve(features, "rms", line.start, line.end),
                onset=pipeline.average_curve(features, "onset", line.start, line.end),
                centroid=pipeline.average_curve(features, "centroid", line.start, line.end),
                novelty=pipeline.average_curve(features, "novelty", line.start, line.end),
                chroma_peak=chroma_peak(features, line.start, line.end),
                chroma_strength=chroma_strength(features, line.start, line.end),
            )
        )
    return result


def chroma_peak(features: dict[str, object], start: float, end: float) -> int:
    import numpy as np

    times = features["times"]
    chroma = features["chroma"]
    mask = (times >= start) & (times < end)
    if not np.any(mask):
        return -1
    means = np.mean(chroma[:, mask], axis=1)
    return int(np.argmax(means))


def chroma_strength(features: dict[str, object], start: float, end: float) -> float:
    import numpy as np

    times = features["times"]
    chroma = features["chroma"]
    mask = (times >= start) & (times < end)
    if not np.any(mask):
        return 0.0
    means = np.mean(chroma[:, mask], axis=1)
    total = float(np.sum(means))
    if total <= 0:
        return 0.0
    return float(np.max(means) / total)


def draft_sections_from_lyrics(
    lines: list[pipeline.LyricLine],
    roles: list[str],
    context: pipeline.StructureContext,
) -> list[pipeline.SectionCandidate]:
    if not lines:
        return []
    sections: list[pipeline.SectionCandidate] = []
    start_index = 0
    current_role = roles[0]
    for index in range(1, len(lines)):
        gap = lines[index].start - lines[index - 1].end
        if roles[index] != current_role or gap >= context.long_gap_seconds:
            sections.append(section_from_lines(lines, start_index, index, current_role))
            start_index = index
            current_role = roles[index]
    sections.append(section_from_lines(lines, start_index, len(lines), current_role))
    return sections


def section_from_lines(
    lines: list[pipeline.LyricLine],
    start_index: int,
    end_index: int,
    role: str,
) -> pipeline.SectionCandidate:
    items = lines[start_index:end_index]
    return pipeline.SectionCandidate(
        start=items[0].start,
        end=items[-1].end,
        section_type=role,
        text=" ".join(item.text for item in items),
        lyric_evidence="歌词语义/重复关系形成的初稿段落",
        vocal_evidence="完整主唱句组",
        acoustic_evidence="尚未声学复核",
        boundary_confidence=0.0,
        type_confidence=0.0,
        need_human_review=False,
    )


def resolve_instrumental_gaps(
    sections: list[pipeline.SectionCandidate],
    features: dict[str, object],
    duration: float,
    context: pipeline.StructureContext,
) -> tuple[list[pipeline.SectionCandidate], list[GapDecision]]:
    if not sections:
        return [], []
    result: list[pipeline.SectionCandidate] = []
    decisions: list[GapDecision] = []

    intro_gap = sections[0].start
    if intro_gap > 1.0:
        intro = pipeline.SectionCandidate(
            0.0,
            sections[0].start,
            "intro",
            "",
            "开头第一句歌词前的无歌词音乐",
            "主唱未正式进入",
            "intro gap from lyric timeline",
            0.0,
            0.0,
            False,
        )
        result.append(intro)
        decisions.append(GapDecision(0.0, sections[0].start, intro_gap, "new_section", "intro", "开头无歌词音乐", 1.0, 0.0))

    result.append(sections[0])
    for previous, following in zip(sections[:-1], sections[1:]):
        gap = following.start - previous.end
        if gap > 0.05:
            decision = resolve_one_gap(previous, following, gap, features, duration, context)
            decisions.append(decision)
            if decision.decision == "new_section":
                result.append(
                    pipeline.SectionCandidate(
                        previous.end,
                        following.start,
                        decision.assigned_to,
                        "",
                        decision.reason,
                        "歌词间无完整主唱",
                        "instrumental gap resolved before acoustic validation",
                        0.0,
                        0.0,
                        decision.assigned_to == "bridge",
                    )
                )
            elif decision.assigned_to == "previous":
                previous.end = following.start
                previous.lyric_evidence = append_evidence(previous.lyric_evidence, "短无歌词间隙并入前一段")
            elif decision.assigned_to == "next":
                following.start = previous.end
                following.lyric_evidence = append_evidence(following.lyric_evidence, "短无歌词间隙并入后一段")
        result.append(following)

    outro_gap = duration - result[-1].end
    if outro_gap > 1.0:
        result.append(
            pipeline.SectionCandidate(
                result[-1].end,
                duration,
                "outro",
                "",
                "最后一句歌词后的无歌词收束",
                "主唱退出或弱化",
                "outro gap from lyric timeline",
                0.0,
                0.0,
                False,
            )
        )
        decisions.append(GapDecision(duration - outro_gap, duration, outro_gap, "new_section", "outro", "结尾无歌词音乐", 0.0, 1.0))
    return result, decisions


def resolve_one_gap(
    previous: pipeline.SectionCandidate,
    following: pipeline.SectionCandidate,
    gap: float,
    features: dict[str, object],
    duration: float,
    context: pipeline.StructureContext,
) -> GapDecision:
    start = previous.end
    end = following.start
    if gap >= context.long_gap_seconds:
        return GapDecision(start, end, gap, "new_section", "bridge", "中间长无歌词音乐，独立为 bridge", 0.0, 0.0)
    previous_similarity = acoustic_similarity(features, start, end, previous.start, previous.end)
    next_similarity = acoustic_similarity(features, start, end, following.start, following.end)
    if next_similarity > previous_similarity + 0.04:
        assigned_to = "next"
        reason = "短无歌词间隙与后一段声学更相似"
    else:
        assigned_to = "previous"
        reason = "短无歌词间隙与前一段声学更相似或差异不显著"
    return GapDecision(start, end, gap, "absorb", assigned_to, reason, previous_similarity, next_similarity)


def acoustic_similarity(
    features: dict[str, object],
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    left = acoustic_vector(features, left_start, left_end)
    right = acoustic_vector(features, right_start, right_end)
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
    return round(1.0 / (1.0 + distance), 3)


def acoustic_vector(features: dict[str, object], start: float, end: float) -> tuple[float, float, float, float]:
    return (
        pipeline.average_curve(features, "rms", start, end),
        pipeline.average_curve(features, "onset", start, end) / 10.0,
        pipeline.average_curve(features, "centroid", start, end) / 5000.0,
        pipeline.average_curve(features, "novelty", start, end),
    )


def validate_with_acoustics(
    sections: list[pipeline.SectionCandidate],
    features: dict[str, object],
) -> list[AcousticValidation]:
    validations: list[AcousticValidation] = []
    if len(sections) < 2:
        return validations
    duration = float(features["duration"])
    for left, right in zip(sections[:-1], sections[1:]):
        boundary = right.start
        semantic_score = semantic_boundary_score(left, right)
        components = acoustic_boundary_components(left, right, features, boundary)
        acoustic_score = acoustic_boundary_score(components)
        override_by_acoustic = acoustic_score >= semantic_score + 0.18 and acoustic_score >= 0.75
        if override_by_acoustic:
            decision = "acoustic_override"
            reason = "多维声学变化强于语义边界，允许声学推翻或移动边界"
            left.need_human_review = True
            right.need_human_review = True
        elif acoustic_score >= semantic_score - 0.08:
            decision = "confirm"
            reason = "声学证据支持或基本支持语义边界"
        else:
            decision = "lyric_kept"
            reason = "声学证据不足以推翻语义切分"
        validations.append(
            AcousticValidation(
                boundary_time=boundary,
                left_type=left.section_type,
                right_type=right.section_type,
                semantic_score=semantic_score,
                acoustic_score=acoustic_score,
                novelty_score=components["novelty"],
                timbre_score=components["timbre"],
                chroma_score=components["chroma"],
                onset_score=components["onset"],
                rms_score=components["rms"],
                vocal_activity_score=components["vocal_activity"],
                override_by_acoustic=override_by_acoustic,
                decision=decision,
                reason=reason,
            )
        )
    return validations


def semantic_boundary_score(left: pipeline.SectionCandidate, right: pipeline.SectionCandidate) -> float:
    score = 0.45
    score += 0.20 * float(left.section_type != right.section_type)
    score += 0.15 * float(left.section_type in {"intro", "bridge", "outro"} or right.section_type in {"intro", "bridge", "outro"})
    score += 0.10 * max(left.type_confidence, right.type_confidence)
    score += 0.10 * float("hook" in f"{left.lyric_evidence} {right.lyric_evidence}".lower())
    return round(min(0.95, score), 2)


def lyric_boundary_confidence(left: pipeline.SectionCandidate, right: pipeline.SectionCandidate) -> float:
    if left.section_type in {"intro", "bridge", "outro"} or right.section_type in {"intro", "bridge", "outro"}:
        return 0.78
    if left.section_type != right.section_type:
        return 0.72
    return 0.55


def acoustic_boundary_components(
    left: pipeline.SectionCandidate,
    right: pipeline.SectionCandidate,
    features: dict[str, object],
    boundary: float,
) -> dict[str, float]:
    duration = float(features["duration"])
    _peak_time, novelty = pipeline.strongest_novelty(features, max(0, boundary - 1.5), min(duration, boundary + 1.5))
    left_start = max(left.start, boundary - min(8.0, max(2.0, left.duration * 0.5)))
    right_end = min(right.end, boundary + min(8.0, max(2.0, right.duration * 0.5)))
    left_vec = acoustic_vector(features, left_start, boundary)
    right_vec = acoustic_vector(features, boundary, right_end)
    rms_delta = abs(left_vec[0] - right_vec[0])
    onset_delta = abs(left_vec[1] - right_vec[1])
    timbre_delta = abs(left_vec[2] - right_vec[2])
    chroma_delta = chroma_distance(features, left_start, boundary, boundary, right_end)
    vocal_activity = 1.0 if (not left.text or not right.text) else 0.0
    return {
        "novelty": round(clamp01(novelty), 3),
        "timbre": round(clamp01(timbre_delta * 3.0), 3),
        "chroma": round(clamp01(chroma_delta), 3),
        "onset": round(clamp01(onset_delta * 2.5), 3),
        "rms": round(clamp01(rms_delta * 8.0), 3),
        "vocal_activity": vocal_activity,
    }


def acoustic_boundary_score(components: dict[str, float]) -> float:
    score = (
        0.25 * components["novelty"]
        + 0.20 * components["timbre"]
        + 0.20 * components["chroma"]
        + 0.15 * components["onset"]
        + 0.10 * components["rms"]
        + 0.10 * components["vocal_activity"]
    )
    return round(min(0.95, score), 2)


def chroma_distance(
    features: dict[str, object],
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    import numpy as np

    times = features["times"]
    chroma = features["chroma"]
    left_mask = (times >= left_start) & (times < left_end)
    right_mask = (times >= right_start) & (times < right_end)
    if not np.any(left_mask) or not np.any(right_mask):
        return 0.0
    left = np.mean(chroma[:, left_mask], axis=1)
    right = np.mean(chroma[:, right_mask], axis=1)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0:
        return 0.0
    cosine = float(np.dot(left, right) / denom)
    return clamp01((1.0 - cosine) * 1.5)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def finalize_sections(
    sections: list[pipeline.SectionCandidate],
    lines: list[pipeline.LyricLine],
    features: dict[str, object],
    title: str,
) -> list[pipeline.SectionCandidate]:
    context = pipeline.structure_context(lines)
    pipeline.absorb_short_fragments_before_bridge(sections)
    pipeline.absorb_terminal_fragments(sections, float(features["duration"]))
    sections = pipeline.merge_adjacent_same_type(sections, max_duration=context.max_chorus_seed_seconds * 1.2)
    sections = pipeline.split_overlong_repeated_sections(sections, lines, context)
    pipeline.snap_section_boundaries(sections, features)
    pipeline.absorb_terminal_fragments(sections, float(features["duration"]))
    pipeline.refresh_section_evidence(sections, features, title)
    return sections


def append_evidence(existing: str, addition: str) -> str:
    if not existing:
        return addition
    if addition in existing:
        return existing
    return f"{existing}；{addition}"


def write_sentence_segments(path: Path, rows: list[SentenceFeature]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["line_index", "start_time", "end_time", "text", "draft_role", "rms", "onset", "centroid", "novelty", "chroma_peak", "chroma_strength"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "line_index": row.line_index,
                    "start_time": pipeline.base.format_timestamp(row.start),
                    "end_time": pipeline.base.format_timestamp(row.end),
                    "text": row.text,
                    "draft_role": pipeline.public_section_type(row.role),
                    "rms": round(row.rms, 6),
                    "onset": round(row.onset, 6),
                    "centroid": round(row.centroid, 3),
                    "novelty": round(row.novelty, 6),
                    "chroma_peak": row.chroma_peak,
                    "chroma_strength": round(row.chroma_strength, 6),
                }
            )


def write_sentence_segments_redacted(path: Path, rows: list[SentenceFeature]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["line_index", "start_time", "end_time", "text_ref", "char_count", "draft_role", "rms", "onset", "centroid", "novelty", "chroma_peak", "chroma_strength"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "line_index": row.line_index,
                    "start_time": pipeline.base.format_timestamp(row.start),
                    "end_time": pipeline.base.format_timestamp(row.end),
                    "text_ref": f"line_{row.line_index:03d}",
                    "char_count": len(row.text),
                    "draft_role": pipeline.public_section_type(row.role),
                    "rms": round(row.rms, 6),
                    "onset": round(row.onset, 6),
                    "centroid": round(row.centroid, 3),
                    "novelty": round(row.novelty, 6),
                    "chroma_peak": row.chroma_peak,
                    "chroma_strength": round(row.chroma_strength, 6),
                }
            )


def write_draft_sections(path: Path, song_id: str, sections: list[pipeline.SectionCandidate]) -> None:
    pipeline.write_sections(path, song_id, sections)


def write_gap_resolution(path: Path, decisions: list[GapDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["start_time", "end_time", "duration", "decision", "assigned_to", "reason", "previous_similarity", "next_similarity"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in decisions:
            writer.writerow(
                {
                    "start_time": pipeline.base.format_timestamp(item.start),
                    "end_time": pipeline.base.format_timestamp(item.end),
                    "duration": pipeline.base.format_timestamp(item.duration),
                    "decision": item.decision,
                    "assigned_to": item.assigned_to,
                    "reason": item.reason,
                    "previous_similarity": item.previous_similarity,
                    "next_similarity": item.next_similarity,
                }
            )


def write_acoustic_validation(path: Path, validations: list[AcousticValidation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "boundary_time",
        "left_type",
        "right_type",
        "semantic_score",
        "acoustic_score",
        "novelty_score",
        "timbre_score",
        "chroma_score",
        "onset_score",
        "rms_score",
        "vocal_activity_score",
        "override_by_acoustic",
        "decision",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in validations:
            writer.writerow(
                {
                    "boundary_time": pipeline.base.format_timestamp(item.boundary_time),
                    "left_type": pipeline.public_section_type(item.left_type),
                    "right_type": pipeline.public_section_type(item.right_type),
                    "semantic_score": item.semantic_score,
                    "acoustic_score": item.acoustic_score,
                    "novelty_score": item.novelty_score,
                    "timbre_score": item.timbre_score,
                    "chroma_score": item.chroma_score,
                    "onset_score": item.onset_score,
                    "rms_score": item.rms_score,
                    "vocal_activity_score": item.vocal_activity_score,
                    "override_by_acoustic": str(item.override_by_acoustic).lower(),
                    "decision": item.decision,
                    "reason": item.reason,
                }
            )
