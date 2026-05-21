#!/usr/bin/env python3
"""Functional pop-song section segmentation from vocals, lyrics, and acoustics."""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import segment_hybrid as base
import vocal_activity


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class StructureContext:
    paragraph_gap_seconds: float
    long_gap_seconds: float
    max_pre_chorus_seconds: float
    max_chorus_seed_seconds: float
    dominant_role_ratio: float
    contrast_similarity_threshold: float


@dataclass
class LyricLine:
    start: float
    end: float
    text: str


@dataclass
class LyricParagraph:
    start_index: int
    end_index: int
    start: float
    end: float
    text: str


@dataclass
class SectionCandidate:
    start: float
    end: float
    section_type: str
    text: str
    lyric_evidence: str
    vocal_evidence: str
    acoustic_evidence: str
    boundary_confidence: float
    type_confidence: float
    need_human_review: bool


def clean_text(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    return text.strip()


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def is_metadata_line(text: str) -> bool:
    compact = base.normalize_text(text)
    metadata_terms = (
        "词曲",
        "詞曲",
        "作词",
        "作詞",
        "作曲",
        "编曲",
        "編曲",
        "混音",
        "母带",
        "母帶",
        "制作人",
        "製作人",
        "composer",
        "lyrics",
    )
    if any(term in compact for term in metadata_terms):
        return True
    return len(compact) <= 6 and compact.startswith(("曲", "词", "詞"))


def text_similarity(left: str, right: str) -> float:
    left_norm = base.normalize_text(left)
    right_norm = base.normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm in right_norm or right_norm in left_norm:
        return 1.0
    return float(SequenceMatcher(None, left_norm, right_norm).ratio())


def read_transcript(path: Path) -> list[LyricLine]:
    lines: list[LyricLine] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            text = clean_text(row.get("text", ""))
            if not text or not has_cjk(text) or is_metadata_line(text):
                continue
            start = base.parse_timestamp(row["start_time"])
            end = base.parse_timestamp(row["end_time"])
            if end > start:
                lines.append(LyricLine(start, end, text))
    return lines


def run_transcription(audio: Path, transcript_csv: Path, model: str, language: str) -> None:
    command = [
        ".venv/bin/python",
        "hybrid_structure/transcribe_whisper.py",
        "--audio",
        str(audio),
        "--model",
        model,
        "--language",
        language,
        "--output",
        str(transcript_csv),
        "--no-vad",
    ]
    subprocess.run(command, check=True)


def scale(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def load_acoustic_features(audio: Path) -> dict[str, object]:
    import librosa
    import numpy as np

    y, sr = librosa.load(audio, sr=22050, mono=True)
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length).reshape(-1)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).reshape(-1)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length).reshape(-1)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)

    def column_delta(matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[1] < 2:
            return np.zeros(len(rms))
        delta = np.linalg.norm(np.diff(matrix, axis=1), axis=0)
        return np.concatenate([[0.0], delta])[: len(rms)]

    novelty_parts = [
        np.array(scale(np.abs(np.diff(rms, prepend=rms[0])).tolist())),
        np.array(scale(onset.tolist())),
        np.array(scale(np.abs(np.diff(centroid, prepend=centroid[0])).tolist())),
        np.array(scale(column_delta(chroma).tolist())),
        np.array(scale(column_delta(mfcc).tolist())),
    ]
    novelty = np.mean(novelty_parts, axis=0)
    return {
        "duration": float(librosa.get_duration(y=y, sr=sr)),
        "times": times,
        "rms": rms,
        "onset": onset,
        "centroid": centroid,
        "chroma": chroma,
        "mfcc": mfcc,
        "novelty": novelty,
    }


def average_curve(features: dict[str, object], name: str, start: float, end: float) -> float:
    import numpy as np

    times = features["times"]
    curve = features[name]
    mask = (times >= start) & (times < end)
    if not np.any(mask):
        return 0.0
    return float(np.mean(curve[mask]))


def strongest_novelty(features: dict[str, object], start: float, end: float) -> tuple[float, float]:
    import numpy as np

    times = features["times"]
    novelty = features["novelty"]
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        return start, 0.0
    local_values = novelty[mask]
    local_times = times[mask]
    idx = int(np.argmax(local_values))
    return float(local_times[idx]), float(local_values[idx])


def snap_boundary(features: dict[str, object], boundary: float, window: float = 1.5) -> tuple[float, float]:
    duration = float(features["duration"])
    snapped, strength = strongest_novelty(features, max(0.0, boundary - window), min(duration, boundary + window))
    if strength >= 0.58 and abs(snapped - boundary) <= window:
        return round(snapped, 2), strength
    return round(boundary, 2), strength


def repeated_line_indices(lines: list[LyricLine], threshold: float = 0.62) -> set[int]:
    repeated: set[int] = set()
    for i, left in enumerate(lines):
        for j in range(i + 1, len(lines)):
            right = lines[j]
            if right.start - left.start < 18.0:
                continue
            if text_similarity(left.text, right.text) >= threshold:
                repeated.add(i)
                repeated.add(j)
    return repeated


def title_hook_indices(lines: list[LyricLine], title: str) -> set[int]:
    if not title:
        return set()
    title_norm = base.normalize_text(title)
    hits: set[int] = set()
    for index, line in enumerate(lines):
        norm = base.normalize_text(line.text)
        if title_norm and (title_norm in norm or text_similarity(line.text, title) >= 0.55):
            hits.add(index)
    return hits


def structure_context(lines: list[LyricLine]) -> StructureContext:
    import numpy as np

    if len(lines) < 2:
        return StructureContext(0.75, 6.0, 24.0, 42.0, 0.72, 0.46)

    gaps = [max(0.0, lines[index].start - lines[index - 1].end) for index in range(1, len(lines))]
    positive_gaps = [gap for gap in gaps if gap > 0.05]
    durations = [line.end - line.start for line in lines]
    song_span = max(lines[-1].end - lines[0].start, 1.0)
    median_line = float(np.median(durations)) if durations else 4.0

    if positive_gaps:
        paragraph_gap = clamp(float(np.percentile(positive_gaps, 65)), 0.55, 1.8)
        long_gap = clamp(float(np.percentile(positive_gaps, 92)), 4.5, 10.0)
    else:
        paragraph_gap = 0.75
        long_gap = 6.0

    return StructureContext(
        paragraph_gap_seconds=paragraph_gap,
        long_gap_seconds=max(long_gap, median_line * 1.25),
        max_pre_chorus_seconds=clamp(song_span * 0.11, median_line * 3.0, 32.0),
        max_chorus_seed_seconds=clamp(song_span * 0.16, median_line * 5.0, 52.0),
        dominant_role_ratio=0.72,
        contrast_similarity_threshold=0.46,
    )


def initial_line_roles(lines: list[LyricLine], title: str) -> list[str]:
    if not lines:
        return []
    context = structure_context(lines)
    paragraphs = group_lyric_paragraphs(lines, context)
    roles = ["verse" for _ in lines]
    repeated = repeated_line_indices(lines)
    title_hits = title_hook_indices(lines, title)

    chorus_seeds = set(title_hits)
    if not chorus_seeds:
        chorus_seeds = set(repeated)

    for seed in sorted(chorus_seeds):
        end = seed
        while end + 1 < len(lines):
            candidate = lines[end + 1]
            gap = lines[end + 1].start - lines[end].end
            if gap >= context.long_gap_seconds or lines[end + 1].end - lines[seed].start > context.max_chorus_seed_seconds:
                break
            if end - seed < 3:
                end += 1
                continue
            if (
                end + 1 not in title_hits
                and resembles_earlier_non_hook(candidate, lines[:seed], title)
            ):
                break
            if end + 1 in repeated or end + 1 in title_hits:
                end += 1
                continue
            break
        for index in range(seed, end + 1):
            roles[index] = "chorus"

    chorus_starts = [
        index
        for index, role in enumerate(roles)
        if role == "chorus" and (index == 0 or roles[index - 1] != "chorus")
    ]
    for chorus_start in chorus_starts:
        pre_start = adaptive_pre_chorus_start(lines, roles, paragraphs, chorus_start, context)
        for index in range(pre_start, chorus_start):
            if roles[index] != "chorus":
                roles[index] = "pre-chorus"

    if not title_hits:
        split_long_no_title_chorus_runs(lines, roles, context)
    stabilize_roles_by_lyric_paragraphs(lines, roles, paragraphs, title_hits, context)
    mark_internal_chorus_contrasts(lines, roles, repeated, title_hits, context)
    mark_bridge_candidates(lines, roles, context)
    protect_lyric_paragraph_continuity(lines, roles, paragraphs, context)
    return roles


def protect_lyric_paragraph_continuity(
    lines: list[LyricLine],
    roles: list[str],
    paragraphs: list[LyricParagraph],
    context: StructureContext,
) -> None:
    """Avoid splitting a continuous sung paragraph on weak bridge evidence."""
    for paragraph in paragraphs:
        index = paragraph.start_index
        while index < paragraph.end_index:
            if roles[index] != "bridge":
                index += 1
                continue
            run_start = index
            while index + 1 < paragraph.end_index and roles[index + 1] == "bridge":
                index += 1
            run_end = index + 1
            if should_absorb_intra_paragraph_bridge(lines, roles, paragraph, run_start, run_end, context):
                for role_index in range(run_start, run_end):
                    roles[role_index] = "chorus"
            index = run_end


def should_absorb_intra_paragraph_bridge(
    lines: list[LyricLine],
    roles: list[str],
    paragraph: LyricParagraph,
    run_start: int,
    run_end: int,
    context: StructureContext,
) -> bool:
    if run_start <= paragraph.start_index or run_end >= paragraph.end_index:
        return False
    previous_role = roles[run_start - 1]
    next_role = roles[run_end]
    if previous_role != "chorus" or next_role != "chorus":
        return False
    before_gap = lines[run_start].start - lines[run_start - 1].end
    after_gap = lines[run_end].start - lines[run_end - 1].end
    if max(before_gap, after_gap) >= context.long_gap_seconds:
        return False
    run_duration = lines[run_end - 1].end - lines[run_start].start
    sustained_contrast_seconds = context.max_pre_chorus_seconds * 0.65
    return run_duration < sustained_contrast_seconds


def mark_internal_chorus_contrasts(
    lines: list[LyricLine],
    roles: list[str],
    repeated: set[int],
    title_hits: set[int],
    context: StructureContext,
) -> None:
    """Split long chorus runs when a non-repeated contrast passage interrupts them."""
    if not lines:
        return
    song_start = lines[0].start
    song_span = max(lines[-1].end - song_start, 1.0)
    index = 0
    while index < len(roles):
        if roles[index] != "chorus":
            index += 1
            continue
        run_start = index
        while index + 1 < len(roles) and roles[index + 1] == "chorus":
            index += 1
        run_end = index + 1
        run_duration = lines[run_end - 1].end - lines[run_start].start
        if run_duration >= context.max_chorus_seed_seconds * 1.5:
            mark_non_repeated_chunks_in_chorus_run(lines, roles, repeated, title_hits, context, run_start, run_end, song_start, song_span)
        index = run_end


def mark_non_repeated_chunks_in_chorus_run(
    lines: list[LyricLine],
    roles: list[str],
    repeated: set[int],
    title_hits: set[int],
    context: StructureContext,
    run_start: int,
    run_end: int,
    song_start: float,
    song_span: float,
) -> None:
    index = run_start
    while index < run_end:
        if index in repeated or index in title_hits:
            index += 1
            continue
        chunk_start = index
        while index < run_end and index not in repeated and index not in title_hits:
            index += 1
        chunk_end = index
        chunk_duration = lines[chunk_end - 1].end - lines[chunk_start].start
        position = (lines[chunk_start].start - song_start) / song_span
        has_chorus_before = any(item in repeated or item in title_hits for item in range(run_start, chunk_start))
        has_chorus_after = any(item in repeated or item in title_hits for item in range(chunk_end, run_end))
        minimum = clamp(song_span * 0.035, context.max_pre_chorus_seconds * 0.35, context.max_pre_chorus_seconds)
        maximum = max(context.max_chorus_seed_seconds, context.max_pre_chorus_seconds)
        if position >= 0.45 and has_chorus_before and has_chorus_after and minimum <= chunk_duration <= maximum:
            for role_index in range(chunk_start, chunk_end):
                roles[role_index] = "bridge"


def group_lyric_paragraphs(lines: list[LyricLine], context: StructureContext | None = None) -> list[LyricParagraph]:
    """Group lyric lines into semantic stanzas before assigning section roles."""
    if not lines:
        return []
    context = context or structure_context(lines)
    boundaries = [0]
    for index in range(1, len(lines)):
        gap = lines[index].start - lines[index - 1].end
        current_start = boundaries[-1]
        current_duration = lines[index - 1].end - lines[current_start].start
        current_lines = index - current_start
        if gap >= context.long_gap_seconds:
            boundaries.append(index)
        elif gap >= context.paragraph_gap_seconds and current_duration >= max(7.0, context.max_pre_chorus_seconds * 0.35) and current_lines >= 2:
            boundaries.append(index)
        elif current_duration >= context.max_chorus_seed_seconds * 0.8 and gap >= context.paragraph_gap_seconds * 0.35:
            boundaries.append(index)
    boundaries.append(len(lines))

    paragraphs: list[LyricParagraph] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        items = lines[start:end]
        paragraphs.append(
            LyricParagraph(
                start_index=start,
                end_index=end,
                start=items[0].start,
                end=items[-1].end,
                text=" ".join(item.text for item in items),
            )
        )
    return paragraphs


def stabilize_roles_by_lyric_paragraphs(
    lines: list[LyricLine],
    roles: list[str],
    paragraphs: list[LyricParagraph],
    title_hits: set[int],
    context: StructureContext,
) -> None:
    """Prefer paragraph-level consistency unless a paragraph contains a clear hook split."""
    for paragraph in paragraphs:
        role_slice = roles[paragraph.start_index : paragraph.end_index]
        if len(set(role_slice)) <= 1:
            continue
        if "bridge" in role_slice:
            dominant = "bridge"
        elif paragraph_contains_title_hook(paragraph, title_hits):
            stabilize_hook_paragraph(lines, roles, paragraph, title_hits, context)
            continue
        else:
            if not paragraph_has_clear_dominant_role(role_slice, context.dominant_role_ratio):
                continue
            dominant = paragraph_majority_role(role_slice)
        for index in range(paragraph.start_index, paragraph.end_index):
            roles[index] = dominant


def paragraph_contains_title_hook(paragraph: LyricParagraph, title_hits: set[int]) -> bool:
    return any(paragraph.start_index <= index < paragraph.end_index for index in title_hits)


def stabilize_hook_paragraph(
    lines: list[LyricLine],
    roles: list[str],
    paragraph: LyricParagraph,
    title_hits: set[int],
    context: StructureContext,
) -> None:
    first_hook = min(index for index in title_hits if paragraph.start_index <= index < paragraph.end_index)
    pre_start = adaptive_pre_chorus_start(
        lines,
        roles,
        [paragraph],
        first_hook,
        context,
        paragraph_start=paragraph.start_index,
    )
    for index in range(paragraph.start_index, paragraph.end_index):
        if pre_start <= index < first_hook and roles[index] != "chorus":
            roles[index] = "pre-chorus"


def adaptive_pre_chorus_start(
    lines: list[LyricLine],
    roles: list[str],
    paragraphs: list[LyricParagraph],
    chorus_start: int,
    context: StructureContext,
    paragraph_start: int | None = None,
) -> int:
    """Find the contiguous lyric buildup before a chorus without fixed line counts."""
    if paragraph_start is None:
        paragraph = paragraph_for_line(paragraphs, chorus_start)
        paragraph_start = paragraph.start_index if paragraph else 0
    start = paragraph_start

    for index in range(chorus_start - 1, start - 1, -1):
        if roles[index] in {"chorus", "bridge"}:
            start = index + 1
            break
        if lines and index + 1 < len(lines):
            gap = lines[index + 1].start - lines[index].end
            if gap >= context.long_gap_seconds:
                start = index + 1
                break

    if lines and chorus_start > start:
        split = strongest_internal_gap(lines, start, chorus_start)
        if split is not None:
            start = max(start, split)
        density_split = strongest_line_density_shift(lines, start, chorus_start)
        if density_split is not None:
            start = max(start, density_split)
        while start < chorus_start and lines[chorus_start].start - lines[start].start > context.max_pre_chorus_seconds:
            start += 1
    return start


def strongest_line_density_shift(lines: list[LyricLine], start: int, end: int) -> int | None:
    """Find where lyric phrasing becomes denser before a hook."""
    import numpy as np

    if end - start < 4:
        return None
    durations = [lines[index].end - lines[index].start for index in range(start, end)]
    best_index: int | None = None
    best_score = 0.0
    for offset in range(1, len(durations)):
        left = durations[:offset]
        right = durations[offset:]
        if len(left) < 2 or len(right) < 2:
            continue
        left_median = float(np.median(left))
        right_median = float(np.median(right))
        if right_median <= 0:
            continue
        score = left_median / right_median
        if score > best_score:
            best_score = score
            best_index = start + offset
    if best_index is not None and best_score >= 1.25:
        return best_index
    return None


def paragraph_for_line(paragraphs: list[LyricParagraph], line_index: int) -> LyricParagraph | None:
    for paragraph in paragraphs:
        if paragraph.start_index <= line_index < paragraph.end_index:
            return paragraph
    return None


def paragraph_has_clear_dominant_role(role_slice: list[str], ratio: float = 0.72) -> bool:
    counts: dict[str, int] = {}
    for role in role_slice:
        counts[role] = counts.get(role, 0) + 1
    return max(counts.values()) / max(len(role_slice), 1) >= ratio


def paragraph_majority_role(role_slice: list[str]) -> str:
    priority = {"chorus": 3, "pre-chorus": 2, "bridge": 1, "verse": 0}
    counts: dict[str, int] = {}
    for role in role_slice:
        counts[role] = counts.get(role, 0) + 1
    return max(counts, key=lambda role: (counts[role], priority.get(role, 0)))


def split_long_no_title_chorus_runs(lines: list[LyricLine], roles: list[str], context: StructureContext) -> None:
    """For songs without title hooks, split long repeated blocks into build + hook."""
    chorus_indices = [index for index, role in enumerate(roles) if role == "chorus"]
    if not chorus_indices:
        return
    first_chorus = min(chorus_indices)
    for index in range(first_chorus):
        roles[index] = "verse"

    index = 0
    while index < len(roles):
        if roles[index] != "chorus":
            index += 1
            continue
        start = index
        while (
            index + 1 < len(roles)
            and roles[index + 1] == "chorus"
            and lines[index + 1].start - lines[index].end < context.long_gap_seconds
        ):
            index += 1
        end = index + 1
        run_duration = lines[end - 1].end - lines[start].start
        if run_duration >= context.max_chorus_seed_seconds:
            split = strongest_internal_gap(lines, start, end)
            if split is not None and has_enough_material_on_both_sides(lines, start, split, end, context):
                for role_index in range(start, split):
                    roles[role_index] = "pre-chorus"
        index = end


def has_enough_material_on_both_sides(
    lines: list[LyricLine],
    start: int,
    split: int,
    end: int,
    context: StructureContext,
) -> bool:
    left_duration = lines[split - 1].end - lines[start].start
    right_duration = lines[end - 1].end - lines[split].start
    minimum = max(6.0, context.max_pre_chorus_seconds * 0.25)
    return left_duration >= minimum and right_duration >= minimum


def strongest_internal_gap(lines: list[LyricLine], start: int, end: int) -> int | None:
    best_index: int | None = None
    best_gap = 0.0
    for index in range(start + 1, end):
        gap = lines[index].start - lines[index - 1].end
        if gap > best_gap:
            best_gap = gap
            best_index = index
    if best_gap >= 0.75:
        return best_index
    return None


def mark_bridge_candidates(lines: list[LyricLine], roles: list[str], context: StructureContext) -> None:
    chorus_indices = [index for index, role in enumerate(roles) if role == "chorus"]
    if len(chorus_indices) < 2:
        return
    duration = lines[-1].end
    min_bridge_candidate = clamp(duration * 0.035, context.max_pre_chorus_seconds * 0.35, context.max_pre_chorus_seconds)

    index = 0
    while index < len(roles):
        if roles[index] != "verse":
            index += 1
            continue
        start = index
        while index + 1 < len(roles) and roles[index + 1] == "verse":
            index += 1
        end = index + 1
        run_duration = lines[end - 1].end - lines[start].start
        if run_duration < min_bridge_candidate:
            index = end
            continue
        after_chorus = any(chorus_index < start for chorus_index in chorus_indices)
        before_later_chorus = any(chorus_index >= end for chorus_index in chorus_indices)
        late_enough = lines[start].start / max(duration, 1.0) >= 0.45
        repeated_narrative = any(resembles_earlier_non_hook(lines[item], lines[:item], "") for item in range(start, end))
        if after_chorus and before_later_chorus and late_enough and not repeated_narrative:
            for role_index in range(start, end):
                roles[role_index] = "bridge"
        index = end


def resembles_earlier_non_hook(line: LyricLine, earlier_lines: list[LyricLine], title: str) -> bool:
    for earlier in earlier_lines:
        if title and text_similarity(earlier.text, title) >= 0.55:
            continue
        if text_similarity(line.text, earlier.text) >= 0.65:
            return True
    return False


def make_role_blocks(lines: list[LyricLine], roles: list[str], min_instrumental_gap: float) -> list[SectionCandidate]:
    if not lines:
        return []
    blocks: list[SectionCandidate] = []

    def add_text_block(start_index: int, end_index: int, role: str) -> None:
        items = lines[start_index:end_index]
        text = " ".join(item.text for item in items)
        blocks.append(
            SectionCandidate(
                start=items[0].start,
                end=items[-1].end,
                section_type=role,
                text=text,
                lyric_evidence="",
                vocal_evidence="",
                acoustic_evidence="",
                boundary_confidence=0.0,
                type_confidence=0.0,
                need_human_review=False,
            )
        )

    current_start = 0
    current_role = roles[0]
    for index in range(1, len(lines)):
        gap = lines[index].start - lines[index - 1].end
        if gap >= min_instrumental_gap or roles[index] != current_role:
            add_text_block(current_start, index, current_role)
            if gap >= min_instrumental_gap:
                blocks.append(
                    SectionCandidate(
                        start=lines[index - 1].end,
                        end=lines[index].start,
                        section_type="bridge",
                        text="",
                        lyric_evidence="长时间无完整歌词",
                        vocal_evidence="主唱退出",
                        acoustic_evidence="",
                        boundary_confidence=0.0,
                        type_confidence=0.0,
                        need_human_review=False,
                    )
                )
            current_start = index
            current_role = roles[index]
    add_text_block(current_start, len(lines), current_role)
    return blocks


def add_intro_outro(blocks: list[SectionCandidate], duration: float) -> list[SectionCandidate]:
    if not blocks:
        return []
    result: list[SectionCandidate] = []
    if blocks[0].start > 1.0:
        result.append(
            SectionCandidate(0.0, blocks[0].start, "intro", "", "无完整歌词/氛围引入", "主唱未正式进入", "", 0, 0, False)
        )
    result.extend(blocks)
    if result[-1].end < duration - 1.0:
        result.append(
            SectionCandidate(result[-1].end, duration, "outro", "", "最后歌词结束后的收束", "主唱退出或弱化", "", 0, 0, False)
        )
    absorb_short_fragments_before_bridge(result)
    absorb_terminal_fragments(result, duration)
    remove_interludes_absorbed_by_outro(result)
    return result


def absorb_terminal_fragments(sections: list[SectionCandidate], duration: float) -> None:
    if len(sections) < 2:
        return
    index = len(sections) - 1
    while index >= 0:
        section = sections[index]
        if (
            section.text
            and section.duration <= 5.0
            and section.start / max(duration, 1.0) >= 0.85
            and all(item.section_type in {"interlude", "outro"} for item in sections[index + 1 :])
        ):
            next_outro = next((item for item in sections[index + 1 :] if item.section_type == "outro"), None)
            if next_outro is not None:
                next_outro.start = section.start
                next_outro.text = " ".join(part for part in (section.text, next_outro.text) if part).strip()
                next_outro.lyric_evidence = "尾句重复/收束"
                sections.pop(index)
            else:
                section.section_type = "outro"
                section.lyric_evidence = "尾句重复/收束"
        index -= 1


def absorb_short_fragments_before_bridge(sections: list[SectionCandidate]) -> None:
    index = 1
    while index < len(sections) - 1:
        section = sections[index]
        following = sections[index + 1]
        if section.text and section.duration <= 5.0 and following.section_type == "bridge":
            previous = sections[index - 1]
            previous.end = section.end
            previous.text = " ".join(part for part in (previous.text, section.text) if part).strip()
            sections.pop(index)
            continue
        index += 1


def remove_interludes_absorbed_by_outro(sections: list[SectionCandidate]) -> None:
    outro = next((item for item in reversed(sections) if item.section_type == "outro"), None)
    if outro is None:
        return
    sections[:] = [
        section
        for section in sections
        if not (section.section_type == "interlude" and section.start >= outro.start)
    ]


def refine_instrumental_labels(sections: list[SectionCandidate], duration: float) -> None:
    for index, section in enumerate(sections):
        if section.section_type not in {"interlude", "bridge"}:
            continue
        if section.text:
            continue
        position = section.start / max(duration, 1.0)
        previous_type = sections[index - 1].section_type if index > 0 else ""
        next_type = sections[index + 1].section_type if index + 1 < len(sections) else ""
        if section.duration >= 8.0:
            section.section_type = "bridge"
            section.lyric_evidence = "长时间无主唱/无完整歌词，作为桥段或器乐桥段候选"
        elif position >= 0.72 and next_type == "chorus":
            section.section_type = "bridge"
            section.lyric_evidence = "中后段无歌词对比段"
        elif previous_type == "chorus" and next_type in {"verse", "pre-chorus", "chorus"}:
            section.section_type = "bridge"
            section.lyric_evidence = "副歌后短器乐过渡，按中段连接功能标为 bridge"


def merge_adjacent_same_type(sections: list[SectionCandidate], max_duration: float = 52.0) -> list[SectionCandidate]:
    merged: list[SectionCandidate] = []
    for section in sections:
        if (
            merged
            and merged[-1].section_type == section.section_type
            and abs(merged[-1].end - section.start) <= 0.1
            and section.section_type not in {"intro", "outro", "interlude", "bridge"}
            and section.end - merged[-1].start <= max_duration
        ):
            merged[-1].end = section.end
            merged[-1].text = " ".join(part for part in (merged[-1].text, section.text) if part).strip()
            continue
        merged.append(section)
    return merged


def classify_evidence(section: SectionCandidate, features: dict[str, object], title: str) -> None:
    rms = average_curve(features, "rms", section.start, section.end)
    onset = average_curve(features, "onset", section.start, section.end)
    novelty_start = strongest_novelty(features, max(0, section.start - 1.0), section.start + 1.0)[1]
    novelty_end = strongest_novelty(features, max(0, section.end - 1.0), min(float(features["duration"]), section.end + 1.0))[1]
    boundary_strength = max(novelty_start, novelty_end)

    if section.section_type == "intro":
        section.lyric_evidence = section.lyric_evidence or "开头无完整歌词"
        section.vocal_evidence = section.vocal_evidence or "完整主唱尚未进入"
        section.acoustic_evidence = "建立氛围；边界靠首次主唱/稳定伴奏进入"
        section.type_confidence = 0.78
    elif section.section_type == "outro":
        section.lyric_evidence = section.lyric_evidence or "最后歌词后进入收束"
        section.vocal_evidence = section.vocal_evidence or "主唱减少或退出"
        section.acoustic_evidence = "歌曲尾部材料；需检查是否只是最后副歌延展"
        section.type_confidence = 0.72
    elif section.section_type == "chorus":
        title_hit = text_similarity(section.text, title) >= 0.45 if title else False
        section.lyric_evidence = "标题/hook复现" if title_hit else "重复歌词或核心 hook 候选"
        section.vocal_evidence = "完整主唱；通常承担主题释放"
        section.acoustic_evidence = acoustic_phrase(rms, onset, boundary_strength, "副歌候选")
        section.type_confidence = 0.82 if title_hit else 0.68
    elif section.section_type == "pre-chorus":
        section.lyric_evidence = "位于 Verse 和 Chorus 之间，歌词功能偏蓄力/过渡"
        section.vocal_evidence = "主唱连续，预备进入 hook"
        section.acoustic_evidence = acoustic_phrase(rms, onset, boundary_strength, "进入副歌前变化")
        section.type_confidence = 0.62
    elif section.section_type == "bridge":
        section.lyric_evidence = section.lyric_evidence or "中后段一次性材料/对比材料候选"
        section.vocal_evidence = section.vocal_evidence or "可能为新视角、转折或无主唱器乐段"
        section.acoustic_evidence = section.acoustic_evidence or acoustic_phrase(rms, onset, boundary_strength, "对比段候选")
        section.type_confidence = max(section.type_confidence, 0.6)
    elif section.section_type == "interlude":
        section.lyric_evidence = section.lyric_evidence or "段落间无完整歌词"
        section.vocal_evidence = section.vocal_evidence or "主唱退出或只剩背景人声"
        section.acoustic_evidence = acoustic_phrase(rms, onset, boundary_strength, "器乐过渡")
        section.type_confidence = 0.7
    else:
        section.lyric_evidence = "低重复叙事推进"
        section.vocal_evidence = "完整主唱叙事"
        section.acoustic_evidence = acoustic_phrase(rms, onset, boundary_strength, "稳定主歌伴奏")
        section.type_confidence = 0.7

    lyric_boundary = 0.8 if section.section_type in {"intro", "outro", "interlude", "bridge"} and not section.text else 0.65
    section.boundary_confidence = round(min(0.95, 0.35 + 0.45 * boundary_strength + 0.2 * lyric_boundary), 2)
    section.type_confidence = round(section.type_confidence, 2)


def refresh_section_evidence(sections: list[SectionCandidate], features: dict[str, object], title: str) -> None:
    for section in sections:
        classify_evidence(section, features, title)
    mark_reviews(sections)


def split_overlong_repeated_sections(
    sections: list[SectionCandidate],
    lines: list[LyricLine],
    context: StructureContext,
) -> list[SectionCandidate]:
    """Split overlong sung sections at internal lyric-cycle restarts."""
    result: list[SectionCandidate] = []
    for section in sections:
        if section.section_type not in {"chorus", "verse", "pre-chorus"}:
            result.append(section)
            continue
        max_expected = context.max_chorus_seed_seconds * 1.2
        if section.duration <= max_expected:
            result.append(section)
            continue
        section_lines = lines_in_section(lines, section)
        split_points = lyric_cycle_split_points(section, section_lines, context)
        if not split_points:
            result.append(section)
            continue
        result.extend(split_section_at_points(section, section_lines, split_points))
    return result


def lines_in_section(lines: list[LyricLine], section: SectionCandidate) -> list[LyricLine]:
    return [
        line
        for line in lines
        if line.end > section.start + 0.05 and line.start < section.end - 0.05
    ]


def lyric_cycle_split_points(
    section: SectionCandidate,
    section_lines: list[LyricLine],
    context: StructureContext,
) -> list[float]:
    if len(section_lines) < 6:
        return []
    min_piece = max(context.max_pre_chorus_seconds * 0.5, 12.0)
    anchors = section_lines[: min(4, max(1, len(section_lines) // 3))]
    split_points: list[float] = []
    last_split = section.start
    for index, line in enumerate(section_lines[1:], start=1):
        if line.start - last_split < min_piece:
            continue
        if section.end - line.start < min_piece:
            continue
        if not resembles_any_anchor(line, anchors):
            continue
        split_points.append(round(line.start, 2))
        last_split = line.start
        anchors = section_lines[index : index + min(4, max(1, len(section_lines[index:]) // 3))]
    return split_points


def resembles_any_anchor(line: LyricLine, anchors: list[LyricLine]) -> bool:
    return any(text_similarity(line.text, anchor.text) >= 0.56 for anchor in anchors)


def split_section_at_points(
    section: SectionCandidate,
    section_lines: list[LyricLine],
    split_points: list[float],
) -> list[SectionCandidate]:
    boundaries = [section.start, *split_points, section.end]
    pieces: list[SectionCandidate] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        text = " ".join(line.text for line in section_lines if line.end > start and line.start < end).strip()
        piece = clone_section(section, start, end)
        piece.text = text
        piece.lyric_evidence = "过长同类段按歌词重复周期拆分"
        piece.need_human_review = True
        pieces.append(piece)
    return pieces


def promote_functional_bridges(sections: list[SectionCandidate], duration: float) -> None:
    """Promote contrast sections that break repeated chorus cycles."""
    chorus_indices = [index for index, section in enumerate(sections) if section.section_type == "chorus"]
    if len(chorus_indices) < 2:
        return

    for index, section in enumerate(sections):
        if section.section_type not in {"verse", "pre-chorus"}:
            continue
        min_bridge_candidate = clamp(duration * 0.025, 6.0, 12.0)
        max_bridge_candidate = clamp(duration * 0.18, 28.0, 55.0)
        if section.duration < min_bridge_candidate or section.duration > max_bridge_candidate:
            continue
        previous_choruses = sum(1 for chorus_index in chorus_indices if chorus_index < index)
        has_later_chorus = any(chorus_index > index for chorus_index in chorus_indices)
        if previous_choruses < 2 or not has_later_chorus:
            continue
        previous_core = nearest_core_section(sections, index, -1)
        next_core = nearest_core_section(sections, index, 1)
        if section.start / max(duration, 1.0) < 0.45:
            continue

        sandwiched_by_chorus = (
            previous_core is not None
            and next_core is not None
            and previous_core.section_type == "chorus"
            and next_core.section_type == "chorus"
        )
        contrast = is_contrast_section(section, sections[:index])
        if sandwiched_by_chorus or contrast:
            section.section_type = "bridge"
            section.lyric_evidence = "重复副歌循环后的对比/转折材料，连接回后续 chorus"
            section.vocal_evidence = "有歌词桥段候选；功能是打破 verse-chorus 循环"
            section.acoustic_evidence = section.acoustic_evidence or "结构位置提示 bridge；需结合听感复核"
            section.type_confidence = max(section.type_confidence, 0.66)
            section.need_human_review = True


def nearest_core_section(sections: list[SectionCandidate], index: int, step: int) -> SectionCandidate | None:
    cursor = index + step
    while 0 <= cursor < len(sections):
        if sections[cursor].section_type not in {"intro", "outro"}:
            return sections[cursor]
        cursor += step
    return None


def is_contrast_section(section: SectionCandidate, previous_sections: list[SectionCandidate]) -> bool:
    if not section.text:
        return True
    comparable = [
        previous
        for previous in previous_sections
        if previous.text and previous.section_type in {"verse", "pre-chorus", "bridge"}
    ]
    if not comparable:
        return True
    best_similarity = max(text_similarity(section.text, previous.text) for previous in comparable)
    return best_similarity < 0.46


def acoustic_phrase(rms: float, onset: float, novelty: float, prefix: str) -> str:
    parts = [prefix]
    if novelty >= 0.65:
        parts.append("边界附近 novelty 强")
    elif novelty >= 0.45:
        parts.append("边界附近 novelty 中等")
    else:
        parts.append("声学边界较弱")
    if rms >= 0.14:
        parts.append("能量较高")
    if onset >= 1.3:
        parts.append("onset/鼓点活跃")
    return "；".join(parts)


def snap_section_boundaries(sections: list[SectionCandidate], features: dict[str, object]) -> None:
    if not sections:
        return
    sections[0].start = 0.0
    for index in range(1, len(sections)):
        raw = sections[index].start
        snapped, strength = snap_boundary(features, raw)
        if abs(snapped - raw) <= 1.5 and strength >= 0.58:
            sections[index - 1].end = snapped
            sections[index].start = snapped
    duration = float(features["duration"])
    sections[-1].end = min(duration, sections[-1].end)


def mark_reviews(sections: list[SectionCandidate]) -> None:
    for index, section in enumerate(sections):
        reasons = [
            section.boundary_confidence < 0.65,
            section.type_confidence < 0.65,
            section.duration < 7.0 and section.section_type not in {"intro", "outro"},
        ]
        if index > 0 and sections[index - 1].section_type == section.section_type:
            reasons.append(True)
        if section.section_type == "chorus" and section.duration > 55.0:
            reasons.append(True)
        section.need_human_review = any(reasons)


@property
def duration(self: SectionCandidate) -> float:
    return max(0.0, self.end - self.start)


SectionCandidate.duration = duration  # type: ignore[attr-defined]


def structure_label(section_type: str) -> str:
    return {
        "intro": "A",
        "verse": "B",
        "pre-chorus": "C",
        "pre_chorus": "C",
        "chorus": "D",
        "bridge": "E",
        "outro": "A'",
    }.get(section_type, "")


def public_section_type(section_type: str) -> str:
    return "pre_chorus" if section_type == "pre-chorus" else section_type


def build_sections(lines: list[LyricLine], title: str, features: dict[str, object], min_instrumental_gap: float) -> list[SectionCandidate]:
    roles = initial_line_roles(lines, title)
    context = structure_context(lines)
    sections = make_role_blocks(lines, roles, min_instrumental_gap)
    sections = add_intro_outro(sections, float(features["duration"]))
    refine_instrumental_labels(sections, float(features["duration"]))
    sections = merge_adjacent_same_type(sections)
    snap_section_boundaries(sections, features)
    promote_functional_bridges(sections, float(features["duration"]))
    sections = split_overlong_repeated_sections(sections, lines, context)
    refresh_section_evidence(sections, features, title)
    return sections


def section_from_low_vocal(start: float, end: float, duration: float) -> SectionCandidate:
    if start <= 1.5:
        section_type = "intro"
        lyric_evidence = "分离人声轨显示开头长时间低人声"
        vocal_evidence = "Demucs vocals stem 低能量"
    elif end >= duration - 2.5 or start >= duration * 0.86:
        section_type = "outro"
        lyric_evidence = "分离人声轨显示尾部长时间低人声/收束"
        vocal_evidence = "Demucs vocals stem 低能量"
    else:
        section_type = "bridge"
        lyric_evidence = "分离人声轨显示中间长时间无主唱/弱主唱"
        vocal_evidence = "Demucs vocals stem 低能量"
    return SectionCandidate(
        start=start,
        end=end,
        section_type=section_type,
        text="",
        lyric_evidence=lyric_evidence,
        vocal_evidence=vocal_evidence,
        acoustic_evidence="人声分离后低 vocal RMS 区间",
        boundary_confidence=0.82,
        type_confidence=0.78 if section_type != "bridge" else 0.72,
        need_human_review=section_type == "bridge",
    )


def apply_low_vocal_regions(
    sections: list[SectionCandidate],
    low_vocal_regions: list[tuple[float, float]],
    duration: float,
    min_region_duration: float = 8.0,
) -> list[SectionCandidate]:
    """Split sung sections around Demucs low-vocal regions."""
    strong_regions = [
        (start, end)
        for start, end in low_vocal_regions
        if end - start >= min_region_duration
    ]
    if not strong_regions:
        return sections

    result = sections
    for raw_start, raw_end in strong_regions:
        start = round(max(0.0, raw_start), 2)
        end = round(min(duration, raw_end), 2)
        if end <= start:
            continue
        inserted = section_from_low_vocal(start, end, duration)
        next_result: list[SectionCandidate] = []
        for section in result:
            if section.end <= start or section.start >= end:
                next_result.append(section)
                continue
            if section.start < start:
                before = clone_section(section, section.start, start)
                if before.duration >= 1.0:
                    next_result.append(before)
            next_result.append(inserted)
            if section.end > end:
                after = clone_section(section, end, section.end)
                if after.duration >= 1.0:
                    next_result.append(after)
        result = sorted(next_result, key=lambda item: (item.start, item.end))

    return close_tiny_gaps(merge_overlaps_and_neighbors(result))


def clone_section(section: SectionCandidate, start: float, end: float) -> SectionCandidate:
    return SectionCandidate(
        start=round(start, 2),
        end=round(end, 2),
        section_type=section.section_type,
        text=section.text,
        lyric_evidence=section.lyric_evidence,
        vocal_evidence=section.vocal_evidence,
        acoustic_evidence=section.acoustic_evidence,
        boundary_confidence=min(section.boundary_confidence, 0.7),
        type_confidence=section.type_confidence,
        need_human_review=True if section.duration != end - start else section.need_human_review,
    )


def merge_overlaps_and_neighbors(sections: list[SectionCandidate]) -> list[SectionCandidate]:
    cleaned: list[SectionCandidate] = []
    for section in sections:
        if section.duration < 1.0:
            continue
        if cleaned and section.section_type == cleaned[-1].section_type and section.start <= cleaned[-1].end + 0.1:
            cleaned[-1].end = max(cleaned[-1].end, section.end)
            cleaned[-1].text = " ".join(part for part in (cleaned[-1].text, section.text) if part).strip()
            cleaned[-1].need_human_review = cleaned[-1].need_human_review or section.need_human_review
            continue
        if cleaned and section.start < cleaned[-1].end:
            section.start = cleaned[-1].end
        cleaned.append(section)
    return cleaned


def close_tiny_gaps(sections: list[SectionCandidate], max_gap: float = 1.0) -> list[SectionCandidate]:
    for index in range(len(sections) - 1):
        current = sections[index]
        following = sections[index + 1]
        gap = following.start - current.end
        if 0 < gap <= max_gap:
            if following.section_type in {"bridge", "outro"}:
                following.start = current.end
            else:
                current.end = following.start
    return sections


def write_sections(path: Path, song_id: str, sections: list[SectionCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "song_id",
        "section_index",
        "start_time",
        "end_time",
        "duration",
        "section_type",
        "structure_label",
        "lyric_evidence",
        "vocal_evidence",
        "acoustic_evidence",
        "boundary_confidence",
        "type_confidence",
        "need_human_review",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, section in enumerate(sections, start=1):
            writer.writerow(
                {
                    "song_id": song_id,
                    "section_index": index,
                    "start_time": base.format_timestamp(section.start),
                    "end_time": base.format_timestamp(section.end),
                    "duration": base.format_timestamp(section.duration),
                    "section_type": public_section_type(section.section_type),
                    "structure_label": structure_label(section.section_type),
                    "lyric_evidence": section.lyric_evidence,
                    "vocal_evidence": section.vocal_evidence,
                    "acoustic_evidence": section.acoustic_evidence,
                    "boundary_confidence": section.boundary_confidence,
                    "type_confidence": section.type_confidence,
                    "need_human_review": str(section.need_human_review).lower(),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run functional section segmentation for pop songs.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--song-id", default="")
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--work-dir", default="outputs/functional_sections")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-transcript", action="store_true")
    parser.add_argument("--min-instrumental-gap", type=float, default=8.0)
    parser.add_argument("--vocals-stem", help="Optional Demucs vocals.wav for low-vocal bridge/intro/outro correction.")
    args = parser.parse_args()

    audio = Path(args.audio)
    work_dir = Path(args.work_dir)
    song_id = args.song_id or audio.stem
    transcript_csv = work_dir / f"{audio.stem}_whisper_segments.csv"

    if not args.reuse_transcript or not transcript_csv.exists():
        run_transcription(audio, transcript_csv, args.model, args.language)

    lines = read_transcript(transcript_csv)
    features = load_acoustic_features(audio)
    sections = build_sections(lines, args.title, features, args.min_instrumental_gap)
    if args.vocals_stem:
        low_vocal_regions, _stats = vocal_activity.detect_low_vocal_regions(Path(args.vocals_stem), min_duration=5.0)
        sections = apply_low_vocal_regions(sections, low_vocal_regions, float(features["duration"]))
        promote_functional_bridges(sections, float(features["duration"]))
        refresh_section_evidence(sections, features, args.title)
    write_sections(Path(args.output), song_id, sections)

    print(f"Transcript: {transcript_csv}")
    print(f"Sections: {args.output}")
    for index, section in enumerate(sections, start=1):
        print(
            f"{index:02d} {base.format_timestamp(section.start)}-{base.format_timestamp(section.end)} "
            f"{section.section_type} "
            f"boundary={section.boundary_confidence:.2f} type={section.type_confidence:.2f} "
            f"review={str(section.need_human_review).lower()}"
        )


if __name__ == "__main__":
    main()
