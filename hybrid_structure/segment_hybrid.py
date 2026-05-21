#!/usr/bin/env python3
"""Hybrid lyric-boundary and audio-evidence section labeling."""

from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CandidateBlock:
    start_time: float
    end_time: float
    text: str


@dataclass
class AudioStats:
    rms: float
    onset: float
    vocal_proxy: float
    centroid: float
    energy_rise: float = 0.0


@dataclass
class Section:
    section_id: int
    start_time: float
    end_time: float
    duration: float
    label: str
    structure_label: str


@dataclass
class DebugRow:
    section_id: int
    start_time: str
    end_time: str
    label: str
    hook_score: float
    build_score: float
    repetition_score: float
    rms: float
    onset: float
    vocal_proxy: float
    energy_rise: float
    reason: str


def parse_timestamp(value: str) -> float:
    value = value.strip()
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"unsupported timestamp: {value}")


def format_timestamp(value: float) -> str:
    value = max(0.0, float(value))
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes}:{seconds:05.2f}"


def normalize_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower(), flags=re.UNICODE)


def text_similarity(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm in right_norm or right_norm in left_norm:
        return 1.0
    return float(SequenceMatcher(None, left_norm, right_norm).ratio())


def structure_label(label: str) -> str:
    return {
        "Intro": "A",
        "Outro": "A",
        "Verse": "B",
        "Pre-chorus": "C",
        "Chorus": "D",
        "Bridge": "E",
    }.get(label, "")


def load_candidate_blocks(path: Path) -> list[CandidateBlock]:
    blocks: list[CandidateBlock] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            start = parse_timestamp(row["start_time"])
            end = parse_timestamp(row["end_time"])
            if end <= start:
                continue
            blocks.append(CandidateBlock(start, end, (row.get("text") or "").strip()))
    return sorted(blocks, key=lambda block: block.start_time)


def load_audio_features(audio_path: Path) -> dict[str, object]:
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    hop_length = 512
    harmonic, percussive = librosa.effects.hpss(y)
    rms = librosa.feature.rms(y=y, hop_length=hop_length).reshape(-1)
    harmonic_rms = librosa.feature.rms(y=harmonic, hop_length=hop_length).reshape(-1)
    percussive_rms = librosa.feature.rms(y=percussive, hop_length=hop_length).reshape(-1)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).reshape(-1)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length).reshape(-1)
    vocal_proxy = harmonic_rms / (rms + 1e-8) * (1.0 - np.clip(percussive_rms / (rms + 1e-8), 0.0, 1.0))
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
    return {
        "duration": float(librosa.get_duration(y=y, sr=sr)),
        "times": times,
        "rms": rms,
        "onset": onset,
        "vocal_proxy": vocal_proxy,
        "centroid": centroid,
    }


def average_curve(features: dict[str, object], curve_name: str, start: float, end: float) -> float:
    import numpy as np

    times = features["times"]
    curve = features[curve_name]
    mask = (times >= start) & (times < end)
    if not np.any(mask):
        return 0.0
    return float(np.mean(curve[mask]))


def audio_stats_for_blocks(features: dict[str, object], blocks: list[CandidateBlock]) -> list[AudioStats]:
    stats: list[AudioStats] = []
    for block in blocks:
        stats.append(
            AudioStats(
                rms=average_curve(features, "rms", block.start_time, block.end_time),
                onset=average_curve(features, "onset", block.start_time, block.end_time),
                vocal_proxy=average_curve(features, "vocal_proxy", block.start_time, block.end_time),
                centroid=average_curve(features, "centroid", block.start_time, block.end_time),
            )
        )
    for index, item in enumerate(stats[:-1]):
        item.energy_rise = stats[index + 1].rms - item.rms
    return stats


def scale_values(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def text_scores(blocks: list[CandidateBlock], title: str) -> list[dict[str, float]]:
    norms = [normalize_text(block.text) for block in blocks]
    title_norm = normalize_text(title)
    counts: dict[str, int] = {}
    for norm in norms:
        if norm:
            counts[norm] = counts.get(norm, 0) + 1

    hook_words = ("hook", "title", "repeated", "light", "hope", "sunshine", "bloom", "chorus")
    build_words = ("build", "transition", "flowing", "toward", "rise", "lift")
    scores: list[dict[str, float]] = []
    for block, norm in zip(blocks, norms):
        words = block.text.lower().split()
        hook_hits = sum(1 for word in hook_words if word in words)
        build_hits = sum(1 for word in build_words if word in words)
        title_hit = max((text_similarity(title, part) for part in block.text.split()), default=0.0) if title_norm else 0.0
        title_hit = max(title_hit, text_similarity(title, block.text)) if title_norm else 0.0
        repetition = 1.0 if norm and counts.get(norm, 0) > 1 else 0.0
        no_lyrics = 1.0 if not norm else 0.0
        scores.append(
            {
                "hook": min(1.0, (1.0 if title_hit >= 0.58 else 0.0) + hook_hits / 3.0),
                "build": min(1.0, build_hits / 2.0),
                "repetition": repetition,
                "no_lyrics": no_lyrics,
            }
        )
    return scores


def make_sections(
    blocks: list[CandidateBlock],
    features: dict[str, object],
    title: str,
    min_bridge_seconds: float,
) -> tuple[list[Section], list[DebugRow]]:
    stats = audio_stats_for_blocks(features, blocks)
    scores = text_scores(blocks, title)
    scaled_rms = scale_values([item.rms for item in stats])
    scaled_onset = scale_values([item.onset for item in stats])
    scaled_vocal = scale_values([item.vocal_proxy for item in stats])
    scaled_rise = scale_values([item.energy_rise for item in stats])
    duration = round(float(features["duration"]), 2)

    sections: list[Section] = []
    debug_rows: list[DebugRow] = []

    def append(start: float, end: float, label: str, reason: str, index: int | None = None) -> None:
        if end - start <= 0.5:
            return
        section = Section(
            section_id=len(sections) + 1,
            start_time=round(start, 2),
            end_time=round(end, 2),
            duration=round(end - start, 2),
            label=label,
            structure_label=structure_label(label),
        )
        sections.append(section)
        if index is None:
            debug_rows.append(
                DebugRow(section.section_id, format_timestamp(start), format_timestamp(end), label, 0, 0, 0, 0, 0, 0, 0, reason)
            )
        else:
            debug_rows.append(
                DebugRow(
                    section.section_id,
                    format_timestamp(start),
                    format_timestamp(end),
                    label,
                    round(scores[index]["hook"], 3),
                    round(scores[index]["build"], 3),
                    round(scores[index]["repetition"], 3),
                    round(stats[index].rms, 5),
                    round(stats[index].onset, 5),
                    round(stats[index].vocal_proxy, 5),
                    round(stats[index].energy_rise, 5),
                    reason,
                )
            )

    if blocks and blocks[0].start_time > 1.0:
        append(0.0, blocks[0].start_time, "Intro", "audio before first lyric block")

    preliminary: list[str] = []
    for index, block in enumerate(blocks):
        block_duration = block.end_time - block.start_time
        song_position = block.start_time / max(duration, 1.0)
        is_final_lyric_block = index == len(blocks) - 1 or all(
            next_score["no_lyrics"] >= 1.0 for next_score in scores[index + 1 :]
        )
        if scores[index]["no_lyrics"] >= 1.0 and block_duration >= min_bridge_seconds:
            preliminary.append("Bridge")
            continue
        if is_final_lyric_block and (song_position >= 0.82 or "outro" in block.text.lower()):
            preliminary.append("Outro")
            continue
        if scores[index]["hook"] >= 0.65:
            preliminary.append("Chorus")
            continue
        if scores[index]["build"] >= 0.75:
            preliminary.append("Pre-chorus")
            continue
        preliminary.append("Verse")

    for index, label in enumerate(preliminary[:-1]):
        if label == "Verse" and preliminary[index + 1] == "Chorus":
            block_duration = blocks[index].end_time - blocks[index].start_time
            if scores[index]["build"] >= 0.45 or (scaled_rise[index] >= 0.55 and block_duration <= 18.0):
                preliminary[index] = "Pre-chorus"

    for index in range(1, len(preliminary) - 1):
        block_duration = blocks[index].end_time - blocks[index].start_time
        if (
            preliminary[index] == "Pre-chorus"
            and preliminary[index - 1] == "Chorus"
            and preliminary[index + 1] == "Chorus"
            and block_duration <= 8.0
        ):
            preliminary[index] = "Chorus"

    for index, (block, label) in enumerate(zip(blocks, preliminary)):
        reason = (
            f"hook={scores[index]['hook']:.2f}, build={scores[index]['build']:.2f}, "
            f"rms_norm={scaled_rms[index]:.2f}, onset_norm={scaled_onset[index]:.2f}, rise_norm={scaled_rise[index]:.2f}"
        )
        append(block.start_time, block.end_time, label, reason, index)

    if blocks and blocks[-1].end_time < duration - 1.0:
        append(blocks[-1].end_time, duration, "Outro", "audio after last lyric block")

    close_small_gaps_by_audio_similarity(sections, debug_rows, features)
    merge_adjacent_same_labels(sections, debug_rows)
    absorb_tiny_sections(sections, debug_rows, features)
    return sections, debug_rows


def merge_adjacent_same_labels(sections: list[Section], debug_rows: list[DebugRow], max_combined_duration: float = 48.0) -> None:
    index = 0
    while index < len(sections) - 1:
        current = sections[index]
        following = sections[index + 1]
        combined_duration = following.end_time - current.start_time
        if (
            current.label == following.label
            and abs(current.end_time - following.start_time) <= 0.05
            and (current.label in {"Intro", "Outro"} or combined_duration <= max_combined_duration)
        ):
            current.end_time = following.end_time
            current.duration = round(current.end_time - current.start_time, 2)
            debug_rows[index].end_time = format_timestamp(current.end_time)
            debug_rows[index].reason = f"{debug_rows[index].reason}; merged adjacent {current.label}"
            sections.pop(index + 1)
            debug_rows.pop(index + 1)
            continue
        index += 1
    for section_id, section in enumerate(sections, start=1):
        section.section_id = section_id
        debug_rows[section_id - 1].section_id = section_id


def absorb_tiny_sections(
    sections: list[Section],
    debug_rows: list[DebugRow],
    features: dict[str, object],
    min_duration: float = 6.0,
) -> None:
    index = 1
    while index < len(sections) - 1:
        current = sections[index]
        if current.duration >= min_duration or current.label in {"Intro", "Outro", "Bridge"}:
            index += 1
            continue
        previous = sections[index - 1]
        following = sections[index + 1]
        current_vector = section_feature_vector(features, current.start_time, current.end_time)
        previous_vector = section_feature_vector(features, previous.start_time, previous.end_time)
        following_vector = section_feature_vector(features, following.start_time, following.end_time)
        previous_distance = vector_distance(current_vector, previous_vector)
        following_distance = vector_distance(current_vector, following_vector)
        if following_distance < previous_distance:
            old_start = following.start_time
            following.start_time = current.start_time
            following.duration = round(following.end_time - following.start_time, 2)
            debug_rows[index + 1].start_time = format_timestamp(following.start_time)
            debug_rows[index + 1].reason = (
                f"{debug_rows[index + 1].reason}; absorbed tiny {current.label} "
                f"{format_timestamp(current.start_time)}-{format_timestamp(current.end_time)} "
                f"(next_dist={following_distance:.4f}, prev_dist={previous_distance:.4f}, old_start={format_timestamp(old_start)})"
            )
        else:
            old_end = previous.end_time
            previous.end_time = current.end_time
            previous.duration = round(previous.end_time - previous.start_time, 2)
            debug_rows[index - 1].end_time = format_timestamp(previous.end_time)
            debug_rows[index - 1].reason = (
                f"{debug_rows[index - 1].reason}; absorbed tiny {current.label} "
                f"{format_timestamp(current.start_time)}-{format_timestamp(current.end_time)} "
                f"(prev_dist={previous_distance:.4f}, next_dist={following_distance:.4f}, old_end={format_timestamp(old_end)})"
            )
        sections.pop(index)
        debug_rows.pop(index)

    merge_adjacent_same_labels(sections, debug_rows)


def section_feature_vector(features: dict[str, object], start: float, end: float) -> list[float]:
    return [
        average_curve(features, "rms", start, end),
        average_curve(features, "onset", start, end),
        average_curve(features, "vocal_proxy", start, end),
        average_curve(features, "centroid", start, end),
    ]


def vector_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5


def close_small_gaps_by_audio_similarity(
    sections: list[Section],
    debug_rows: list[DebugRow],
    features: dict[str, object],
    max_gap_seconds: float = 6.0,
) -> None:
    """Make final sections continuous by assigning small gaps to the most similar neighbor."""
    if len(sections) < 2:
        return

    for index in range(len(sections) - 1):
        current = sections[index]
        following = sections[index + 1]
        gap = round(following.start_time - current.end_time, 2)
        if gap <= 0:
            continue
        if gap <= max_gap_seconds:
            gap_vector = section_feature_vector(features, current.end_time, following.start_time)
            previous_vector = section_feature_vector(features, current.start_time, current.end_time)
            following_vector = section_feature_vector(features, following.start_time, following.end_time)
            previous_distance = vector_distance(gap_vector, previous_vector)
            following_distance = vector_distance(gap_vector, following_vector)
            if following_distance < previous_distance:
                old_start = following.start_time
                following.start_time = current.end_time
                following.duration = round(following.end_time - following.start_time, 2)
                debug_rows[index + 1].start_time = format_timestamp(following.start_time)
                debug_rows[index + 1].reason = (
                    f"{debug_rows[index + 1].reason}; absorbed {gap:.2f}s gap before section "
                    f"(next_dist={following_distance:.4f}, prev_dist={previous_distance:.4f}, old_start={format_timestamp(old_start)})"
                )
            else:
                old_end = current.end_time
                current.end_time = following.start_time
                current.duration = round(current.end_time - current.start_time, 2)
                debug_rows[index].end_time = format_timestamp(current.end_time)
                debug_rows[index].reason = (
                    f"{debug_rows[index].reason}; absorbed {gap:.2f}s gap after section "
                    f"(prev_dist={previous_distance:.4f}, next_dist={following_distance:.4f}, old_end={format_timestamp(old_end)})"
                )

    for index, section in enumerate(sections, start=1):
        section.section_id = index
        section.duration = round(section.end_time - section.start_time, 2)


def write_sections(path: Path, sections: list[Section]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["section_id", "start_time", "end_time", "duration", "label", "structure_label"],
        )
        writer.writeheader()
        for section in sections:
            writer.writerow(
                {
                    "section_id": section.section_id,
                    "start_time": format_timestamp(section.start_time),
                    "end_time": format_timestamp(section.end_time),
                    "duration": format_timestamp(section.duration),
                    "label": section.label,
                    "structure_label": section.structure_label,
                }
            )


def write_debug(path: Path, rows: list[DebugRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid lyric-boundary/audio-evidence section labeling.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--lyric-sections", required=True, help="CSV with start_time,end_time,text.")
    parser.add_argument("--title", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug-output")
    parser.add_argument("--json-output")
    parser.add_argument("--min-bridge-seconds", type=float, default=8.0)
    args = parser.parse_args()

    blocks = load_candidate_blocks(Path(args.lyric_sections))
    features = load_audio_features(Path(args.audio))
    sections, debug_rows = make_sections(blocks, features, args.title, args.min_bridge_seconds)
    write_sections(Path(args.output), sections)
    if args.debug_output:
        write_debug(Path(args.debug_output), debug_rows)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(section) for section in sections], ensure_ascii=False, indent=2), encoding="utf-8")

    for section in sections:
        print(
            f"{section.section_id:02d} "
            f"{format_timestamp(section.start_time)}-{format_timestamp(section.end_time)} "
            f"{format_timestamp(section.duration)} "
            f"{section.label} {section.structure_label}"
        )


if __name__ == "__main__":
    main()
