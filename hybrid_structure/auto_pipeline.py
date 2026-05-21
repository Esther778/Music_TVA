#!/usr/bin/env python3
"""End-to-end mp3-to-section prototype using Whisper timestamps."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path
from difflib import SequenceMatcher

import segment_hybrid as hybrid


def read_transcript(path: Path) -> list[hybrid.CandidateBlock]:
    rows: list[hybrid.CandidateBlock] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            text = clean_transcript_text((row.get("text") or "").strip())
            if not text or not has_cjk(text) or is_metadata_line(text):
                continue
            start = hybrid.parse_timestamp(row["start_time"])
            end = hybrid.parse_timestamp(row["end_time"])
            rows.append(
                hybrid.CandidateBlock(
                    start,
                    end,
                    text,
                )
            )
    return rows


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def clean_transcript_text(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    return text.strip()


def is_metadata_line(text: str) -> bool:
    compact = hybrid.normalize_text(text)
    metadata_terms = ("词曲", "詞曲", "作词", "作詞", "作曲", "编曲", "編曲", "混音", "母带", "母帶", "制作人", "製作人", "李宗盛", "composer", "lyrics")
    if any(term in compact for term in metadata_terms):
        return True
    return len(compact) <= 6 and compact.startswith(("曲", "词", "詞"))


def fuzzy_similarity(left: str, right: str) -> float:
    left_norm = hybrid.normalize_text(left)
    right_norm = hybrid.normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm in right_norm or right_norm in left_norm:
        return 1.0
    return float(SequenceMatcher(None, left_norm, right_norm).ratio())


def is_title_hook(block: hybrid.CandidateBlock, title: str) -> bool:
    if title and fuzzy_similarity(block.text, title) >= 0.55:
        return True
    return False


def is_verse_restart(block: hybrid.CandidateBlock, first_lyric_text: str) -> bool:
    return fuzzy_similarity(block.text, first_lyric_text) >= 0.55


def phrase_similarity(left: hybrid.CandidateBlock, right: hybrid.CandidateBlock) -> float:
    left_text = re.sub(r"^[啊呀哦喔呜嗯]+", "", left.text)
    right_text = re.sub(r"^[啊呀哦喔呜嗯]+", "", right.text)
    return fuzzy_similarity(left_text, right_text)


def find_repeated_spans(
    transcript: list[hybrid.CandidateBlock],
    min_len: int = 2,
    max_len: int = 8,
    similarity_threshold: float = 0.58,
) -> list[tuple[int, int]]:
    """Find repeated phrase spans as candidate refrain/chorus material."""
    candidates: list[tuple[int, int, float]] = []
    n = len(transcript)
    for first in range(n):
        for second in range(first + min_len, n):
            if transcript[second].start_time - transcript[first].start_time < 20.0:
                continue
            length = 0
            scores: list[float] = []
            while first + length < n and second + length < n and length < max_len:
                score = phrase_similarity(transcript[first + length], transcript[second + length])
                if score < similarity_threshold:
                    break
                scores.append(score)
                length += 1
            if length >= min_len:
                duration = transcript[first + length - 1].end_time - transcript[first].start_time
                if duration >= 5.0:
                    avg_score = sum(scores) / len(scores)
                    candidates.append((first, first + length, avg_score))
                    candidates.append((second, second + length, avg_score))

    candidates.sort(key=lambda item: ((item[1] - item[0]), item[2]), reverse=True)
    selected: list[tuple[int, int]] = []
    for start, end, _score in candidates:
        if any(not (end <= s or start >= e) for s, e in selected):
            continue
        selected.append((start, end))
    return sorted(selected)


def group_introductory_phrases(transcript: list[hybrid.CandidateBlock]) -> list[int]:
    boundaries = {0, len(transcript)}
    for index in range(1, len(transcript)):
        gap = transcript[index].start_time - transcript[index - 1].end_time
        if gap >= 8.0:
            boundaries.add(index)
    return sorted(boundaries)


def make_blocks_from_boundaries(
    transcript: list[hybrid.CandidateBlock],
    boundaries: list[int],
    repeated_spans: list[tuple[int, int]],
    min_gap_bridge_seconds: float,
) -> list[hybrid.CandidateBlock]:
    repeated_set = set(repeated_spans)
    blocks: list[hybrid.CandidateBlock] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        items = transcript[start:end]
        text = " ".join(item.text for item in items)
        if (start, end) in repeated_set:
            text = f"hook repeated {text}"
        block = hybrid.CandidateBlock(items[0].start_time, items[-1].end_time, text)
        if blocks:
            gap = block.start_time - blocks[-1].end_time
            if gap >= min_gap_bridge_seconds:
                blocks.append(hybrid.CandidateBlock(blocks[-1].end_time, block.start_time, ""))
        blocks.append(block)
    return blocks


def sentence_roles_from_repetition(
    transcript: list[hybrid.CandidateBlock],
    title: str,
    min_gap_bridge_seconds: float,
) -> list[str]:
    roles = ["Verse" for _ in transcript]
    repeated_spans = find_repeated_spans(transcript)

    for start, end in repeated_spans:
        for index in range(start, end):
            roles[index] = "Chorus"

    for index, item in enumerate(transcript):
        if is_title_hook(item, title):
            roles[index] = "Chorus"

    apply_title_hook_guard(transcript, roles, title)

    chorus_starts = [
        index
        for index, role in enumerate(roles)
        if role == "Chorus" and (index == 0 or roles[index - 1] != "Chorus")
    ]
    for chorus_start in chorus_starts:
        pre_start = max(0, chorus_start - 4)
        for index in range(pre_start, chorus_start):
            if roles[index] == "Chorus":
                continue
            if transcript[chorus_start].start_time - transcript[index].end_time > 24.0:
                continue
            if index + 1 < len(transcript) and transcript[index + 1].start_time - transcript[index].end_time >= min_gap_bridge_seconds:
                continue
            roles[index] = "Pre-chorus"

    extend_chorus_semantic_paragraphs(transcript, roles)
    protect_semantic_continuity(transcript, roles)
    return roles


def apply_title_hook_guard(transcript: list[hybrid.CandidateBlock], roles: list[str], title: str) -> None:
    """When title hooks exist, keep repeated verse text from becoming chorus."""
    title_hook_indices = [index for index, item in enumerate(transcript) if is_title_hook(item, title)]
    if not title_hook_indices:
        return
    title_hook_set = set(title_hook_indices)
    first_title_hook = min(title_hook_indices)
    for index, role in enumerate(roles):
        if role != "Chorus":
            continue
        if index < first_title_hook:
            roles[index] = "Verse"
            continue
        if index not in title_hook_set and any(0 < hook_index - index <= 4 for hook_index in title_hook_indices):
            roles[index] = "Verse"
            continue
        nearest_title_hook = min(abs(index - hook_index) for hook_index in title_hook_indices)
        if nearest_title_hook > 4:
            roles[index] = "Verse"


def starts_with_same_phrase(left_text: str, right_text: str, min_chars: int = 2) -> bool:
    left_norm = hybrid.normalize_text(left_text)
    right_norm = hybrid.normalize_text(right_text)
    if len(left_norm) < min_chars or len(right_norm) < min_chars:
        return False
    return left_norm[:min_chars] == right_norm[:min_chars]


def parallel_line_similarity(left: hybrid.CandidateBlock, right: hybrid.CandidateBlock) -> float:
    score = semantic_similarity(left, right)
    if starts_with_same_phrase(left.text, right.text):
        score = max(score, 0.55)
    return score


def extend_chorus_semantic_paragraphs(transcript: list[hybrid.CandidateBlock], roles: list[str]) -> None:
    """Promote parallel repeated lyric paragraphs around known hook anchors."""
    n = len(transcript)
    for first in range(n):
        if roles[first] != "Chorus":
            continue
        for second in range(first + 1, n):
            if roles[second] != "Chorus":
                continue
            if transcript[second].start_time - transcript[first].start_time < 20.0:
                continue
            if parallel_line_similarity(transcript[first], transcript[second]) < 0.55:
                continue

            length = 0
            while first + length < n and second + length < n and length < 14:
                left_index = first + length
                right_index = second + length
                gap_left = 0.0 if left_index == first else transcript[left_index].start_time - transcript[left_index - 1].end_time
                gap_right = 0.0 if right_index == second else transcript[right_index].start_time - transcript[right_index - 1].end_time
                if gap_left >= 6.0 or gap_right >= 6.0:
                    break

                score = parallel_line_similarity(transcript[left_index], transcript[right_index])
                if score < 0.38:
                    break
                length += 1

            if length < 4:
                continue
            first_duration = transcript[first + length - 1].end_time - transcript[first].start_time
            second_duration = transcript[second + length - 1].end_time - transcript[second].start_time
            if first_duration < 10.0 or second_duration < 10.0:
                continue
            for index in range(first, first + length):
                if roles[index] != "Bridge":
                    roles[index] = "Chorus"
            for index in range(second, second + length):
                if roles[index] != "Bridge":
                    roles[index] = "Chorus"


def semantic_similarity(left: hybrid.CandidateBlock, right: hybrid.CandidateBlock) -> float:
    left_norm = re.sub(r"^[啊呀哦喔呜嗯]+", "", hybrid.normalize_text(left.text))
    right_norm = re.sub(r"^[啊呀哦喔呜嗯]+", "", hybrid.normalize_text(right.text))
    if not left_norm or not right_norm:
        return 0.0
    if left_norm in right_norm or right_norm in left_norm:
        return 1.0
    prefix_bonus = 0.15 if left_norm[:2] == right_norm[:2] else 0.0
    return min(1.0, SequenceMatcher(None, left_norm, right_norm).ratio() + prefix_bonus)


def protect_semantic_continuity(transcript: list[hybrid.CandidateBlock], roles: list[str]) -> None:
    """Avoid splitting one semantic passage into multiple roles too early."""
    for index in range(1, len(transcript)):
        previous_role = roles[index - 1]
        current_role = roles[index]
        if previous_role == current_role:
            continue
        if {previous_role, current_role} & {"Bridge", "Intro", "Outro"}:
            continue
        similarity = semantic_similarity(transcript[index - 1], transcript[index])
        short_gap = transcript[index].start_time - transcript[index - 1].end_time <= 3.0
        if similarity >= 0.52 and short_gap:
            if current_role == "Chorus" and previous_role == "Verse":
                roles[index - 1] = "Pre-chorus"
            elif previous_role == "Chorus" and current_role == "Verse":
                roles[index] = previous_role
            else:
                roles[index] = previous_role


def make_blocks_from_sentence_roles(
    transcript: list[hybrid.CandidateBlock],
    roles: list[str],
    min_gap_bridge_seconds: float,
    max_role_block_seconds: float = 42.0,
) -> list[hybrid.CandidateBlock]:
    if not transcript:
        return []

    blocks: list[hybrid.CandidateBlock] = []
    current_start = 0
    current_role = roles[0]

    def role_text(role: str, text: str) -> str:
        if role == "Chorus":
            return f"hook repeated {text}"
        if role == "Pre-chorus":
            return f"build transition {text}"
        if role == "Outro":
            return f"outro {text}"
        return text

    def append_role_block(start_index: int, end_index: int, role: str) -> None:
        if end_index <= start_index:
            return
        items = transcript[start_index:end_index]
        text = " ".join(item.text for item in items)
        blocks.append(hybrid.CandidateBlock(items[0].start_time, items[-1].end_time, role_text(role, text)))

    for index in range(1, len(transcript)):
        gap = transcript[index].start_time - transcript[index - 1].end_time
        role_changed = roles[index] != current_role
        current_duration = transcript[index - 1].end_time - transcript[current_start].start_time
        too_long = current_duration >= max_role_block_seconds and roles[index] == current_role
        if gap >= min_gap_bridge_seconds or role_changed or too_long:
            append_role_block(current_start, index, current_role)
            if gap >= min_gap_bridge_seconds:
                blocks.append(hybrid.CandidateBlock(transcript[index - 1].end_time, transcript[index].start_time, ""))
            current_start = index
            current_role = roles[index]

    append_role_block(current_start, len(transcript), current_role)
    return blocks


def write_blocks(path: Path, blocks: list[hybrid.CandidateBlock]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start_time", "end_time", "text"])
        writer.writeheader()
        for block in blocks:
            writer.writerow(
                {
                    "start_time": hybrid.format_timestamp(block.start_time),
                    "end_time": hybrid.format_timestamp(block.end_time),
                    "text": block.text,
                }
            )


def merge_transcript_to_blocks(
    transcript: list[hybrid.CandidateBlock],
    max_merge_gap: float,
    max_block_seconds: float,
    min_gap_bridge_seconds: float,
) -> list[hybrid.CandidateBlock]:
    if not transcript:
        return []

    blocks: list[hybrid.CandidateBlock] = []
    current = hybrid.CandidateBlock(transcript[0].start_time, transcript[0].end_time, transcript[0].text)
    for item in transcript[1:]:
        gap = item.start_time - current.end_time
        current_duration = current.end_time - current.start_time
        if gap >= min_gap_bridge_seconds:
            blocks.append(current)
            blocks.append(hybrid.CandidateBlock(current.end_time, item.start_time, ""))
            current = hybrid.CandidateBlock(item.start_time, item.end_time, item.text)
        elif gap <= max_merge_gap and current_duration < max_block_seconds:
            current.end_time = item.end_time
            current.text = f"{current.text} {item.text}".strip()
        else:
            blocks.append(current)
            current = hybrid.CandidateBlock(item.start_time, item.end_time, item.text)
    blocks.append(current)
    return blocks


def infer_song_blocks(
    transcript: list[hybrid.CandidateBlock],
    title: str,
    min_gap_bridge_seconds: float,
) -> list[hybrid.CandidateBlock]:
    """Create section-scale lyric blocks from Whisper line timestamps."""
    if not transcript:
        return []

    roles = sentence_roles_from_repetition(transcript, title, min_gap_bridge_seconds)
    return make_blocks_from_sentence_roles(transcript, roles, min_gap_bridge_seconds)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mp3 -> Whisper -> hybrid structure segmentation.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--work-dir", default="outputs/whisper_hybrid")
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug-output")
    parser.add_argument("--compare-manual", action="store_true")
    parser.add_argument("--max-merge-gap", type=float, default=4.0)
    parser.add_argument("--max-block-seconds", type=float, default=32.0)
    parser.add_argument("--min-gap-bridge-seconds", type=float, default=8.0)
    args = parser.parse_args()

    audio = Path(args.audio)
    work_dir = Path(args.work_dir)
    stem = audio.stem
    transcript_csv = work_dir / f"{stem}_whisper_segments.csv"
    blocks_csv = work_dir / f"{stem}_auto_lyric_blocks.csv"

    run_transcription(audio, transcript_csv, args.model, args.language)
    transcript = read_transcript(transcript_csv)
    blocks = infer_song_blocks(transcript, args.title, args.min_gap_bridge_seconds)
    if not blocks:
        blocks = merge_transcript_to_blocks(
            transcript,
            max_merge_gap=args.max_merge_gap,
            max_block_seconds=args.max_block_seconds,
            min_gap_bridge_seconds=args.min_gap_bridge_seconds,
        )
    write_blocks(blocks_csv, blocks)

    features = hybrid.load_audio_features(audio)
    sections, debug_rows = hybrid.make_sections(blocks, features, args.title, args.min_gap_bridge_seconds)
    hybrid.write_sections(Path(args.output), sections)
    if args.debug_output:
        hybrid.write_debug(Path(args.debug_output), debug_rows)

    print(f"Transcript: {transcript_csv}")
    print(f"Auto lyric blocks: {blocks_csv}")
    print(f"Sections: {args.output}")
    for section in sections:
        print(
            f"{section.section_id:02d} "
            f"{hybrid.format_timestamp(section.start_time)}-{hybrid.format_timestamp(section.end_time)} "
            f"{hybrid.format_timestamp(section.duration)} "
            f"{section.label} {section.structure_label}"
        )


if __name__ == "__main__":
    main()
