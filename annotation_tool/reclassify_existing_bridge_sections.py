#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import urllib.parse
from collections import defaultdict
from pathlib import Path

from import_local_results import get, update


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "outputs" / "functional_sections" / "bridge_interlude_reclassification_report.csv"

NO_VOCAL_RE = re.compile(r"无歌词|无完整歌词|主唱退出|低人声|弱主唱|无人声|器乐|间奏|instrumental|interlude", re.I)
CONTRAST_RE = re.compile(r"转折|对比|新材料|重新蓄力|情绪变化|新和声|新旋律|middle\s*8|Middle\s*8", re.I)
FILLER_TEXT = {"", "前奏", "尾奏", "间奏", "instrumental", "interlude"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reclassify existing bridge sections into bridge/interlude without rerunning the full pipeline."
    )
    parser.add_argument("--apply", action="store_true", help="Write updates to Supabase. Default is dry-run.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="CSV report path.")
    parser.add_argument(
        "--include-human",
        action="store_true",
        default=True,
        help="Also update linked human_annotations for changed machine sections.",
    )
    args = parser.parse_args()

    songs = {row["id"]: row for row in paged_get("songs", "select=id,title,artist,status&order=title.asc")}
    latest_runs = latest_run_ids()
    auto_sections = paged_get(
        "auto_sections",
        "select=id,song_id,run_id,section_index,start_time_sec,end_time_sec,section_type,structure_label,"
        "lyric_evidence,vocal_evidence,acoustic_evidence,need_human_review,type_confidence,boundary_confidence"
        "&section_type=eq.bridge",
    )
    latest_bridge_sections = [row for row in auto_sections if row.get("run_id") == latest_runs.get(row.get("song_id"))]
    human_annotations = paged_get(
        "human_annotations",
        "select=id,song_id,source_auto_section_id,section_index,start_time_sec,end_time_sec,section_type,comment,updated_at"
        "&section_type=eq.bridge",
    )
    lyric_rows = paged_get(
        "lyric_lines",
        "select=song_id,line_index,start_time_sec,end_time_sec,text&order=song_id.asc,line_index.asc",
    )
    lyrics_by_song: dict[str, list[dict]] = defaultdict(list)
    for row in lyric_rows:
        lyrics_by_song[row["song_id"]].append(row)

    decisions: list[dict] = []
    changed_auto_ids: set[str] = set()
    for section in latest_bridge_sections:
        decision = classify_section(section, lyrics_by_song.get(section["song_id"], []), songs)
        decisions.append(decision)
        if decision["new_type"] == "interlude":
            changed_auto_ids.add(section["id"])

    human_decisions: list[dict] = []
    for annotation in human_annotations:
        linked_changed = annotation.get("source_auto_section_id") in changed_auto_ids
        if linked_changed:
            decision = classify_human_annotation(annotation, lyrics_by_song.get(annotation["song_id"], []), songs, linked_changed=True)
            human_decisions.append(decision)
            continue
        if not annotation.get("source_auto_section_id"):
            decision = classify_human_annotation(annotation, lyrics_by_song.get(annotation["song_id"], []), songs, linked_changed=False)
            if decision["new_type"] == "interlude":
                human_decisions.append(decision)

    write_report(Path(args.report), decisions, human_decisions)

    auto_changes = [row for row in decisions if row["new_type"] == "interlude"]
    human_changes = [row for row in human_decisions if row["new_type"] == "interlude"]
    print(f"latest auto bridge sections checked: {len(decisions)}")
    print(f"auto bridge -> interlude candidates: {len(auto_changes)}")
    print(f"human bridge -> interlude candidates: {len(human_changes)}")
    print(f"report: {args.report}")

    if not args.apply:
        print("dry-run only; rerun with --apply to update Supabase")
        return

    for row in auto_changes:
        payload = {
            "section_type": "interlude",
            "structure_label": "I",
            "lyric_evidence": append_migration_note(row["lyric_evidence"], row["reason"]),
            "need_human_review": True,
        }
        update("auto_sections", payload, {"id": f"eq.{row['section_id']}"})

    if args.include_human:
        for row in human_changes:
            payload = {
                "section_type": "interlude",
                "comment": append_migration_note(row["comment"], row["reason"]),
            }
            update("human_annotations", payload, {"id": f"eq.{row['section_id']}"})

    print(f"updated auto_sections: {len(auto_changes)}")
    print(f"updated human_annotations: {len(human_changes) if args.include_human else 0}")


def latest_run_ids() -> dict[str, str]:
    rows = paged_get("model_runs", "select=id,song_id,created_at&order=song_id.asc,created_at.desc")
    latest: dict[str, str] = {}
    for row in rows:
        latest.setdefault(row["song_id"], row["id"])
    return latest


def paged_get(table: str, query: str, page_size: int = 1000) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    separator = "&" if query else ""
    while True:
        page = get(table, f"{query}{separator}limit={page_size}&offset={offset}")
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size


def classify_section(section: dict, lyrics: list[dict], songs: dict[str, dict]) -> dict:
    evidence = joined_evidence(section)
    lyric_stats = overlapping_lyric_stats(section, lyrics)
    has_contrast = bool(CONTRAST_RE.search(evidence))
    no_vocal_evidence = bool(NO_VOCAL_RE.search(evidence))
    has_real_lyrics = lyric_stats["real_line_count"] > 0 and lyric_stats["real_char_count"] >= 6
    short_transition = duration(section) <= 18 and lyric_stats["real_line_count"] <= 1

    if has_contrast and has_real_lyrics:
        new_type = "bridge"
        reason = "kept_bridge_contrast_with_lyrics"
    elif no_vocal_evidence and not has_contrast:
        new_type = "interlude"
        reason = "no_vocal_or_instrumental_evidence"
    elif not has_real_lyrics:
        new_type = "interlude"
        reason = "no_effective_lyric_lines_in_section"
    elif short_transition and no_vocal_evidence:
        new_type = "interlude"
        reason = "short_transition_with_low_vocal_evidence"
    else:
        new_type = "bridge"
        reason = "kept_bridge_lyrical_or_uncertain"

    song = songs.get(section["song_id"], {})
    return {
        "table": "auto_sections",
        "song_title": song.get("title", ""),
        "song_artist": song.get("artist", ""),
        "song_id": section["song_id"],
        "section_id": section["id"],
        "section_index": section.get("section_index", ""),
        "start_time_sec": section.get("start_time_sec", ""),
        "end_time_sec": section.get("end_time_sec", ""),
        "duration_sec": round(duration(section), 3),
        "old_type": section.get("section_type", ""),
        "new_type": new_type,
        "reason": reason,
        "real_line_count": lyric_stats["real_line_count"],
        "real_char_count": lyric_stats["real_char_count"],
        "lyric_evidence": section.get("lyric_evidence") or "",
        "vocal_evidence": section.get("vocal_evidence") or "",
        "acoustic_evidence": section.get("acoustic_evidence") or "",
        "comment": "",
    }


def classify_human_annotation(annotation: dict, lyrics: list[dict], songs: dict[str, dict], *, linked_changed: bool) -> dict:
    fake_section = {
        "start_time_sec": annotation.get("start_time_sec"),
        "end_time_sec": annotation.get("end_time_sec"),
    }
    lyric_stats = overlapping_lyric_stats(fake_section, lyrics)
    comment = annotation.get("comment") or ""
    has_contrast = bool(CONTRAST_RE.search(comment))
    no_vocal_evidence = bool(NO_VOCAL_RE.search(comment))
    if linked_changed:
        new_type = "interlude"
        reason = "linked_auto_section_reclassified"
    elif no_vocal_evidence and not has_contrast:
        new_type = "interlude"
        reason = "manual_comment_indicates_interlude"
    elif lyric_stats["real_line_count"] == 0 and not has_contrast:
        new_type = "interlude"
        reason = "manual_bridge_has_no_effective_lyrics"
    else:
        new_type = "bridge"
        reason = "kept_manual_bridge"

    song = songs.get(annotation["song_id"], {})
    return {
        "table": "human_annotations",
        "song_title": song.get("title", ""),
        "song_artist": song.get("artist", ""),
        "song_id": annotation["song_id"],
        "section_id": annotation["id"],
        "section_index": annotation.get("section_index", ""),
        "start_time_sec": annotation.get("start_time_sec", ""),
        "end_time_sec": annotation.get("end_time_sec", ""),
        "duration_sec": round(duration(annotation), 3),
        "old_type": annotation.get("section_type", ""),
        "new_type": new_type,
        "reason": reason,
        "real_line_count": lyric_stats["real_line_count"],
        "real_char_count": lyric_stats["real_char_count"],
        "lyric_evidence": "",
        "vocal_evidence": "",
        "acoustic_evidence": "",
        "comment": comment,
    }


def overlapping_lyric_stats(section: dict, lyrics: list[dict]) -> dict[str, int]:
    start = float(section.get("start_time_sec") or 0)
    end = float(section.get("end_time_sec") or start)
    real_lines = []
    for line in lyrics:
        line_start = float(line.get("start_time_sec") or 0)
        line_end = float(line.get("end_time_sec") or line_start)
        if line_end <= start or line_start >= end:
            continue
        text = normalize_text(line.get("text") or "")
        if is_real_lyric(text):
            real_lines.append(text)
    return {
        "real_line_count": len(real_lines),
        "real_char_count": sum(len(re.findall(r"[\u4e00-\u9fffA-Za-z]", text)) for text in real_lines),
    }


def is_real_lyric(text: str) -> bool:
    if text.lower() in FILLER_TEXT:
        return False
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", text):
        return False
    return True


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def joined_evidence(section: dict) -> str:
    return " ".join(
        str(section.get(field) or "")
        for field in ("lyric_evidence", "vocal_evidence", "acoustic_evidence")
    )


def duration(section: dict) -> float:
    return max(0.0, float(section.get("end_time_sec") or 0) - float(section.get("start_time_sec") or 0))


def append_migration_note(value: str, reason: str) -> str:
    note = f"bridge/interlude taxonomy migration: {reason}"
    value = value or ""
    if note in value:
        return value
    return f"{value}；{note}".strip("；")


def write_report(path: Path, auto_rows: list[dict], human_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "table",
        "song_title",
        "song_artist",
        "song_id",
        "section_id",
        "section_index",
        "start_time_sec",
        "end_time_sec",
        "duration_sec",
        "old_type",
        "new_type",
        "reason",
        "real_line_count",
        "real_char_count",
        "lyric_evidence",
        "vocal_evidence",
        "acoustic_evidence",
        "comment",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(auto_rows)
        writer.writerows(human_rows)


if __name__ == "__main__":
    main()
