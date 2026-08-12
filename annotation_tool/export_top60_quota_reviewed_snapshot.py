#!/usr/bin/env python3
"""Export the QQ top-60 quota reviewed corpus from Supabase.

The database keeps historical reviewed songs. This exporter freezes only the
100-song corpus named in qq_music_top60_song_quota.csv.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import time
import unicodedata
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPABASE_URL = "https://itcarzyhlkiovskfdqut.supabase.co"
SUPABASE_KEY = "sb_publishable_IG4iiZuIDAoxhzZSPZnRRA_xSl0VRfN"
CHINA_TZ = timezone(timedelta(hours=8))
DATE_TAG = datetime.now(CHINA_TZ).strftime("%Y%m%d")


def supabase_get(path: str, attempts: int = 20) -> list[dict[str, Any]]:
    last_error = ""
    headers = [
        "-H",
        f"apikey: {SUPABASE_KEY}",
        "-H",
        f"Authorization: Bearer {SUPABASE_KEY}",
    ]
    for attempt in range(attempts):
        completed = subprocess.run(
            [
                "curl",
                "--ipv4",
                "-sS",
                "--connect-timeout",
                "5",
                "--max-time",
                "15",
                *headers,
                f"{SUPABASE_URL}{path}",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            return json.loads(completed.stdout or "[]")
        last_error = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else str(completed.returncode)
        time.sleep(min(10, 1 + attempt // 2))
    raise RuntimeError(f"Supabase GET failed for {path}: {last_error}")


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"^\s*\d+[\.\u3001_-]*\s*", "", text)
    for old, new in [
        ("（", "("),
        ("）", ")"),
        ("，", ","),
        ("、", ","),
        ("_", "-"),
    ]:
        text = text.replace(old, new)
    text = re.sub(r"\.(mp3|flac|wav|m4a)$", "", text)
    return re.sub(r"\s+", "", text)


def quota_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_order = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if not str(row.get("序号") or "").isdigit():
                continue
            for item in [part.strip() for part in str(row.get("现有歌曲") or "").split("；") if part.strip()]:
                selected_order += 1
                raw = re.sub(r"^\s*\d+[\.\u3001_-]*\s*", "", item)
                raw = re.sub(r"\.(mp3|flac|wav|m4a)$", "", raw, flags=re.I)
                parts = re.split(r"[-_]", raw, maxsplit=1)
                probes = [raw]
                if len(parts) == 2:
                    probes.extend(parts)
                rows.append(
                    {
                        "selected_order": selected_order,
                        "rank": int(row["序号"]),
                        "quota_artist": row["歌手名"],
                        "quota_item": item,
                        "probes": probes,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    songs: list[dict[str, Any]] = []
    select = ",".join(
        [
            "id",
            "title",
            "artist",
            "audio_path",
            "status",
            "duration_sec",
            "duration_label",
            "source",
            "created_at",
            "updated_at",
        ]
    )
    page_size = 50
    for offset in range(0, 500, page_size):
        page = supabase_get(f"/rest/v1/songs?select={select}&order=id.asc&limit={page_size}&offset={offset}")
        songs.extend(page)
        if len(page) < page_size:
            break

    by_audio: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_combo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for song in songs:
        by_audio[normalize(Path(str(song.get("audio_path") or "")).stem)].append(song)
        by_combo[normalize(f"{song.get('title', '')}-{song.get('artist', '')}")].append(song)
        by_combo[normalize(f"{song.get('artist', '')}-{song.get('title', '')}")].append(song)
        by_title[normalize(song.get("title"))].append(song)

    selected: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for quota in quota_rows(PROJECT_ROOT / "qq_music_top60_song_quota.csv"):
        candidates: list[dict[str, Any]] = []
        match_method = ""
        key = normalize(quota["quota_item"])
        if by_audio.get(key):
            candidates = by_audio[key]
            match_method = "audio_path"
        elif by_combo.get(key):
            candidates = by_combo[key]
            match_method = "title_artist"
        else:
            for probe in quota["probes"]:
                hits = by_title.get(normalize(probe), [])
                if hits:
                    candidates = hits
                    match_method = "title_probe"
                    break
        if not candidates:
            unmatched.append(quota)
            continue
        candidates = sorted(
            candidates,
            key=lambda song: (
                song.get("status") != "reviewed",
                "old_version" in str(song.get("audio_path") or ""),
                str(song.get("id") or ""),
            ),
        )
        song = candidates[0]
        selected.append(
            {
                "selected_order": quota["selected_order"],
                "rank": quota["rank"],
                "quota_artist": quota["quota_artist"],
                "quota_item": quota["quota_item"],
                "song_id": song.get("id"),
                "title": song.get("title"),
                "artist": song.get("artist"),
                "audio_path": song.get("audio_path"),
                "local_audio_path": f"songs/{Path(str(song.get('audio_path') or '')).name}",
                "audio_exists": (PROJECT_ROOT / "songs" / Path(str(song.get("audio_path") or "")).name).is_file(),
                "duration_sec": song.get("duration_sec"),
                "duration_label": song.get("duration_label"),
                "source": song.get("source"),
                "status": song.get("status"),
                "created_at": song.get("created_at"),
                "updated_at": song.get("updated_at"),
                "match_method": match_method,
            }
        )

    duplicate_ids = [song_id for song_id, count in Counter(row["song_id"] for row in selected).items() if count > 1]
    not_reviewed = [row for row in selected if row["status"] != "reviewed"]
    if unmatched or duplicate_ids or not_reviewed or len(selected) != 100:
        raise SystemExit(
            json.dumps(
                {
                    "selected": len(selected),
                    "unmatched": unmatched,
                    "duplicate_ids": duplicate_ids,
                    "not_reviewed": not_reviewed,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    ids = [str(row["song_id"]) for row in selected]
    annotations: list[dict[str, Any]] = []
    select_annotations = ",".join(
        [
            "song_id",
            "song_title",
            "song_artist",
            "section_order",
            "section_type",
            "start_time_sec",
            "end_time_sec",
            "duration_sec",
            "song_reviewed_at",
            "song_status",
            "original_section_index",
            "annotation_kind",
            "is_approved",
            "confidence",
            "annotator",
            "comment",
            "source_run_id",
            "source_auto_section_id",
            "human_annotation_id",
            "created_at",
            "updated_at",
        ]
    )
    for start in range(0, len(ids), 20):
        chunk = ids[start : start + 20]
        ors = ",".join(f"song_id.eq.{urllib.parse.quote(song_id, safe='')}" for song_id in chunk)
        annotations.extend(
            supabase_get(
                f"/rest/v1/human_annotations_readable?select={select_annotations}&or=({ors})&order=song_id.asc,section_order.asc"
            )
        )

    annotation_counts = Counter(str(row["song_id"]) for row in annotations)
    missing_annotations = [song_id for song_id in ids if annotation_counts[song_id] == 0]
    if missing_annotations:
        raise SystemExit(f"Missing readable annotations for: {missing_annotations}")
    for row in selected:
        row["readable_section_count"] = annotation_counts[str(row["song_id"])]

    manifest_dir = PROJECT_ROOT / "research/phase2_section_aware_va/data/manifests"
    raw_dir = manifest_dir / "raw"
    raw_songs = raw_dir / f"reviewed_songs_top60_quota_100_{DATE_TAG}.json"
    raw_annotations = raw_dir / f"human_annotations_readable_top60_quota_100_{DATE_TAG}.json"
    manifest = manifest_dir / f"top60_quota_100_reviewed_manifest_{DATE_TAG}.csv"
    annotation_csv = manifest_dir / f"top60_quota_100_human_annotations_readable_{DATE_TAG}.csv"
    audit = manifest_dir / f"top60_quota_100_audit_{DATE_TAG}.json"

    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_songs.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw_annotations.write_text(json.dumps(annotations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(manifest, selected, list(selected[0].keys()))
    write_csv(annotation_csv, annotations, list(annotations[0].keys()))

    summary = {
        "created_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "rule": "QQ音乐歌手榜前60；1-10每人3首，11-30每人2首，31-60每人1首；合唱歌按补缺优先分配",
        "target_artists": 60,
        "target_songs": 100,
        "matched_songs": len(selected),
        "unique_song_ids": len(set(ids)),
        "readable_annotation_songs": len(annotation_counts),
        "readable_annotation_rows": len(annotations),
        "database_total_songs": len(songs),
        "database_total_reviewed": sum(1 for song in songs if song.get("status") == "reviewed"),
        "local_audio_missing": [row for row in selected if not row["audio_exists"]],
        "files": {
            str(raw_songs.relative_to(PROJECT_ROOT)): sha256(raw_songs),
            str(raw_annotations.relative_to(PROJECT_ROOT)): sha256(raw_annotations),
            str(manifest.relative_to(PROJECT_ROOT)): sha256(manifest),
            str(annotation_csv.relative_to(PROJECT_ROOT)): sha256(annotation_csv),
        },
    }
    audit.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
