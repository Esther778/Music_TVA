#!/usr/bin/env python3
"""Build 18 supplemental model-free candidates to expand each VA anchor set to five."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from build_va_anchor_candidates import (
    BOUNDARIES_PATH,
    BLIND_SAMPLE_PATH,
    PROJECT_ROOT,
    SONGS_PATH,
    TEMPLATE_PATH,
    acoustic_descriptors,
    probe_duration,
    resolve_audio_path,
    sha256,
)


PHASE_ROOT = PROJECT_ROOT / "research/phase2_section_aware_va"
EXISTING_ANCHORS = (
    PHASE_ROOT
    / "outputs/human_validation/calibration_candidates/frozen_va_anchors_v1.json"
)
OUTPUT_DIR = PHASE_ROOT / "outputs/human_validation/calibration_candidates_v2_supplement"
CLIP_DIR = OUTPUT_DIR / "audio_clips"
MANIFEST_CSV = OUTPUT_DIR / "va_anchor_supplement_candidates.csv"
MANIFEST_JSON = OUTPUT_DIR / "va_anchor_supplement_candidates_manifest.json"
OUTPUT_HTML = OUTPUT_DIR / "va_anchor_supplement_candidate_review.html"


SUPPLEMENT_CANDIDATES = [
    {
        "candidate_id": "V_LOW_03",
        "dimension": "valence",
        "proposed_level": "low",
        "song_id": "local_08a3a00d94cc",
        "start_sec": 58.5,
        "end_sec": 77.5,
        "selection_basis": "消极Valence候选；人工质检边界内的代表性演唱段",
    },
    {
        "candidate_id": "V_LOW_04",
        "dimension": "valence",
        "proposed_level": "low",
        "song_id": "local_9a87bda83802",
        "start_sec": 37.0,
        "end_sec": 57.0,
        "selection_basis": "消极Valence候选；人工质检边界内的代表性演唱段",
    },
    {
        "candidate_id": "V_LOW_05",
        "dimension": "valence",
        "proposed_level": "low",
        "song_id": "local_b5ba16fe7c29",
        "start_sec": 58.0,
        "end_sec": 78.0,
        "selection_basis": "消极Valence候选；人工质检边界内的代表性演唱段",
    },
    {
        "candidate_id": "V_MID_03",
        "dimension": "valence",
        "proposed_level": "mid",
        "song_id": "local_00db930e5d96",
        "start_sec": 61.0,
        "end_sec": 81.0,
        "selection_basis": "中性或正负混合Valence候选；完整稳定的演唱片段",
    },
    {
        "candidate_id": "V_MID_04",
        "dimension": "valence",
        "proposed_level": "mid",
        "song_id": "local_8b7603c2a838",
        "start_sec": 199.5,
        "end_sec": 219.5,
        "selection_basis": "中性或正负混合Valence候选；人工质检边界内的代表段",
    },
    {
        "candidate_id": "V_MID_05",
        "dimension": "valence",
        "proposed_level": "mid",
        "song_id": "local_b44c1555a13a",
        "start_sec": 49.5,
        "end_sec": 69.5,
        "selection_basis": "中性或正负混合Valence候选；人工质检边界内的代表段",
    },
    {
        "candidate_id": "V_HIGH_03",
        "dimension": "valence",
        "proposed_level": "high",
        "song_id": "local_087084e3d85c",
        "start_sec": 64.5,
        "end_sec": 84.5,
        "selection_basis": "积极Valence候选；人工质检边界内的代表性演唱段",
    },
    {
        "candidate_id": "V_HIGH_04",
        "dimension": "valence",
        "proposed_level": "high",
        "song_id": "local_b94d86b05e5f",
        "start_sec": 41.0,
        "end_sec": 57.0,
        "selection_basis": "积极Valence候选；人工质检边界内的代表性演唱段",
    },
    {
        "candidate_id": "V_HIGH_05",
        "dimension": "valence",
        "proposed_level": "high",
        "song_id": "local_c3684d3070cf",
        "start_sec": 60.5,
        "end_sec": 80.5,
        "selection_basis": "积极Valence候选；人工质检边界内的代表性演唱段",
    },
    {
        "candidate_id": "A_LOW_03",
        "dimension": "arousal",
        "proposed_level": "low",
        "song_id": "local_5e1483908235",
        "start_sec": 37.0,
        "end_sec": 57.0,
        "selection_basis": "低唤醒候选；平静、低激活且非静音的演唱段",
    },
    {
        "candidate_id": "A_LOW_04",
        "dimension": "arousal",
        "proposed_level": "low",
        "song_id": "local_0953f10adcc5",
        "start_sec": 43.0,
        "end_sec": 63.0,
        "selection_basis": "低唤醒候选；低能量且非静音的完整演唱片段",
    },
    {
        "candidate_id": "A_LOW_05",
        "dimension": "arousal",
        "proposed_level": "low",
        "song_id": "i_only_care_about_you",
        "start_sec": 57.0,
        "end_sec": 77.0,
        "selection_basis": "低唤醒候选；节奏舒缓、激活程度较低的演唱段",
    },
    {
        "candidate_id": "A_MID_03",
        "dimension": "arousal",
        "proposed_level": "mid",
        "song_id": "local_93f7de29c0da",
        "start_sec": 80.5,
        "end_sec": 100.5,
        "selection_basis": "中等唤醒候选；节奏与能量适中且持续稳定",
    },
    {
        "candidate_id": "A_MID_04",
        "dimension": "arousal",
        "proposed_level": "mid",
        "song_id": "local_8a09e5ac1557",
        "start_sec": 73.5,
        "end_sec": 93.5,
        "selection_basis": "中等唤醒候选；人工质检边界内的稳定演唱段",
    },
    {
        "candidate_id": "A_MID_05",
        "dimension": "arousal",
        "proposed_level": "mid",
        "song_id": "local_42f647302ce2",
        "start_sec": 105.0,
        "end_sec": 125.0,
        "selection_basis": "中等唤醒候选；人工质检边界内的稳定演唱段",
    },
    {
        "candidate_id": "A_HIGH_03",
        "dimension": "arousal",
        "proposed_level": "high",
        "song_id": "local_bf35bb3d23d8",
        "start_sec": 36.0,
        "end_sec": 51.0,
        "selection_basis": "高唤醒候选；快速、强节奏且高激活的演唱段",
    },
    {
        "candidate_id": "A_HIGH_04",
        "dimension": "arousal",
        "proposed_level": "high",
        "song_id": "local_52fb89921d0f",
        "start_sec": 153.5,
        "end_sec": 173.5,
        "selection_basis": "高唤醒候选；强力度、高激活的代表性演唱段",
    },
    {
        "candidate_id": "A_HIGH_05",
        "dimension": "arousal",
        "proposed_level": "high",
        "song_id": "local_eb95c3935b89",
        "start_sec": 199.5,
        "end_sec": 219.5,
        "selection_basis": "高唤醒候选；强力度、高激活的代表性演唱段",
    },
]


def load_blind_ids() -> set[str]:
    with BLIND_SAMPLE_PATH.open(newline="", encoding="utf-8-sig") as handle:
        return {row["song_id"] for row in csv.DictReader(handle)}


def main() -> None:
    songs = json.loads(SONGS_PATH.read_text(encoding="utf-8"))
    boundaries = json.loads(BOUNDARIES_PATH.read_text(encoding="utf-8"))
    existing = json.loads(EXISTING_ANCHORS.read_text(encoding="utf-8"))["anchors"]
    songs_by_id = {row["id"]: row for row in songs}
    blind_ids = load_blind_ids()
    existing_song_ids = {str(row["song_id"]) for row in existing}
    new_song_ids = {str(row["song_id"]) for row in SUPPLEMENT_CANDIDATES}
    if len(SUPPLEMENT_CANDIDATES) != 18 or len(new_song_ids) != 18:
        raise RuntimeError("Expected 18 candidates from 18 unique songs")
    if blind_ids & new_song_ids:
        raise RuntimeError("Supplement candidates overlap the blind sample")
    if existing_song_ids & new_song_ids:
        raise RuntimeError("Supplement candidates repeat existing anchor songs")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for candidate in SUPPLEMENT_CANDIDATES:
        song = songs_by_id.get(str(candidate["song_id"]))
        if song is None:
            raise RuntimeError(f"Unknown song: {candidate['song_id']}")
        start_sec = float(candidate["start_sec"])
        end_sec = float(candidate["end_sec"])
        containing = [
            row
            for row in boundaries
            if row["song_id"] == candidate["song_id"]
            and float(row["start_time_sec"]) <= start_sec + 1e-6
            and float(row["end_time_sec"]) >= end_sec - 1e-6
        ]
        if len(containing) != 1:
            raise RuntimeError(
                f"{candidate['candidate_id']} must be inside one reviewed boundary; "
                f"found {len(containing)}"
            )
        source = resolve_audio_path(song["audio_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        duration = end_sec - start_sec
        clip_name = f"{str(candidate['candidate_id']).lower()}_{candidate['song_id']}.mp3"
        clip_path = CLIP_DIR / clip_name
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start_sec:.3f}",
                "-i",
                str(source),
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:a:0",
                "-vn",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "128k",
                "-ar",
                "44100",
                "-ac",
                "2",
                str(clip_path),
            ],
            check=True,
        )
        measured_duration = probe_duration(clip_path)
        duration_error = measured_duration - duration
        if abs(duration_error) > 0.08:
            failures.append(
                f"{candidate['candidate_id']}: duration error {duration_error:.6f}s"
            )
        descriptors = acoustic_descriptors(clip_path)
        rows.append(
            {
                **candidate,
                "title": song["title"],
                "artist": song["artist"],
                "section_order": int(containing[0]["section_order"]),
                "duration_sec": round(duration, 3),
                "measured_duration_sec": round(measured_duration, 6),
                "duration_error_sec": round(duration_error, 6),
                "source_audio": str(source.relative_to(PROJECT_ROOT)),
                "clip_relative_path": f"audio_clips/{clip_name}",
                "clip_sha256": sha256(clip_path),
                **{key: round(value, 6) for key, value in descriptors.items()},
                "status": "pass" if abs(duration_error) <= 0.08 else "fail",
            }
        )

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "<title>VA 参照片段候选确认</title>": "<title>VA 参照片段补充候选确认</title>",
        "<h1>VA 参照片段候选确认</h1>": "<h1>VA 参照片段补充候选确认</h1>",
        "12 个候选片段 · 来自正式 15 首盲测样本之外 · 不包含 Teacher 预测": (
            "18 个补充候选片段 · 每个等级新增 3 个 · 不包含 Teacher 预测"
        ),
        "当前只是候选审核，不是重新标注。": "当前只是补充候选审核，不是重新标注。",
        'const STORAGE_KEY = "va_anchor_candidate_review_v1";': (
            'const STORAGE_KEY = "va_anchor_supplement_candidate_review_v2";'
        ),
        'link.download = "va_anchor_candidate_review.csv";': (
            'link.download = "va_anchor_supplement_candidate_review.csv";'
        ),
    }
    for old, new in replacements.items():
        if template.count(old) != 1:
            raise RuntimeError(f"Expected one template marker: {old}")
        template = template.replace(old, new)
    if template.count("__DATA_JSON__") != 1:
        raise RuntimeError("Template must contain one data placeholder")
    html = template.replace(
        "__DATA_JSON__",
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"),
    )
    OUTPUT_HTML.write_text(html, encoding="utf-8")

    created_at = datetime.now(timezone.utc).isoformat()
    counts = {
        f"{dimension}_{level}": sum(
            row["dimension"] == dimension and row["proposed_level"] == level
            for row in rows
        )
        for dimension in ("valence", "arousal")
        for level in ("low", "mid", "high")
    }
    manifest = {
        "created_at_utc": created_at,
        "purpose": "Human review of supplemental anchors to reach five per VA level",
        "status": "supplement_candidate_only_not_frozen",
        "existing_anchor_count": len(existing),
        "supplement_candidate_count": len(rows),
        "projected_anchor_count_after_acceptance": len(existing) + len(rows),
        "candidates_per_dimension_level": counts,
        "unique_new_songs": len(new_song_ids),
        "blind_test_overlap_count": len(blind_ids & new_song_ids),
        "existing_anchor_song_overlap_count": len(existing_song_ids & new_song_ids),
        "teacher_predictions_used": False,
        "reviewed_boundaries_used": True,
        "failures": failures,
        "csv_path": str(MANIFEST_CSV),
        "html_path": str(OUTPUT_HTML),
    }
    MANIFEST_JSON.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError("Supplement candidate audit failed")


if __name__ == "__main__":
    main()
