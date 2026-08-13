#!/usr/bin/env python3
"""Upload top60 quota 100-song pseudo-VA outputs into dedicated Supabase tables."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "research/phase2_section_aware_va"
DATA = PHASE2 / "data/top60_quota_100_20260813"
STAGE1 = DATA / "stage1"
STAGE2 = DATA / "stage2"
STAGE3 = DATA / "stage3"
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://itcarzyhlkiovskfdqut.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_IG4iiZuIDAoxhzZSPZnRRA_xSl0VRfN")
RUN_LABEL = "top60_quota_100_teacher_v1_20260814"
DATASET_VERSION = "qq_music_top60_100_20260813"
CHINA_TZ = timezone(timedelta(hours=8))


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def postgrest(method: str, path: str, payload: Any | None = None, prefer: str = "return=representation") -> Any:
    headers = [
        "-H", f"apikey: {SUPABASE_KEY}",
        "-H", f"Authorization: Bearer {SUPABASE_KEY}",
        "-H", "Content-Type: application/json",
        "-H", f"Prefer: {prefer}",
    ]
    cmd = [
        "curl", "--ipv4", "-sS", "--connect-timeout", "10", "--max-time", "120",
        "-X", method, *headers,
    ]
    if payload is not None:
        cmd.extend(["--data-binary", json.dumps(payload, ensure_ascii=False)])
    cmd.append(f"{SUPABASE_URL}{path}")
    last = ""
    for attempt in range(5):
        completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        if completed.returncode == 0:
            text = completed.stdout.strip()
            if not text:
                return None
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid Supabase JSON response for {path}: {text[:500]}") from exc
            if isinstance(value, dict) and value.get("code"):
                raise RuntimeError(f"Supabase error for {path}: {value}")
            return value
        last = completed.stderr.strip()
        time.sleep(2 + attempt)
    raise RuntimeError(f"Supabase request failed for {path}: {last}")


def insert_batches(table: str, rows: list[dict[str, Any]], batch_size: int = 1000) -> None:
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        postgrest("POST", f"/rest/v1/{table}", batch, prefer="return=minimal")
        print(f"inserted {table} {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)


def main() -> None:
    stage1_protocol = json.loads((STAGE1 / "top60_quota_100_protocol.json").read_text(encoding="utf-8"))
    inference_manifest = json.loads((STAGE2 / "top60_quota_100_inference_manifest.json").read_text(encoding="utf-8"))
    frame_audit = json.loads((STAGE2 / "top60_quota_100_output_audit_summary.json").read_text(encoding="utf-8"))
    alignment_manifest = json.loads((STAGE3 / "top60_quota_100_alignment_manifest.json").read_text(encoding="utf-8"))
    if frame_audit.get("overall_status") != "pass":
        raise RuntimeError("Frame audit is not pass")
    if alignment_manifest.get("status") != "pass":
        raise RuntimeError("Alignment audit is not pass")

    existing = postgrest(
        "GET",
        f"/rest/v1/paper_top60_quota_100_va_runs?select=id&dataset_version=eq.{DATASET_VERSION}&run_label=eq.{RUN_LABEL}",
    )
    for row in existing or []:
        postgrest("DELETE", f"/rest/v1/paper_top60_quota_100_va_runs?id=eq.{row['id']}", prefer="return=minimal")
    run_payload = [{
        "dataset_version": DATASET_VERSION,
        "run_label": RUN_LABEL,
        "status": "running",
        "songs": int(inference_manifest["songs"]),
        "sections": int(alignment_manifest["sections"]),
        "frames": int(inference_manifest["frames"]),
        "teacher_valence_model_id": inference_manifest["model_ids"]["valence"],
        "teacher_arousal_model_id": inference_manifest["model_ids"]["arousal"],
        "feature_backbone": inference_manifest["feature_backbone"],
        "frame_hop_sec": inference_manifest["frame_hop_sec"],
        "teacher_window_sec": inference_manifest["teacher_window_sec"],
        "teacher_stride_sec": inference_manifest["teacher_stride_sec"],
        "source_manifest_sha256": stage1_protocol["manifest_sha256"],
        "frame_va_sha256": inference_manifest["combined_output_sha256"],
        "section_features_sha256": alignment_manifest["section_features_sha256"],
        "song_features_sha256": alignment_manifest["song_features_sha256"],
        "manifest": {
            "stage1": stage1_protocol,
            "stage2": inference_manifest,
            "frame_audit": frame_audit,
            "stage3": alignment_manifest,
        },
    }]
    run = postgrest("POST", "/rest/v1/paper_top60_quota_100_va_runs", run_payload)[0]
    run_id = run["id"]
    print(f"created run {run_id}", flush=True)

    frames = pd.read_csv(STAGE2 / "top60_quota_100_frame_va.csv", dtype={"song_id": str})
    frames["frame_index"] = frames.groupby("song_id").cumcount() + 1
    frame_rows = []
    for row in frames.itertuples(index=False):
        frame_rows.append({
            "run_id": run_id,
            "song_id": row.song_id,
            "frame_index": int(row.frame_index),
            "time_sec": clean_value(row.time_sec),
            "valence": clean_value(row.valence),
            "arousal": clean_value(row.arousal),
            "valence_0_1": clean_value(row.valence_0_1),
            "arousal_0_1": clean_value(row.arousal_0_1),
            "valence_teacher_model_id": row.valence_teacher_model_id,
            "arousal_teacher_model_id": row.arousal_teacher_model_id,
            "feature_backbone": row.feature_backbone,
        })
    insert_batches("paper_top60_quota_100_va_frames", frame_rows, 1000)

    sections = pd.read_csv(STAGE3 / "top60_quota_100_section_features.csv", dtype={"song_id": str})
    section_rows = []
    for row in sections.itertuples(index=False):
        section_rows.append({
            "run_id": run_id,
            "song_id": row.song_id,
            "human_annotation_id": row.human_annotation_id,
            "section_order": int(row.section_order),
            "section_type": row.section_type,
            "start_time_sec": clean_value(row.section_start_sec),
            "end_time_sec": clean_value(row.section_end_sec),
            "duration_sec": clean_value(row.section_duration_sec),
            "assigned_frame_count": int(row.frame_count),
            "mean_valence": clean_value(row.mean_valence),
            "std_valence": clean_value(row.std_valence),
            "min_valence": clean_value(row.min_valence),
            "max_valence": clean_value(row.max_valence),
            "entry_valence": clean_value(row.entry_valence),
            "exit_valence": clean_value(row.exit_valence),
            "delta_valence": clean_value(row.delta_valence),
            "slope_valence": clean_value(row.slope_valence_per_sec),
            "mean_arousal": clean_value(row.mean_arousal),
            "std_arousal": clean_value(row.std_arousal),
            "min_arousal": clean_value(row.min_arousal),
            "max_arousal": clean_value(row.max_arousal),
            "entry_arousal": clean_value(row.entry_arousal),
            "exit_arousal": clean_value(row.exit_arousal),
            "delta_arousal": clean_value(row.delta_arousal),
            "slope_arousal": clean_value(row.slope_arousal_per_sec),
            "previous_section_type": clean_value(row.previous_section_type),
            "next_section_type": clean_value(row.next_section_type),
            "change_from_prev_valence": clean_value(row.change_from_prev_valence),
            "change_from_prev_arousal": clean_value(row.change_from_prev_arousal),
            "change_to_next_valence": clean_value(row.change_to_next_valence),
            "change_to_next_arousal": clean_value(row.change_to_next_arousal),
        })
    insert_batches("paper_top60_quota_100_section_va", section_rows, 500)

    songs = pd.read_csv(STAGE3 / "top60_quota_100_song_structure_features.csv", dtype={"song_id": str})
    song_rows = []
    for row in songs.itertuples(index=False):
        song_rows.append({
            "run_id": run_id,
            "song_id": row.song_id,
            "title": row.title,
            "artist": clean_value(row.artist),
            "duration_sec": clean_value(row.duration_sec),
            "section_count": int(row.section_count),
            "frame_count": int(row.assigned_frame_count),
            "mean_valence": clean_value(row.mean_valence),
            "mean_arousal": clean_value(row.mean_arousal),
            "valence_range": clean_value(row.global_valence_range),
            "arousal_range": clean_value(row.global_arousal_range),
            "start_valence": clean_value(row.start_section_mean_valence),
            "end_valence": clean_value(row.end_section_mean_valence),
            "net_valence_change": clean_value(row.net_section_valence_change),
            "start_arousal": clean_value(row.start_section_mean_arousal),
            "end_arousal": clean_value(row.end_section_mean_arousal),
            "net_arousal_change": clean_value(row.net_section_arousal_change),
            "arousal_peak_section_type": clean_value(row.arousal_peak_section_type),
            "arousal_peak_section_order": int(row.arousal_peak_section_order),
            "valence_peak_section_type": clean_value(row.valence_peak_section_type),
            "valence_peak_section_order": int(row.valence_peak_section_order),
        })
    insert_batches("paper_top60_quota_100_song_va", song_rows, 500)

    postgrest(
        "PATCH",
        f"/rest/v1/paper_top60_quota_100_va_runs?id=eq.{run_id}",
        [{"status": "pass", "updated_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds")}],
        prefer="return=minimal",
    )
    receipt = {
        "created_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "dataset_version": DATASET_VERSION,
        "run_label": RUN_LABEL,
        "run_id": run_id,
        "frames_inserted": len(frame_rows),
        "sections_inserted": len(section_rows),
        "songs_inserted": len(song_rows),
        "target_tables": [
            "paper_top60_quota_100_va_runs",
            "paper_top60_quota_100_va_frames",
            "paper_top60_quota_100_section_va",
            "paper_top60_quota_100_song_va",
        ],
    }
    out = DATA / "supabase_upload_receipt.json"
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
