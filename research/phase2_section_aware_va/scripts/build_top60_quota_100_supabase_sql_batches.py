#!/usr/bin/env python3
"""Build SQL batches for uploading top60 quota 100 VA outputs via Supabase MCP."""

from __future__ import annotations

import json
import math
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
SQL_DIR = DATA / "supabase_sql_batches"
RUN_LABEL = "top60_quota_100_teacher_v1_20260814"
DATASET_VERSION = "qq_music_top60_100_20260813"
RUN_ID = "11111111-6010-4100-8000-202608140001"
CHINA_TZ = timezone(timedelta(hours=8))


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def sql_literal_json(value: Any) -> str:
    return "'" + json.dumps(value, ensure_ascii=False).replace("'", "''") + "'::jsonb"


def write_batch(path: Path, table: str, rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    cols = ", ".join(name for name, _ in columns)
    defs = ", ".join(f"{name} {typ}" for name, typ in columns)
    sql = f"""
insert into public.{table} ({cols})
select {cols}
from jsonb_to_recordset({sql_literal_json(rows)}) as x({defs})
on conflict do nothing;
"""
    path.write_text(sql.strip() + "\n", encoding="utf-8")


def chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[start:start + size] for start in range(0, len(rows), size)]


def main() -> None:
    SQL_DIR.mkdir(parents=True, exist_ok=True)
    for old_batch in SQL_DIR.glob("*.sql"):
        old_batch.unlink()
    stage1_protocol = json.loads((STAGE1 / "top60_quota_100_protocol.json").read_text(encoding="utf-8"))
    inference_manifest = json.loads((STAGE2 / "top60_quota_100_inference_manifest.json").read_text(encoding="utf-8"))
    frame_audit = json.loads((STAGE2 / "top60_quota_100_output_audit_summary.json").read_text(encoding="utf-8"))
    alignment_manifest = json.loads((STAGE3 / "top60_quota_100_alignment_manifest.json").read_text(encoding="utf-8"))
    if frame_audit.get("overall_status") != "pass" or alignment_manifest.get("status") != "pass":
        raise RuntimeError("Cannot build upload SQL before audits pass")

    run_manifest = {
        "stage1": stage1_protocol,
        "stage2": inference_manifest,
        "frame_audit": frame_audit,
        "stage3": alignment_manifest,
    }
    run_sql = f"""
begin;
delete from public.paper_top60_quota_100_va_runs
where dataset_version = '{DATASET_VERSION}' and run_label = '{RUN_LABEL}';
insert into public.paper_top60_quota_100_va_runs (
  id, dataset_version, run_label, status, songs, sections, frames,
  teacher_valence_model_id, teacher_arousal_model_id, feature_backbone,
  frame_hop_sec, teacher_window_sec, teacher_stride_sec,
  source_manifest_sha256, frame_va_sha256, section_features_sha256, song_features_sha256,
  manifest, created_at, updated_at
) values (
  '{RUN_ID}'::uuid, '{DATASET_VERSION}', '{RUN_LABEL}', 'running',
  {int(inference_manifest['songs'])}, {int(alignment_manifest['sections'])}, {int(inference_manifest['frames'])},
  '{inference_manifest['model_ids']['valence']}', '{inference_manifest['model_ids']['arousal']}',
  '{inference_manifest['feature_backbone'].replace("'", "''")}',
  {float(inference_manifest['frame_hop_sec'])}, {float(inference_manifest['teacher_window_sec'])}, {float(inference_manifest['teacher_stride_sec'])},
  '{stage1_protocol['manifest_sha256']}', '{inference_manifest['combined_output_sha256']}',
  '{alignment_manifest['section_features_sha256']}', '{alignment_manifest['song_features_sha256']}',
  {sql_literal_json(run_manifest)}, now(), now()
);
commit;
"""
    (SQL_DIR / "000_run.sql").write_text(run_sql.strip() + "\n", encoding="utf-8")

    frames = pd.read_csv(STAGE2 / "top60_quota_100_frame_va.csv", dtype={"song_id": str})
    frames["frame_index"] = frames.groupby("song_id").cumcount() + 1
    frame_rows = [{
        "run_id": RUN_ID,
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
    } for row in frames.itertuples(index=False)]
    frame_cols = [
        ("run_id", "uuid"), ("song_id", "text"), ("frame_index", "integer"), ("time_sec", "numeric"),
        ("valence", "numeric"), ("arousal", "numeric"), ("valence_0_1", "numeric"), ("arousal_0_1", "numeric"),
        ("valence_teacher_model_id", "text"), ("arousal_teacher_model_id", "text"), ("feature_backbone", "text"),
    ]
    for idx, batch in enumerate(chunks(frame_rows, 5000), start=1):
        write_batch(SQL_DIR / f"frames_{idx:03d}.sql", "paper_top60_quota_100_va_frames", batch, frame_cols)

    sections = pd.read_csv(STAGE3 / "top60_quota_100_section_features.csv", dtype={"song_id": str})
    section_rows = [{
        "run_id": RUN_ID, "song_id": row.song_id, "human_annotation_id": row.human_annotation_id,
        "section_order": int(row.section_order), "section_type": row.section_type,
        "start_time_sec": clean_value(row.section_start_sec), "end_time_sec": clean_value(row.section_end_sec),
        "duration_sec": clean_value(row.section_duration_sec), "assigned_frame_count": int(row.frame_count),
        "mean_valence": clean_value(row.mean_valence), "std_valence": clean_value(row.std_valence),
        "min_valence": clean_value(row.min_valence), "max_valence": clean_value(row.max_valence),
        "entry_valence": clean_value(row.entry_valence), "exit_valence": clean_value(row.exit_valence),
        "delta_valence": clean_value(row.delta_valence), "slope_valence": clean_value(row.slope_valence_per_sec),
        "mean_arousal": clean_value(row.mean_arousal), "std_arousal": clean_value(row.std_arousal),
        "min_arousal": clean_value(row.min_arousal), "max_arousal": clean_value(row.max_arousal),
        "entry_arousal": clean_value(row.entry_arousal), "exit_arousal": clean_value(row.exit_arousal),
        "delta_arousal": clean_value(row.delta_arousal), "slope_arousal": clean_value(row.slope_arousal_per_sec),
        "previous_section_type": clean_value(row.previous_section_type), "next_section_type": clean_value(row.next_section_type),
        "change_from_prev_valence": clean_value(row.change_from_prev_valence),
        "change_from_prev_arousal": clean_value(row.change_from_prev_arousal),
        "change_to_next_valence": clean_value(row.change_to_next_valence),
        "change_to_next_arousal": clean_value(row.change_to_next_arousal),
    } for row in sections.itertuples(index=False)]
    section_cols = [
        ("run_id", "uuid"), ("song_id", "text"), ("human_annotation_id", "uuid"), ("section_order", "integer"),
        ("section_type", "text"), ("start_time_sec", "numeric"), ("end_time_sec", "numeric"),
        ("duration_sec", "numeric"), ("assigned_frame_count", "integer"),
        ("mean_valence", "numeric"), ("std_valence", "numeric"), ("min_valence", "numeric"), ("max_valence", "numeric"),
        ("entry_valence", "numeric"), ("exit_valence", "numeric"), ("delta_valence", "numeric"), ("slope_valence", "numeric"),
        ("mean_arousal", "numeric"), ("std_arousal", "numeric"), ("min_arousal", "numeric"), ("max_arousal", "numeric"),
        ("entry_arousal", "numeric"), ("exit_arousal", "numeric"), ("delta_arousal", "numeric"), ("slope_arousal", "numeric"),
        ("previous_section_type", "text"), ("next_section_type", "text"),
        ("change_from_prev_valence", "numeric"), ("change_from_prev_arousal", "numeric"),
        ("change_to_next_valence", "numeric"), ("change_to_next_arousal", "numeric"),
    ]
    write_batch(SQL_DIR / "sections.sql", "paper_top60_quota_100_section_va", section_rows, section_cols)

    songs = pd.read_csv(STAGE3 / "top60_quota_100_song_structure_features.csv", dtype={"song_id": str})
    song_rows = [{
        "run_id": RUN_ID, "song_id": row.song_id, "title": row.title, "artist": clean_value(row.artist),
        "duration_sec": clean_value(row.duration_sec), "section_count": int(row.section_count),
        "frame_count": int(row.assigned_frame_count), "mean_valence": clean_value(row.mean_valence),
        "mean_arousal": clean_value(row.mean_arousal), "valence_range": clean_value(row.global_valence_range),
        "arousal_range": clean_value(row.global_arousal_range), "start_valence": clean_value(row.start_section_mean_valence),
        "end_valence": clean_value(row.end_section_mean_valence), "net_valence_change": clean_value(row.net_section_valence_change),
        "start_arousal": clean_value(row.start_section_mean_arousal), "end_arousal": clean_value(row.end_section_mean_arousal),
        "net_arousal_change": clean_value(row.net_section_arousal_change),
        "arousal_peak_section_type": clean_value(row.arousal_peak_section_type),
        "arousal_peak_section_order": int(row.arousal_peak_section_order),
        "valence_peak_section_type": clean_value(row.valence_peak_section_type),
        "valence_peak_section_order": int(row.valence_peak_section_order),
    } for row in songs.itertuples(index=False)]
    song_cols = [
        ("run_id", "uuid"), ("song_id", "text"), ("title", "text"), ("artist", "text"),
        ("duration_sec", "numeric"), ("section_count", "integer"), ("frame_count", "integer"),
        ("mean_valence", "numeric"), ("mean_arousal", "numeric"), ("valence_range", "numeric"), ("arousal_range", "numeric"),
        ("start_valence", "numeric"), ("end_valence", "numeric"), ("net_valence_change", "numeric"),
        ("start_arousal", "numeric"), ("end_arousal", "numeric"), ("net_arousal_change", "numeric"),
        ("arousal_peak_section_type", "text"), ("arousal_peak_section_order", "integer"),
        ("valence_peak_section_type", "text"), ("valence_peak_section_order", "integer"),
    ]
    write_batch(SQL_DIR / "songs.sql", "paper_top60_quota_100_song_va", song_rows, song_cols)

    finalize = f"""
update public.paper_top60_quota_100_va_runs
set status = 'pass', updated_at = now()
where id = '{RUN_ID}'::uuid;

select
  (select count(*) from public.paper_top60_quota_100_va_runs where id = '{RUN_ID}'::uuid) as runs,
  (select count(*) from public.paper_top60_quota_100_va_frames where run_id = '{RUN_ID}'::uuid) as frames,
  (select count(distinct song_id) from public.paper_top60_quota_100_va_frames where run_id = '{RUN_ID}'::uuid) as frame_songs,
  (select count(*) from public.paper_top60_quota_100_section_va where run_id = '{RUN_ID}'::uuid) as sections,
  (select count(distinct song_id) from public.paper_top60_quota_100_section_va where run_id = '{RUN_ID}'::uuid) as section_songs,
  (select count(*) from public.paper_top60_quota_100_song_va where run_id = '{RUN_ID}'::uuid) as songs;
"""
    (SQL_DIR / "999_finalize_and_verify.sql").write_text(finalize.strip() + "\n", encoding="utf-8")
    receipt = {
        "created_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "sql_dir": str(SQL_DIR.resolve()),
        "run_id": RUN_ID,
        "run_label": RUN_LABEL,
        "frame_rows": len(frame_rows),
        "section_rows": len(section_rows),
        "song_rows": len(song_rows),
        "sql_files": sorted(path.name for path in SQL_DIR.glob("*.sql")),
    }
    (DATA / "supabase_sql_batch_manifest.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
