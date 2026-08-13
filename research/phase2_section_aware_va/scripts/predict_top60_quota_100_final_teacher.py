#!/usr/bin/env python3
"""Run frozen VA Teachers for the top60 quota 100-song corpus.

Existing verified full-corpus outputs are reused when song_id and audio hash
match. Missing songs are inferred with the same frozen Teacher deployment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import predict_blind_sample_final_teacher as core


PHASE2 = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PHASE2 / "data/top60_quota_100_20260813/stage1/top60_quota_100_manifest.csv"
DEFAULT_OUTPUT = PHASE2 / "data/top60_quota_100_20260813/stage2"
REUSE_STAGE2 = PHASE2 / "data/full_corpus/stage2"


def link_or_copy_verified(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if core.sha256_file(source) != core.sha256_file(target):
            raise RuntimeError(f"Existing target differs from source: {target}")
        return "existing_verified"
    try:
        os.link(source, target)
        return "hard_linked"
    except OSError:
        import shutil

        shutil.copy2(source, target)
        if core.sha256_file(source) != core.sha256_file(target):
            raise RuntimeError(f"Copied target differs from source: {target}")
        return "copied_verified"


def append_log(path: Path, row: dict[str, Any]) -> None:
    fields = [
        "song_id", "title", "status", "frames", "duration_sec", "mert_chunks",
        "feature_seconds", "inference_seconds", "total_seconds", "error",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def can_reuse(row: dict[str, str], output_dir: Path) -> tuple[bool, dict[str, Any]]:
    song_id = row["song_id"]
    feature = output_dir / "features" / f"{song_id}.npz"
    frame = output_dir / "frame_va" / f"{song_id}_frame_va.csv"
    meta = output_dir / "metadata" / f"{song_id}.json"
    if not (feature.is_file() and frame.is_file() and meta.is_file()):
        return False, {}
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    if metadata.get("audio_sha256") != row["audio_sha256"]:
        return False, {}
    if metadata.get("feature_sha256") != core.sha256_file(feature):
        return False, {}
    if metadata.get("frame_va_sha256") != core.sha256_file(frame):
        return False, {}
    return True, {"feature": feature, "frame": frame, "metadata": meta, "metadata_json": metadata}


def infer_one(
    row: dict[str, str],
    args: argparse.Namespace,
    mert: Any,
    processor: Any,
    transformers_version: str,
    teachers: dict[str, Any],
    smile_mean: np.ndarray,
    smile_std: np.ndarray,
    feature_cols: list[str],
    teacher_hashes: dict[str, str],
    mert_device: Any,
    teacher_device: Any,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    song_id = row["song_id"]
    audio_path = core.PROJECT_ROOT / row["audio_path"].lstrip("/")
    feature_path = args.output_dir / "features" / f"{song_id}.npz"
    output_path = args.output_dir / "frame_va" / f"{song_id}_frame_va.csv"
    meta_path = args.output_dir / "metadata" / f"{song_id}.json"
    started = time.time()
    feature_started = time.time()
    if feature_path.exists() and not args.force_features:
        cached = np.load(feature_path, allow_pickle=True)
        if str(cached["audio_sha256"].item()) != row["audio_sha256"]:
            raise RuntimeError("Cached feature audio hash differs from frozen manifest")
        target_times = cached["time_sec"].astype(np.float32)
        mert_embedding = cached["mert_hidden6"].astype(np.float32)
        smile_sequence = cached["opensmile_sequence"].astype(np.float32)
        feature = np.concatenate((mert_embedding, smile_sequence), axis=1).astype(np.float32)
        old_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        duration_sec = float(old_meta.get("duration_sec", target_times[-1]))
        mert_meta = {"chunks_processed": int(old_meta.get("mert_chunks", 0))}
    else:
        _, duration_sec = core.load_audio(audio_path, 24000)
        target_times = np.arange(0.5, duration_sec + 1e-6, args.hop_sec, dtype=np.float32)
        mert_embedding, mert_meta = core.extract_mert_hidden6(
            audio_path,
            target_times,
            mert,
            processor,
            mert_device,
            args.chunk_sec,
            args.overlap_sec,
            args.pool_window_sec,
        )
        target_times = target_times[: len(mert_embedding)]
        smile_raw = core.extract_opensmile_dynamic(
            audio_path,
            target_times,
            feature_cols,
            args.opensmile_window_sec,
        )
        smile_sequence = ((smile_raw - smile_mean) / smile_std).astype(np.float32)
        feature = np.concatenate((mert_embedding, smile_sequence), axis=1).astype(np.float32)
        np.savez_compressed(
            feature_path,
            song_id=song_id,
            time_sec=target_times,
            mert_hidden6=mert_embedding,
            opensmile_raw=smile_raw,
            opensmile_sequence=smile_sequence,
            audio_sha256=row["audio_sha256"],
            feature_backbone=core.FEATURE_BACKBONE,
        )
    if feature.shape[1] != 1028 or not np.isfinite(feature).all():
        raise RuntimeError(f"Invalid combined feature shape/content: {feature.shape}")
    feature_seconds = time.time() - feature_started
    inference_started = time.time()
    predictions: dict[str, np.ndarray] = {}
    global_means: dict[str, float] = {}
    window_counts: dict[str, int] = {}
    window_frames = max(1, int(round(args.teacher_window_sec / args.hop_sec)))
    stride_frames = max(1, int(round(args.teacher_stride_sec / args.hop_sec)))
    for dimension, (model, mean, std, _) in teachers.items():
        predictions[dimension], global_means[dimension], window_counts[dimension] = core.predict_teacher_windowed(
            feature,
            model,
            mean,
            std,
            teacher_device,
            window_frames,
            stride_frames,
        )
    inference_seconds = time.time() - inference_started
    output = pd.DataFrame({
        "song_id": song_id,
        "title": row["title"],
        "artist": row["artist"],
        "time_sec": target_times,
        "valence": predictions["valence"],
        "arousal": predictions["arousal"],
        "valence_0_1": (predictions["valence"] + 1.0) / 2.0,
        "arousal_0_1": (predictions["arousal"] + 1.0) / 2.0,
        "valence_teacher_model_id": core.MODEL_IDS["valence"],
        "arousal_teacher_model_id": core.MODEL_IDS["arousal"],
        "feature_backbone": core.FEATURE_BACKBONE,
    })
    output.to_csv(output_path, index=False)
    meta = {
        "song_id": song_id,
        "title": row["title"],
        "artist": row["artist"],
        "audio_path": row["audio_path"],
        "audio_sha256": row["audio_sha256"],
        "duration_sec": duration_sec,
        "frames": len(output),
        "frame_hop_sec": args.hop_sec,
        "first_frame_sec": float(target_times[0]),
        "last_frame_sec": float(target_times[-1]),
        "mert_chunks": int(mert_meta["chunks_processed"]),
        "mert_model": args.model_name,
        "mert_layer": 6,
        "transformers_version": transformers_version,
        "feature_backbone": core.FEATURE_BACKBONE,
        "valence_teacher_model_id": core.MODEL_IDS["valence"],
        "arousal_teacher_model_id": core.MODEL_IDS["arousal"],
        "valence_global_mean": global_means["valence"],
        "arousal_global_mean": global_means["arousal"],
        "teacher_window_sec": args.teacher_window_sec,
        "teacher_stride_sec": args.teacher_stride_sec,
        "teacher_window_blend": "Hann weighted overlap-add",
        "valence_window_count": window_counts["valence"],
        "arousal_window_count": window_counts["arousal"],
        "teacher_hashes": teacher_hashes,
        "feature_sha256": core.sha256_file(feature_path),
        "frame_va_sha256": core.sha256_file(output_path),
        "feature_domain_notice": "Raw-MP3 OpenSMILE reproduction is not bit-equivalent to PMEmo official feature CSV; output is pseudo-VA under external complete-song validation.",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_row = {
        "song_id": song_id,
        "title": row["title"],
        "status": "ok",
        "frames": len(output),
        "duration_sec": round(duration_sec, 6),
        "mert_chunks": int(mert_meta["chunks_processed"]),
        "feature_seconds": round(feature_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "total_seconds": round(time.time() - started, 3),
        "error": "",
    }
    completed = {
        "song_id": song_id,
        "title": row["title"],
        "frames": len(output),
        "output": str(output_path),
        "output_sha256": core.sha256_file(output_path),
        "feature": str(feature_path),
        "feature_sha256": core.sha256_file(feature_path),
        "source": "new_inference",
    }
    return output, completed, log_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-stage2-dir", type=Path, default=REUSE_STAGE2)
    parser.add_argument("--mert-device", default="auto")
    parser.add_argument("--teacher-device", default="cpu")
    parser.add_argument("--model-name", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--chunk-sec", type=float, default=30.0)
    parser.add_argument("--overlap-sec", type=float, default=1.0)
    parser.add_argument("--pool-window-sec", type=float, default=0.5)
    parser.add_argument("--opensmile-window-sec", type=float, default=1.0)
    parser.add_argument("--teacher-window-sec", type=float, default=30.0)
    parser.add_argument("--teacher-stride-sec", type=float, default=15.0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    if len(rows) != 100:
        raise RuntimeError(f"Top60 quota corpus must contain 100 songs, got {len(rows)}")
    teacher_hashes = core.verify_teacher_files()
    core.verify_audio_manifest(rows)
    for directory in (args.output_dir / "features", args.output_dir / "frame_va", args.output_dir / "metadata"):
        directory.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "inference_log.csv"

    all_outputs: list[pd.DataFrame] = []
    completed: list[dict[str, Any]] = []
    pending: list[dict[str, str]] = []
    reuse_rows: list[dict[str, Any]] = []
    started_all = time.time()
    for row in rows:
        reusable, paths = can_reuse(row, args.reuse_stage2_dir)
        song_id = row["song_id"]
        if not reusable or args.force:
            pending.append(row)
            continue
        targets = {
            "feature": args.output_dir / "features" / f"{song_id}.npz",
            "frame": args.output_dir / "frame_va" / f"{song_id}_frame_va.csv",
            "metadata": args.output_dir / "metadata" / f"{song_id}.json",
        }
        methods = {
            key: link_or_copy_verified(paths[key], targets[key])
            for key in ("feature", "frame", "metadata")
        }
        output = pd.read_csv(targets["frame"])
        all_outputs.append(output)
        completed.append({
            "song_id": song_id,
            "title": row["title"],
            "frames": len(output),
            "output": str(targets["frame"]),
            "output_sha256": core.sha256_file(targets["frame"]),
            "feature": str(targets["feature"]),
            "feature_sha256": core.sha256_file(targets["feature"]),
            "source": "reused_full_corpus_99",
        })
        append_log(log_path, {
            "song_id": song_id,
            "title": row["title"],
            "status": "reused_verified",
            "frames": len(output),
            "duration_sec": paths["metadata_json"]["duration_sec"],
            "mert_chunks": paths["metadata_json"]["mert_chunks"],
            "feature_seconds": 0,
            "inference_seconds": 0,
            "total_seconds": 0,
            "error": "",
        })
        reuse_rows.append({
            "song_id": song_id,
            "title": row["title"],
            "audio_sha256": row["audio_sha256"],
            "frame_va_sha256": core.sha256_file(targets["frame"]),
            "feature_sha256": core.sha256_file(targets["feature"]),
            "link_method": ";".join(sorted(set(methods.values()))),
        })

    print(json.dumps({"reused": len(reuse_rows), "requires_inference": len(pending)}, ensure_ascii=False), flush=True)
    if pending:
        smile_norm = np.load(core.OPENSMILE_NORM_PATH, allow_pickle=True)
        smile_mean = smile_norm["mean"].astype(np.float32)
        smile_std = smile_norm["std"].astype(np.float32)
        feature_cols = [str(value) for value in smile_norm["feature_cols"].tolist()]
        mert_device = core.resolve_device(args.mert_device)
        teacher_device = core.resolve_device(args.teacher_device)
        print(json.dumps({
            "songs": len(pending),
            "mert_device": str(mert_device),
            "teacher_device": str(teacher_device),
            "teacher_hashes": teacher_hashes,
        }, ensure_ascii=False), flush=True)
        mert, processor, transformers_version = core.load_mert(args.model_name, mert_device, args.allow_download)
        teachers = {dimension: core.load_teacher(dimension, teacher_device) for dimension in ("valence", "arousal")}
        for index, row in enumerate(pending, start=1):
            try:
                output, done, log_row = infer_one(
                    row,
                    args,
                    mert,
                    processor,
                    transformers_version,
                    teachers,
                    smile_mean,
                    smile_std,
                    feature_cols,
                    teacher_hashes,
                    mert_device,
                    teacher_device,
                )
                all_outputs.append(output)
                completed.append(done)
            except Exception as exc:  # noqa: BLE001
                log_row = {
                    "song_id": row["song_id"],
                    "title": row["title"],
                    "status": "error",
                    "frames": 0,
                    "duration_sec": "",
                    "mert_chunks": "",
                    "feature_seconds": "",
                    "inference_seconds": "",
                    "total_seconds": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            append_log(log_path, log_row)
            print(f"{index}/{len(pending)} {row['song_id']} {row['title']} status={log_row['status']} frames={log_row['frames']}", flush=True)
            if log_row["status"] == "error":
                raise RuntimeError(log_row["error"])

    order = {row["song_id"]: index for index, row in enumerate(rows)}
    combined = pd.concat(all_outputs, ignore_index=True)
    combined["_order"] = combined["song_id"].map(order)
    combined = combined.sort_values(["_order", "time_sec"]).drop(columns=["_order"])
    combined_path = args.output_dir / "top60_quota_100_frame_va.csv"
    combined.to_csv(combined_path, index=False)
    reuse_path = args.output_dir / "reuse_audit.csv"
    pd.DataFrame(reuse_rows).to_csv(reuse_path, index=False)
    run_manifest = {
        "created_at": datetime.now(core.CHINA_TZ).isoformat(timespec="seconds"),
        "protocol": "Top60 quota 100-song audio-only inference with frozen final clean VA teachers",
        "songs": len(completed),
        "reused_full_corpus_99": len(reuse_rows),
        "new_inference": len(pending),
        "frames": int(sum(row["frames"] for row in completed)),
        "elapsed_sec": round(time.time() - started_all, 3),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": core.sha256_file(args.manifest),
        "teacher_hashes": teacher_hashes,
        "model_ids": core.MODEL_IDS,
        "feature_backbone": core.FEATURE_BACKBONE,
        "frame_hop_sec": args.hop_sec,
        "teacher_window_sec": args.teacher_window_sec,
        "teacher_stride_sec": args.teacher_stride_sec,
        "teacher_window_blend": "Hann weighted overlap-add",
        "section_boundaries_read": False,
        "human_va_answers_read": False,
        "combined_output": str(combined_path.resolve()),
        "combined_output_sha256": core.sha256_file(combined_path),
        "reuse_audit": str(reuse_path.resolve()),
        "completed": completed,
    }
    (args.output_dir / "top60_quota_100_inference_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
