#!/usr/bin/env python3
"""Run the frozen final clean VA teachers on the blind 15-song audio sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_musicva")

import numpy as np
import opensmile
import pandas as pd
import torch
import torchaudio


PHASE2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE2_ROOT.parents[1]
PMEMO_ROOT = PROJECT_ROOT / "research/phase1f_external_va_datasets/pmemo"
FINAL_ROOT = PMEMO_ROOT / "final_clean_teacher"
FINAL_OUTPUT = FINAL_ROOT / "outputs/final_clean_teacher_v1/final_fit"
FINAL_TRAIN_SCRIPTS = FINAL_ROOT / "scripts"
sys.path.insert(0, str(FINAL_TRAIN_SCRIPTS))

from train_final_clean_teacher import GlobalLocalBiGRU  # noqa: E402


DEFAULT_MANIFEST = PHASE2_ROOT / "data/blind_validation/stage2/blind_sample_15.csv"
DEFAULT_OUTPUT = PHASE2_ROOT / "data/blind_validation/stage3"
CONFIG_PATH = FINAL_ROOT / "configs/final_clean_teacher_v1.json"
OPENSMILE_NORM_PATH = PMEMO_ROOT / "outputs/opensmile_baseline/dataset/feature_norm.npz"
EXPECTED_FROZEN_HASHES = {
    "config": "cb32a52d4b7c90f3b41aca17f82ab66cde8d33634497d6862bf0b6ab2b8288a1",
    "valence_model": "6cddf9ce3f12f341dac5d53b99c108675fdac9c8c17e323164f77388d88918ae",
    "arousal_model": "aba66d037ec8f4c30917a67dbb9d97f0dfc2fc781fa4191d4202812051c368bb",
}
MODEL_IDS = {
    "valence": "pmemo_final_clean_valence_teacher_v1",
    "arousal": "pmemo_final_clean_arousal_teacher_v1",
}
FEATURE_BACKBONE = "MERT-v1-95M hidden-6 + PMEmo-style OpenSMILE ComParE_2016 dynamic 260d"
CHINA_TZ = timezone(timedelta(hours=8))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def verify_teacher_files() -> dict[str, str]:
    paths = {
        "config": CONFIG_PATH,
        "valence_model": FINAL_OUTPUT / "valence/model.pt",
        "arousal_model": FINAL_OUTPUT / "arousal/model.pt",
        "valence_normalizer": FINAL_OUTPUT / "valence/normalizer.npz",
        "arousal_normalizer": FINAL_OUTPUT / "arousal/normalizer.npz",
        "opensmile_feature_norm": OPENSMILE_NORM_PATH,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": expected, "actual": hashes[name]}
        for name, expected in EXPECTED_FROZEN_HASHES.items()
        if hashes[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"Frozen Teacher hash mismatch: {mismatches}")
    return hashes


def verify_audio_manifest(rows: list[dict[str, str]]) -> None:
    mismatches = []
    for row in rows:
        path = PROJECT_ROOT / row["audio_path"].lstrip("/")
        actual = sha256_file(path)
        if actual != row["audio_sha256"]:
            mismatches.append({"song_id": row["song_id"], "expected": row["audio_sha256"], "actual": actual})
    if mismatches:
        raise RuntimeError(f"Frozen audio hash mismatch: {mismatches}")


def import_transformers() -> tuple[Any, Any, str]:
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    import transformers

    return AutoModel, Wav2Vec2FeatureExtractor, transformers.__version__


def load_mert(model_name: str, device: torch.device, allow_download: bool) -> tuple[Any, Any, str]:
    AutoModel, Wav2Vec2FeatureExtractor, version = import_transformers()
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=not allow_download,
    )
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=not allow_download,
    )
    model.to(device)
    model.eval()
    return model, processor, version


def load_audio(path: Path, sample_rate: int) -> tuple[np.ndarray, float]:
    waveform, source_rate = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0, keepdim=True)
    if int(source_rate) != sample_rate:
        waveform = torchaudio.functional.resample(waveform, int(source_rate), sample_rate)
    audio = waveform.squeeze(0).numpy().astype(np.float32)
    return audio, len(audio) / sample_rate


def pool_chunk_to_targets(
    hidden: np.ndarray,
    chunk_start_sec: float,
    chunk_duration_sec: float,
    target_times: np.ndarray,
    pool_window_sec: float,
    sums: np.ndarray | None,
    counts: np.ndarray,
) -> np.ndarray:
    hidden_frames, hidden_dim = hidden.shape
    if sums is None:
        sums = np.zeros((len(target_times), hidden_dim), dtype=np.float64)
    hidden_times = chunk_start_sec + np.linspace(0.0, chunk_duration_sec, hidden_frames, endpoint=False)
    half = pool_window_sec / 2.0
    in_chunk = np.where((target_times >= chunk_start_sec) & (target_times < chunk_start_sec + chunk_duration_sec))[0]
    for target_index in in_chunk:
        target_time = float(target_times[target_index])
        mask = (hidden_times >= target_time - half) & (hidden_times < target_time + half)
        pooled = hidden[int(np.argmin(np.abs(hidden_times - target_time)))] if not np.any(mask) else hidden[mask].mean(axis=0)
        sums[target_index] += pooled
        counts[target_index] += 1
    return sums


@torch.no_grad()
def extract_mert_hidden6(
    audio_path: Path,
    target_times: np.ndarray,
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    chunk_sec: float,
    overlap_sec: float,
    pool_window_sec: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    sample_rate = int(getattr(processor, "sampling_rate", None) or 24000)
    audio, duration_sec = load_audio(audio_path, sample_rate)
    times = target_times[target_times < duration_sec]
    step_sec = max(chunk_sec - overlap_sec, 1.0)
    sums: np.ndarray | None = None
    counts = np.zeros(len(times), dtype=np.int32)
    chunks = 0
    chunk_start = 0.0
    process_until = min(duration_sec, float(times.max()) + pool_window_sec + overlap_sec)
    while chunk_start < process_until:
        chunk_end = min(chunk_start + chunk_sec, duration_sec)
        chunk = audio[int(chunk_start * sample_rate) : int(chunk_end * sample_rate)]
        if not len(chunk):
            break
        inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[6].squeeze(0).detach().cpu().numpy().astype(np.float32)
        sums = pool_chunk_to_targets(
            hidden,
            chunk_start,
            chunk_end - chunk_start,
            times,
            pool_window_sec,
            sums,
            counts,
        )
        chunks += 1
        chunk_start += step_sec
    if sums is None or np.any(counts == 0):
        raise RuntimeError(f"MERT failed to cover all target frames: {audio_path}")
    embedding = (sums / counts[:, None]).astype(np.float32)
    return embedding, {
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "chunks_processed": chunks,
        "frames": len(times),
        "dim": embedding.shape[1],
    }


def index_seconds(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    index = frame.index
    starts = np.asarray([value.total_seconds() for value in index.get_level_values("start")], dtype=np.float32)
    ends = np.asarray([value.total_seconds() for value in index.get_level_values("end")], dtype=np.float32)
    return frame.reset_index(drop=True), (starts + ends) / 2.0


def extract_opensmile_dynamic(
    audio_path: Path,
    target_times: np.ndarray,
    feature_cols: list[str],
    window_sec: float,
) -> np.ndarray:
    lld_smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    delta_smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
    )
    lld, lld_times = index_seconds(lld_smile.process_file(str(audio_path)))
    delta, delta_times = index_seconds(delta_smile.process_file(str(audio_path)))
    lld_values = lld.to_numpy(dtype=np.float64)
    delta_values = delta.to_numpy(dtype=np.float64)
    available_names = [
        *(f"{column}_{stat}" for column in lld.columns for stat in ("mean", "std")),
        *(f"{column}_{stat}" for column in delta.columns for stat in ("mean", "std")),
    ]
    name_to_index = {name: index for index, name in enumerate(available_names)}
    # PMEmo's exported CSV uses the legacy `pcm_fftMag_` MFCC prefix, while
    # current openSMILE exposes the same ComParE descriptors as `mfcc_*`.
    for name, index in list(name_to_index.items()):
        if name.startswith("mfcc_sma"):
            name_to_index[f"pcm_fftMag_{name}"] = index
    missing = [name for name in feature_cols if name not in name_to_index]
    if missing:
        raise RuntimeError(f"OpenSMILE feature columns cannot be reproduced: {missing[:5]}")
    reorder = np.asarray([name_to_index[name] for name in feature_cols], dtype=np.int64)
    rows = []
    for target in target_times:
        lld_mask = (lld_times > target - window_sec) & (lld_times <= target)
        delta_mask = (delta_times > target - window_sec) & (delta_times <= target)
        if not np.any(lld_mask):
            lld_mask[int(np.argmin(np.abs(lld_times - target)))] = True
        if not np.any(delta_mask):
            delta_mask[int(np.argmin(np.abs(delta_times - target)))] = True
        lld_part = lld_values[lld_mask]
        delta_part = delta_values[delta_mask]
        lld_stats = np.column_stack((lld_part.mean(axis=0), lld_part.std(axis=0))).reshape(-1)
        delta_stats = np.column_stack((delta_part.mean(axis=0), delta_part.std(axis=0))).reshape(-1)
        rows.append(np.concatenate((lld_stats, delta_stats))[reorder])
    return np.asarray(rows, dtype=np.float32)


def load_teacher(dimension: str, device: torch.device) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, dict[str, Any]]:
    checkpoint = torch.load(FINAL_OUTPUT / dimension / "model.pt", map_location=device)
    config = checkpoint["config"]
    model = GlobalLocalBiGRU(
        int(checkpoint["input_dim"]),
        config,
        float(config[dimension]["local_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    normalizer = np.load(FINAL_OUTPUT / dimension / "normalizer.npz")
    return model, normalizer["mean"].astype(np.float32), normalizer["std"].astype(np.float32), checkpoint


@torch.no_grad()
def predict_teacher_windowed(
    feature: np.ndarray,
    model: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    window_frames: int,
    stride_frames: int,
) -> tuple[np.ndarray, float, int]:
    normalized = ((feature - mean) / std).astype(np.float32)
    length = len(normalized)
    if length <= window_frames:
        starts = [0]
    else:
        starts = list(range(0, length - window_frames + 1, stride_frames))
        if starts[-1] != length - window_frames:
            starts.append(length - window_frames)
    prediction_sum = np.zeros(length, dtype=np.float64)
    weight_sum = np.zeros(length, dtype=np.float64)
    for start in starts:
        end = min(start + window_frames, length)
        values = normalized[start:end]
        tensor = torch.from_numpy(values)[None, :, :].to(device)
        mask = torch.ones((1, len(values)), dtype=torch.bool, device=device)
        lengths = torch.tensor([len(values)], dtype=torch.long)
        prediction, _ = model(tensor, mask, lengths)
        prediction = prediction.squeeze(0).cpu().numpy().astype(np.float64)
        weights = np.hanning(len(values) + 2)[1:-1]
        weights = np.maximum(weights, 1e-3)
        prediction_sum[start:end] += prediction * weights
        weight_sum[start:end] += weights
    if np.any(weight_sum <= 0):
        raise RuntimeError("Sliding-window inference left uncovered frames")
    merged = (prediction_sum / weight_sum).astype(np.float32)
    return merged, float(merged.mean()), len(starts)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
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
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    if args.limit is not None:
        rows = rows[: args.limit]
    teacher_hashes = verify_teacher_files()
    verify_audio_manifest(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output_dir / "features"
    frame_dir = args.output_dir / "frame_va"
    meta_dir = args.output_dir / "metadata"
    for directory in (feature_dir, frame_dir, meta_dir):
        directory.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "inference_log.csv"

    smile_norm = np.load(OPENSMILE_NORM_PATH, allow_pickle=True)
    smile_mean = smile_norm["mean"].astype(np.float32)
    smile_std = smile_norm["std"].astype(np.float32)
    feature_cols = [str(value) for value in smile_norm["feature_cols"].tolist()]
    mert_device = resolve_device(args.mert_device)
    teacher_device = resolve_device(args.teacher_device)
    print(json.dumps({
        "songs": len(rows),
        "mert_device": str(mert_device),
        "teacher_device": str(teacher_device),
        "teacher_hashes": teacher_hashes,
    }, ensure_ascii=False), flush=True)
    mert, processor, transformers_version = load_mert(args.model_name, mert_device, args.allow_download)
    teachers = {
        dimension: load_teacher(dimension, teacher_device)
        for dimension in ("valence", "arousal")
    }

    all_outputs = []
    completed = []
    started_all = time.time()
    for index, row in enumerate(rows, start=1):
        song_id = row["song_id"]
        audio_path = PROJECT_ROOT / row["audio_path"].lstrip("/")
        feature_path = feature_dir / f"{song_id}.npz"
        output_path = frame_dir / f"{song_id}_frame_va.csv"
        meta_path = meta_dir / f"{song_id}.json"
        started = time.time()
        try:
            if output_path.exists() and meta_path.exists() and not args.force:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta["audio_sha256"] != row["audio_sha256"]:
                    raise RuntimeError("Existing output audio hash differs from frozen manifest")
                output = pd.read_csv(output_path)
                status = "skipped_verified"
                feature_seconds = 0.0
                inference_seconds = 0.0
                mert_chunks = int(meta["mert_chunks"])
                duration_sec = float(meta["duration_sec"])
            else:
                feature_started = time.time()
                if feature_path.exists() and not args.force_features:
                    cached = np.load(feature_path, allow_pickle=True)
                    cached_audio_hash = str(cached["audio_sha256"].item())
                    if cached_audio_hash != row["audio_sha256"]:
                        raise RuntimeError("Cached feature audio hash differs from frozen manifest")
                    target_times = cached["time_sec"].astype(np.float32)
                    mert_embedding = cached["mert_hidden6"].astype(np.float32)
                    smile_sequence = cached["opensmile_sequence"].astype(np.float32)
                    feature = np.concatenate((mert_embedding, smile_sequence), axis=1).astype(np.float32)
                    old_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                    duration_sec = float(old_meta.get("duration_sec", target_times[-1]))
                    mert_meta = {"chunks_processed": int(old_meta.get("mert_chunks", 0))}
                else:
                    _, duration_sec = load_audio(audio_path, 24000)
                    target_times = np.arange(0.5, duration_sec + 1e-6, args.hop_sec, dtype=np.float32)
                    mert_embedding, mert_meta = extract_mert_hidden6(
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
                    smile_raw = extract_opensmile_dynamic(
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
                        feature_backbone=FEATURE_BACKBONE,
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
                    predictions[dimension], global_means[dimension], window_counts[dimension] = predict_teacher_windowed(
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
                    "valence_teacher_model_id": MODEL_IDS["valence"],
                    "arousal_teacher_model_id": MODEL_IDS["arousal"],
                    "feature_backbone": FEATURE_BACKBONE,
                })
                output.to_csv(output_path, index=False)
                mert_chunks = int(mert_meta["chunks_processed"])
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
                    "mert_chunks": mert_chunks,
                    "mert_model": args.model_name,
                    "mert_layer": 6,
                    "transformers_version": transformers_version,
                    "feature_backbone": FEATURE_BACKBONE,
                    "valence_teacher_model_id": MODEL_IDS["valence"],
                    "arousal_teacher_model_id": MODEL_IDS["arousal"],
                    "valence_global_mean": global_means["valence"],
                    "arousal_global_mean": global_means["arousal"],
                    "teacher_window_sec": args.teacher_window_sec,
                    "teacher_stride_sec": args.teacher_stride_sec,
                    "teacher_window_blend": "Hann weighted overlap-add",
                    "valence_window_count": window_counts["valence"],
                    "arousal_window_count": window_counts["arousal"],
                    "teacher_hashes": teacher_hashes,
                    "feature_sha256": sha256_file(feature_path),
                    "frame_va_sha256": sha256_file(output_path),
                    "feature_domain_notice": "Raw-MP3 OpenSMILE reproduction is not bit-equivalent to PMEmo official feature CSV; output is pseudo-VA under external complete-song validation.",
                }
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                status = "ok"
            all_outputs.append(output)
            completed.append({
                "song_id": song_id,
                "title": row["title"],
                "frames": len(output),
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "feature": str(feature_path),
                "feature_sha256": sha256_file(feature_path),
            })
            log_row = {
                "song_id": song_id,
                "title": row["title"],
                "status": status,
                "frames": len(output),
                "duration_sec": round(duration_sec, 6),
                "mert_chunks": mert_chunks,
                "feature_seconds": round(feature_seconds, 3),
                "inference_seconds": round(inference_seconds, 3),
                "total_seconds": round(time.time() - started, 3),
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            log_row = {
                "song_id": song_id,
                "title": row["title"],
                "status": "error",
                "frames": 0,
                "duration_sec": "",
                "mert_chunks": "",
                "feature_seconds": "",
                "inference_seconds": "",
                "total_seconds": round(time.time() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        append_log(log_path, log_row)
        print(f"{index}/{len(rows)} {song_id} {row['title']} status={log_row['status']} frames={log_row['frames']} seconds={log_row['total_seconds']}", flush=True)
        if log_row["status"] == "error":
            raise RuntimeError(log_row["error"])

    combined_path = args.output_dir / "blind_sample_15_frame_va.csv"
    pd.concat(all_outputs, ignore_index=True).to_csv(combined_path, index=False)
    run_manifest = {
        "created_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "protocol": "Stage 3 audio-only inference with frozen final clean VA teachers",
        "songs": len(completed),
        "frames": int(sum(row["frames"] for row in completed)),
        "elapsed_sec": round(time.time() - started_all, 3),
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": sha256_file(args.manifest),
        "teacher_hashes": teacher_hashes,
        "model_ids": MODEL_IDS,
        "feature_backbone": FEATURE_BACKBONE,
        "frame_hop_sec": args.hop_sec,
        "teacher_window_sec": args.teacher_window_sec,
        "teacher_stride_sec": args.teacher_stride_sec,
        "teacher_window_blend": "Hann weighted overlap-add",
        "section_boundaries_read": False,
        "section_types_read": False,
        "human_va_answers_read": False,
        "combined_output": str(combined_path),
        "combined_output_sha256": sha256_file(combined_path),
        "completed": completed,
    }
    (args.output_dir / "stage3_inference_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
