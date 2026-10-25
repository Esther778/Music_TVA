#!/usr/bin/env python3
"""Extract PMEmo-aligned MERT embeddings from PMEmo chorus audio."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "pmemo2019_dynamic_frame_manifest_normalized.csv"
DEFAULT_SPLIT_INDEX = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence_index.csv"
DEFAULT_OUT = ROOT / "outputs" / "mert_pmemo" / "features"
DEFAULT_AUDIT = ROOT / "outputs" / "mert_pmemo" / "feature_audit.csv"


def import_transformers():
    try:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        import transformers
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: transformers. Install transformers<=4.44.0 first.") from exc
    return AutoModel, Wav2Vec2FeatureExtractor, transformers.__version__


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mert(model_name: str, device: torch.device, local_files_only: bool):
    AutoModel, Wav2Vec2FeatureExtractor, transformers_version = import_transformers()
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
    model.to(device)
    model.eval()
    return model, processor, transformers_version


def choose_hidden(outputs: Any, layer: str | int) -> torch.Tensor:
    if isinstance(layer, int):
        return outputs.hidden_states[layer]
    if layer == "last":
        return outputs.last_hidden_state
    if layer.startswith("hidden_"):
        return outputs.hidden_states[int(layer.split("_", 1)[1])]
    raise ValueError(f"Unsupported layer spec: {layer}")


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
        if not np.any(mask):
            nearest = int(np.argmin(np.abs(hidden_times - target_time)))
            pooled = hidden[nearest]
        else:
            pooled = hidden[mask].mean(axis=0)
        sums[target_index] += pooled
        counts[target_index] += 1
    return sums


@torch.no_grad()
def extract_song(
    audio_path: Path,
    target_times: np.ndarray,
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    chunk_sec: float,
    overlap_sec: float,
    pool_window_sec: float,
    layer: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sample_rate = int(getattr(processor, "sampling_rate", None) or 24000)
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration_sec = len(audio) / sample_rate
    times = target_times[target_times < duration_sec]
    if len(times) == 0:
        raise RuntimeError(f"No PMEmo label times inside audio duration: {audio_path}")
    step_sec = max(chunk_sec - overlap_sec, 1.0)
    wants_hidden_states = layer.startswith("hidden_")
    sums: np.ndarray | None = None
    counts = np.zeros(len(times), dtype=np.int32)
    chunks = 0
    chunk_start = 0.0
    process_until_sec = min(duration_sec, float(np.max(times)) + pool_window_sec + overlap_sec)
    while chunk_start < process_until_sec:
        chunk_end = min(chunk_start + chunk_sec, duration_sec)
        start_sample = int(chunk_start * sample_rate)
        end_sample = int(chunk_end * sample_rate)
        chunk = audio[start_sample:end_sample]
        if len(chunk) == 0:
            break
        inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs, output_hidden_states=wants_hidden_states)
        hidden = choose_hidden(outputs, layer).squeeze(0).detach().cpu().numpy().astype(np.float32)
        sums = pool_chunk_to_targets(hidden, chunk_start, chunk_end - chunk_start, times, pool_window_sec, sums, counts)
        chunks += 1
        chunk_start += step_sec
    if sums is None:
        raise RuntimeError(f"No MERT frames produced for {audio_path}")
    missing = counts == 0
    missing_filled = 0
    if np.any(missing):
        valid = np.where(~missing)[0]
        if len(valid) == 0:
            raise RuntimeError(f"No target frames covered for {audio_path}")
        for idx in np.where(missing)[0]:
            nearest_valid = valid[int(np.argmin(np.abs(valid - idx)))]
            sums[idx] = sums[nearest_valid]
            counts[idx] = counts[nearest_valid]
            missing_filled += 1
    embedding = (sums / counts[:, None]).astype(np.float32)
    meta = {
        "duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "chunks_processed": chunks,
        "target_frames": int(len(target_times)),
        "frames_extracted": int(len(times)),
        "dropped_frames_outside_audio": int(len(target_times) - len(times)),
        "missing_target_frames_filled": int(missing_filled),
    }
    return times.astype(np.float32), embedding, meta


def build_rows(manifest_path: Path, split_index_path: Path, split: str, limit: int | None) -> list[dict[str, Any]]:
    manifest = pd.read_csv(manifest_path)
    split_index = pd.read_csv(split_index_path)[["musicId", "split"]]
    manifest = manifest.merge(split_index, on="musicId", how="left")
    if split:
        manifest = manifest[manifest["split"] == split]
    rows = []
    for music_id, group in manifest.groupby("musicId", sort=True):
        group = group.sort_values("time_sec")
        rows.append(
            {
                "musicId": int(music_id),
                "split": str(group["split"].iloc[0]),
                "audio_path": str(group["audio_path"].iloc[0]),
                "time_sec": group["time_sec"].to_numpy(dtype=np.float32),
            }
        )
    if limit is not None:
        rows = rows[:limit]
    return rows


def append_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["musicId", "split", "status", "feature_path", "frames", "dim", "duration_sec", "chunks_processed", "seconds", "error"]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-index", type=Path, default=DEFAULT_SPLIT_INDEX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--model-name", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-sec", type=float, default=30.0)
    parser.add_argument("--overlap-sec", type=float, default=1.0)
    parser.add_argument("--pool-window-sec", type=float, default=0.5)
    parser.add_argument("--layer", default="last")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model, processor, transformers_version = load_mert(args.model_name, device, local_files_only=not args.allow_download)
    rows = build_rows(args.manifest, args.split_index, args.split, args.limit)
    print(json.dumps({"songs_to_process": len(rows), "device": str(device), "transformers": transformers_version}, ensure_ascii=False), flush=True)
    audit_rows = []
    started_all = time.time()
    for idx, row in enumerate(rows, start=1):
        sid = int(row["musicId"])
        feature_path = args.out_dir / f"{sid}.npz"
        started = time.time()
        if feature_path.exists() and not args.force:
            status = "skipped_exists"
            frames = int(np.load(feature_path)["embedding"].shape[0])
            dim = int(np.load(feature_path)["embedding"].shape[1])
            meta = {"duration_sec": "", "chunks_processed": ""}
            err = ""
        else:
            try:
                times, embedding, meta = extract_song(
                    Path(row["audio_path"]),
                    row["time_sec"],
                    model,
                    processor,
                    device,
                    args.chunk_sec,
                    args.overlap_sec,
                    args.pool_window_sec,
                    args.layer,
                )
                np.savez_compressed(
                    feature_path,
                    musicId=sid,
                    split=row["split"],
                    time_sec=times,
                    embedding=embedding.astype(np.float32),
                    model_name=args.model_name,
                    transformers_version=transformers_version,
                    layer=args.layer,
                    pooling="target_window_mean",
                    pool_window_sec=float(args.pool_window_sec),
                    audio_path=row["audio_path"],
                )
                status = "ok"
                frames = int(embedding.shape[0])
                dim = int(embedding.shape[1])
                err = ""
            except Exception as exc:  # noqa: BLE001
                status = "error"
                frames = 0
                dim = 0
                meta = {"duration_sec": "", "chunks_processed": ""}
                err = f"{type(exc).__name__}: {exc}"
        seconds = time.time() - started
        audit_rows.append(
            {
                "musicId": sid,
                "split": row["split"],
                "status": status,
                "feature_path": str(feature_path),
                "frames": frames,
                "dim": dim,
                "duration_sec": meta.get("duration_sec", ""),
                "chunks_processed": meta.get("chunks_processed", ""),
                "seconds": round(seconds, 3),
                "error": err,
            }
        )
        print(f"{idx}/{len(rows)} musicId={sid} status={status} frames={frames} dim={dim} seconds={seconds:.1f}", flush=True)
        if len(audit_rows) >= 10:
            append_audit(args.audit, audit_rows)
            audit_rows = []
    if audit_rows:
        append_audit(args.audit, audit_rows)
    print(json.dumps({"done": len(rows), "elapsed_sec": round(time.time() - started_all, 3)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
