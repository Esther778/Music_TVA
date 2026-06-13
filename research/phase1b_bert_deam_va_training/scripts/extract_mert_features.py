#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def import_transformers():
    try:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        import transformers
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install with "
            '".venv/bin/python -m pip install \"transformers<=4.44.0\"" before running MERT extraction.'
        ) from exc
    return AutoModel, Wav2Vec2FeatureExtractor, transformers.__version__


def resolve_device(configured: str) -> torch.device:
    if configured != "auto":
        return torch.device(configured)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mert(config: dict[str, Any], device: torch.device):
    AutoModel, Wav2Vec2FeatureExtractor, transformers_version = import_transformers()
    model = AutoModel.from_pretrained(config["model_name"], trust_remote_code=bool(config["trust_remote_code"]))
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        config["model_name"],
        trust_remote_code=bool(config["trust_remote_code"]),
    )
    model.to(device)
    model.eval()
    return model, processor, transformers_version


def target_times(config: dict[str, Any], duration_sec: float | None = None) -> np.ndarray:
    start = float(config["label_start_ms"]) / 1000.0
    hop = float(config["label_hop_ms"]) / 1000.0
    frames = int(config["label_frames"])
    times = start + np.arange(frames, dtype=np.float32) * hop
    if duration_sec is None:
        return times
    return times[times < duration_sec]


def choose_hidden(outputs: Any, layer: str | int) -> torch.Tensor:
    if isinstance(layer, int):
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError("Requested hidden layer, but model output has no hidden_states.")
        return outputs.hidden_states[layer]
    if layer == "last":
        return outputs.last_hidden_state
    if layer.startswith("hidden_"):
        index = int(layer.split("_", 1)[1])
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError("Requested hidden layer, but model output has no hidden_states.")
        return outputs.hidden_states[index]
    raise ValueError(f"Unsupported layer spec: {layer}")


def pool_chunk_to_targets(
    hidden: np.ndarray,
    chunk_start_sec: float,
    chunk_duration_sec: float,
    all_target_times: np.ndarray,
    pool_window_sec: float,
    sums: np.ndarray | None,
    counts: np.ndarray,
) -> np.ndarray:
    hidden_frames, hidden_dim = hidden.shape
    if sums is None:
        sums = np.zeros((len(all_target_times), hidden_dim), dtype=np.float64)
    if hidden_frames == 0:
        return sums
    hidden_times = chunk_start_sec + np.linspace(0.0, chunk_duration_sec, hidden_frames, endpoint=False)
    half_window = pool_window_sec / 2.0
    in_chunk = np.where(
        (all_target_times >= chunk_start_sec)
        & (all_target_times < chunk_start_sec + chunk_duration_sec)
    )[0]
    for target_index in in_chunk:
        target_time = float(all_target_times[target_index])
        mask = (hidden_times >= target_time - half_window) & (hidden_times < target_time + half_window)
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
    model: torch.nn.Module,
    processor: Any,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sample_rate = int(getattr(processor, "sampling_rate", None) or config["target_sample_rate"])
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration_sec = len(audio) / sample_rate
    all_label_times = target_times(config)
    times = target_times(config, duration_sec)
    if len(times) == 0:
        raise RuntimeError(
            f"Audio is too short for DEAM dynamic labels: duration={duration_sec:.3f}s "
            f"label_start={all_label_times[0]:.3f}s"
        )

    chunk_sec = float(config["chunk_sec"])
    overlap_sec = float(config["chunk_overlap_sec"])
    step_sec = max(chunk_sec - overlap_sec, 1.0)
    pool_window_sec = float(config["pool_window_sec"])
    layer = config["layer"]
    wants_hidden_states = isinstance(layer, str) and layer.startswith("hidden_")

    sums: np.ndarray | None = None
    counts = np.zeros(len(times), dtype=np.int32)
    chunks_processed = 0

    chunk_start = 0.0
    while chunk_start < duration_sec:
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
        sums = pool_chunk_to_targets(
            hidden,
            chunk_start,
            chunk_end - chunk_start,
            times,
            pool_window_sec,
            sums,
            counts,
        )
        chunks_processed += 1
        chunk_start += step_sec

    if sums is None:
        raise RuntimeError(f"No model frames produced for {audio_path}")
    missing = counts == 0
    missing_filled = 0
    if np.any(missing):
        valid = np.where(~missing)[0]
        if len(valid) == 0:
            raise RuntimeError(f"No target label frames covered for {audio_path}")
        for idx in np.where(missing)[0]:
            nearest_valid = valid[int(np.argmin(np.abs(valid - idx)))]
            sums[idx] = sums[nearest_valid]
            counts[idx] = counts[nearest_valid]
            missing_filled += 1
    embedding = (sums / counts[:, None]).astype(np.float32)
    meta = {
        "duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "chunks_processed": chunks_processed,
        "label_total_frames": len(all_label_times),
        "label_frames_extracted": len(times),
        "dropped_label_frames_outside_audio": int(len(all_label_times) - len(times)),
        "missing_target_frames_filled": missing_filled,
    }
    return times.astype(np.float32), embedding, meta


def load_manifest(path: Path, split: str, limit: int | None) -> list[dict[str, str]]:
    rows = pd.read_csv(path).to_dict("records")
    if split:
        rows = [row for row in rows if row["split"] == split]
    rows = sorted(rows, key=lambda row: int(row["song_id"]))
    if limit is not None:
        rows = rows[:limit]
    return [{key: str(value) for key, value in row.items()} for row in rows]


def append_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "song_id",
        "split",
        "status",
        "feature_path",
        "frames",
        "dim",
        "duration_sec",
        "sample_rate",
        "chunks_processed",
        "missing_target_frames_filled",
        "dropped_label_frames_outside_audio",
        "elapsed_sec",
        "error",
    ]
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MERT/BERT-style DEAM features aligned to 0.5s VA labels.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "stage2_mert_features.json"))
    parser.add_argument("--manifest", default=str(ROOT / "data" / "manifests" / "split_manifest.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "features_mert"))
    parser.add_argument("--audit", default=str(ROOT / "outputs" / "feature_audit.csv"))
    parser.add_argument("--split", choices=["", "train", "val", "test"], default="")
    parser.add_argument("--limit", type=int, default=None, help="Small smoke-test limit.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    device = resolve_device(str(config["device"]))
    model, processor, transformers_version = load_mert(config, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(Path(args.manifest), args.split, args.limit)
    audit_rows = []
    started_all = time.time()
    for row in manifest_rows:
        started = time.time()
        song_id = row["song_id"]
        out_path = output_dir / f"{song_id}.npz"
        if out_path.is_file() and not args.force:
            audit_rows.append(
                {
                    "song_id": song_id,
                    "split": row["split"],
                    "status": "skipped_existing",
                    "feature_path": str(out_path),
                    "frames": "",
                    "dim": "",
                    "duration_sec": "",
                    "sample_rate": "",
                    "chunks_processed": "",
                    "missing_target_frames_filled": "",
                    "dropped_label_frames_outside_audio": "",
                    "elapsed_sec": round(time.time() - started, 3),
                    "error": "",
                }
            )
            continue
        try:
            times, embedding, meta = extract_song(Path(row["audio_path"]), model, processor, config, device)
            np.savez_compressed(
                out_path,
                song_id=song_id,
                split=row["split"],
                time_sec=times,
                embedding=embedding,
                model_name=config["model_name"],
                transformers_version=transformers_version,
                sample_rate=meta["sample_rate"],
                layer=config["layer"],
                pooling=config["pooling"],
                pool_window_sec=config["pool_window_sec"],
            )
            audit_rows.append(
                {
                    "song_id": song_id,
                    "split": row["split"],
                    "status": "ok",
                    "feature_path": str(out_path),
                    "frames": embedding.shape[0],
                    "dim": embedding.shape[1],
                    "duration_sec": round(float(meta["duration_sec"]), 3),
                    "sample_rate": meta["sample_rate"],
                    "chunks_processed": meta["chunks_processed"],
                    "missing_target_frames_filled": meta["missing_target_frames_filled"],
                    "dropped_label_frames_outside_audio": meta["dropped_label_frames_outside_audio"],
                    "elapsed_sec": round(time.time() - started, 3),
                    "error": "",
                }
            )
            print(f"ok song_id={song_id} frames={embedding.shape[0]} dim={embedding.shape[1]}")
        except Exception as exc:
            audit_rows.append(
                {
                    "song_id": song_id,
                    "split": row["split"],
                    "status": "error",
                    "feature_path": str(out_path),
                    "frames": "",
                    "dim": "",
                    "duration_sec": "",
                    "sample_rate": "",
                    "chunks_processed": "",
                    "missing_target_frames_filled": "",
                    "dropped_label_frames_outside_audio": "",
                    "elapsed_sec": round(time.time() - started, 3),
                    "error": repr(exc),
                }
            )
            print(f"error song_id={song_id} {exc}")
        append_audit(Path(args.audit), audit_rows)
        audit_rows = []

    print(f"processed={len(manifest_rows)} elapsed_sec={round(time.time() - started_all, 3)}")


if __name__ == "__main__":
    main()
