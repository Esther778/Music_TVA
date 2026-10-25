#!/usr/bin/env python3
"""Extract PMEmo-aligned MuQ embeddings without reading the locked test split."""

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
DEFAULT_SPLIT_INDEX = (
    ROOT
    / "data"
    / "splits"
    / "pmemo_final_locked_20260713"
    / "devfolds_3fold_20260713"
    / "fold_01"
    / "sequence_index.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "muq_pmemo_locked_development"


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_rows(manifest_path: Path, split_index_path: Path, limit: int | None) -> list[dict[str, Any]]:
    manifest = pd.read_csv(manifest_path)
    split_index = pd.read_csv(split_index_path)[["musicId", "split"]]
    if not set(split_index["split"].unique()).issubset({"train", "val"}):
        raise ValueError("Split index must contain development train/val rows only.")
    manifest = manifest.merge(split_index, on="musicId", how="inner")
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
    return rows if limit is None else rows[:limit]


def pool_hidden_to_targets(
    hidden: np.ndarray,
    hidden_times: np.ndarray,
    target_times: np.ndarray,
    pool_window_sec: float,
) -> np.ndarray:
    pooled = np.empty((len(target_times), hidden.shape[1]), dtype=np.float32)
    half = float(pool_window_sec) / 2.0
    for index, target_time in enumerate(target_times):
        use = (hidden_times >= target_time - half) & (hidden_times < target_time + half)
        if np.any(use):
            pooled[index] = hidden[use].mean(axis=0)
        else:
            pooled[index] = hidden[int(np.argmin(np.abs(hidden_times - target_time)))]
    return pooled


@torch.no_grad()
def extract_song(
    audio_path: Path,
    target_times: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    layers: list[int],
    chunk_sec: float,
    overlap_sec: float,
    pool_window_sec: float,
    pre_roll_sec: float,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    sample_rate = 24000
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration_sec = len(audio) / sample_rate
    times = target_times[target_times < duration_sec]
    if len(times) == 0:
        raise RuntimeError(f"No label times inside audio duration: {audio_path}")
    process_start = max(0.0, float(times.min()) - float(pre_roll_sec))
    process_end = min(duration_sec, float(times.max()) + pool_window_sec)
    step_sec = max(float(chunk_sec) - float(overlap_sec), 1.0)
    layer_sums: dict[int, np.ndarray | None] = {layer: None for layer in layers}
    counts = np.zeros(len(times), dtype=np.int32)
    chunks = 0
    chunk_start = process_start
    label_rate = float(getattr(model.config, "label_rate", 25.0))

    while chunk_start < process_end:
        chunk_end = min(chunk_start + float(chunk_sec), process_end)
        start_sample = int(round(chunk_start * sample_rate))
        end_sample = int(round(chunk_end * sample_rate))
        waveform = torch.from_numpy(audio[start_sample:end_sample]).float().unsqueeze(0).to(device)
        if waveform.shape[1] < sample_rate:
            waveform = torch.nn.functional.pad(waveform, (0, sample_rate - waveform.shape[1]))
        outputs = model(waveform, output_hidden_states=True)
        hidden_times = chunk_start + np.arange(outputs.hidden_states[0].shape[1], dtype=np.float64) / label_rate
        in_chunk = np.where((times >= chunk_start) & (times < chunk_end))[0]
        for layer in layers:
            hidden = outputs.hidden_states[layer].squeeze(0).detach().cpu().numpy().astype(np.float32)
            pooled = pool_hidden_to_targets(hidden, hidden_times, times[in_chunk], pool_window_sec)
            if layer_sums[layer] is None:
                layer_sums[layer] = np.zeros((len(times), hidden.shape[1]), dtype=np.float64)
            layer_sums[layer][in_chunk] += pooled
        counts[in_chunk] += 1
        chunks += 1
        chunk_start += step_sec

    if np.any(counts == 0):
        missing = np.where(counts == 0)[0]
        raise RuntimeError(f"Uncovered target frames: {missing[:10].tolist()}")
    embeddings = {
        layer: (layer_sums[layer] / counts[:, None]).astype(np.float32)
        for layer in layers
        if layer_sums[layer] is not None
    }
    return times.astype(np.float32), embeddings, {
        "duration_sec": duration_sec,
        "process_start_sec": process_start,
        "process_end_sec": process_end,
        "chunks": chunks,
        "frames": len(times),
    }


def append_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "musicId", "split", "status", "frames", "dim", "chunks",
        "duration_sec", "seconds", "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-index", type=Path, default=DEFAULT_SPLIT_INDEX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-name", default="OpenMuQ/MuQ-large-msd-iter")
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 9, 12])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-sec", type=float, default=60.0)
    parser.add_argument("--overlap-sec", type=float, default=1.0)
    parser.add_argument("--pool-window-sec", type=float, default=0.5)
    parser.add_argument("--pre-roll-sec", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from muq import MuQ

    device = resolve_device(args.device)
    model = MuQ.from_pretrained(args.model_name, local_files_only=True).to(device).eval()
    if min(args.layers) < 0 or max(args.layers) > int(model.config.encoder_depth):
        raise ValueError(f"Layer indexes out of range: {args.layers}")
    rows = build_rows(args.manifest, args.split_index, args.limit)
    for layer in args.layers:
        (args.out_dir / f"layer_{layer:02d}" / "features").mkdir(parents=True, exist_ok=True)
    audit_path = args.out_dir / "feature_audit.csv"
    print(
        json.dumps(
            {"songs": len(rows), "device": str(device), "layers": args.layers, "model": args.model_name},
            ensure_ascii=False,
        ),
        flush=True,
    )
    audit_rows: list[dict[str, Any]] = []
    started_all = time.time()
    for position, row in enumerate(rows, start=1):
        sid = int(row["musicId"])
        paths = {
            layer: args.out_dir / f"layer_{layer:02d}" / "features" / f"{sid}.npz"
            for layer in args.layers
        }
        started = time.time()
        if all(path.exists() for path in paths.values()) and not args.force:
            status = "skipped_exists"
            sample = np.load(next(iter(paths.values())))
            frames, dim = sample["embedding"].shape
            meta = {"chunks": "", "duration_sec": ""}
            error = ""
        else:
            try:
                times, embeddings, meta = extract_song(
                    Path(row["audio_path"]),
                    row["time_sec"],
                    model,
                    device,
                    args.layers,
                    args.chunk_sec,
                    args.overlap_sec,
                    args.pool_window_sec,
                    args.pre_roll_sec,
                )
                for layer, embedding in embeddings.items():
                    np.savez_compressed(
                        paths[layer],
                        musicId=sid,
                        split=row["split"],
                        time_sec=times,
                        embedding=embedding,
                        model_name=args.model_name,
                        layer=f"hidden_{layer}",
                        label_rate=float(model.config.label_rate),
                        pooling="target_window_mean",
                        pool_window_sec=float(args.pool_window_sec),
                        pre_roll_sec=float(args.pre_roll_sec),
                        audio_path=row["audio_path"],
                    )
                status = "ok"
                frames, dim = next(iter(embeddings.values())).shape
                error = ""
            except Exception as exc:  # noqa: BLE001
                status = "error"
                frames = dim = 0
                meta = {"chunks": "", "duration_sec": ""}
                error = f"{type(exc).__name__}: {exc}"
        seconds = time.time() - started
        audit_rows.append(
            {
                "musicId": sid,
                "split": row["split"],
                "status": status,
                "frames": int(frames),
                "dim": int(dim),
                "chunks": meta.get("chunks", ""),
                "duration_sec": meta.get("duration_sec", ""),
                "seconds": round(seconds, 3),
                "error": error,
            }
        )
        print(
            f"{position}/{len(rows)} musicId={sid} status={status} frames={frames} "
            f"dim={dim} seconds={seconds:.2f}",
            flush=True,
        )
        if len(audit_rows) >= 10:
            append_audit(audit_path, audit_rows)
            audit_rows = []
    if audit_rows:
        append_audit(audit_path, audit_rows)
    print(json.dumps({"elapsed_sec": time.time() - started_all, "songs": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
