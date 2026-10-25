from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
MERT_ROOT = PROJECT_ROOT / "research" / "phase1b_bert_deam_va_training"


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def target_times(config: dict, duration_sec: float) -> np.ndarray:
    start = float(config["label_start_ms"]) / 1000.0
    hop = float(config["label_hop_ms"]) / 1000.0
    frames = int(config["label_frames"])
    times = start + np.arange(frames, dtype=np.float32) * hop
    return times[times < duration_sec]


def pool_layer(
    hidden: np.ndarray,
    chunk_start_sec: float,
    chunk_duration_sec: float,
    target: np.ndarray,
    pool_window_sec: float,
    sums: np.ndarray,
    counts: np.ndarray,
) -> None:
    hidden_times = chunk_start_sec + np.linspace(
        0.0, chunk_duration_sec, len(hidden), endpoint=False
    )
    half_window = pool_window_sec / 2.0
    target_indices = np.where(
        (target >= chunk_start_sec)
        & (target < chunk_start_sec + chunk_duration_sec)
    )[0]
    for target_index in target_indices:
        time_sec = float(target[target_index])
        mask = (hidden_times >= time_sec - half_window) & (
            hidden_times < time_sec + half_window
        )
        if np.any(mask):
            pooled = hidden[mask].mean(axis=0)
        else:
            pooled = hidden[int(np.argmin(np.abs(hidden_times - time_sec)))]
        sums[target_index] += pooled
        counts[target_index] += 1


@torch.no_grad()
def extract_song(
    audio_path: Path,
    model: torch.nn.Module,
    processor,
    config: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    sample_rate = int(processor.sampling_rate)
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration_sec = len(audio) / sample_rate
    times = target_times(config, duration_sec)
    layers = [int(layer) for layer in config["layers"]]
    hidden_size = int(model.config.hidden_size)
    sums = np.zeros(
        (len(layers), len(times), hidden_size), dtype=np.float64
    )
    counts = np.zeros((len(layers), len(times)), dtype=np.int32)
    chunk_sec = float(config["chunk_sec"])
    overlap_sec = float(config["chunk_overlap_sec"])
    step_sec = max(chunk_sec - overlap_sec, 1.0)
    chunk_start = 0.0
    while chunk_start < duration_sec:
        chunk_end = min(chunk_start + chunk_sec, duration_sec)
        start_sample = int(chunk_start * sample_rate)
        end_sample = int(chunk_end * sample_rate)
        chunk = audio[start_sample:end_sample]
        inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        if len(outputs.hidden_states) <= max(layers):
            raise ValueError(
                f"Requested layer {max(layers)}, model returned {len(outputs.hidden_states)} states."
            )
        for layer_index, hidden_index in enumerate(layers):
            hidden = (
                outputs.hidden_states[hidden_index]
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            pool_layer(
                hidden,
                chunk_start,
                chunk_end - chunk_start,
                times,
                float(config["pool_window_sec"]),
                sums[layer_index],
                counts[layer_index],
            )
        chunk_start += step_sec

    for layer_index in range(len(layers)):
        missing = counts[layer_index] == 0
        valid = np.where(~missing)[0]
        if len(valid) == 0:
            raise ValueError(f"No pooled frames for {audio_path}.")
        for missing_index in np.where(missing)[0]:
            nearest = valid[int(np.argmin(np.abs(valid - missing_index)))]
            sums[layer_index, missing_index] = sums[layer_index, nearest]
            counts[layer_index, missing_index] = counts[layer_index, nearest]
    embeddings = sums / counts[:, :, None]
    return times, embeddings.transpose(1, 0, 2).astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract selected MERT hidden layers in one forward pass.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "mert_multilayer_features.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MERT_ROOT / "data" / "manifests" / "split_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "mert_multilayer",
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test", "train_val"], default="train_val"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    device = resolve_device(config["device"])
    model = AutoModel.from_pretrained(
        config["model_name"],
        trust_remote_code=bool(config["trust_remote_code"]),
        local_files_only=True,
    ).to(device)
    if bool(config.get("truncate_at_max_layer", False)):
        max_layer = max(int(layer) for layer in config["layers"])
        if not hasattr(model.encoder, "layers"):
            raise ValueError("MERT encoder does not expose a layers collection.")
        model.encoder.layers = model.encoder.layers[:max_layer]
        model.config.num_hidden_layers = max_layer
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        config["model_name"],
        trust_remote_code=bool(config["trust_remote_code"]),
        local_files_only=True,
    )
    model.eval()
    manifest = pd.read_csv(args.manifest)
    if args.split == "train_val":
        manifest = manifest[manifest["split"].isin(["train", "val"])]
    else:
        manifest = manifest[manifest["split"] == args.split]
    manifest = manifest.sort_values("song_id")
    if args.limit is not None:
        manifest = manifest.head(args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = 0
    skipped = 0
    for index, row in enumerate(manifest.to_dict("records"), start=1):
        output = args.output_dir / f"{int(row['song_id'])}.npz"
        if output.is_file() and not args.force:
            skipped += 1
            continue
        times, embeddings = extract_song(
            Path(row["audio_path"]), model, processor, config, device
        )
        np.savez_compressed(
            output,
            song_id=int(row["song_id"]),
            split=row["split"],
            time_sec=times,
            layers=np.asarray(config["layers"], dtype=np.int16),
            embeddings=embeddings,
            model_name=config["model_name"],
        )
        completed += 1
        elapsed = time.time() - started
        print(
            f"[{index}/{len(manifest)}] song_id={int(row['song_id'])} "
            f"split={row['split']} shape={embeddings.shape} "
            f"elapsed_sec={elapsed:.1f}",
            flush=True,
        )
    print(
        json.dumps(
            {
                "requested": len(manifest),
                "completed": completed,
                "skipped": skipped,
                "layers": config["layers"],
                "truncate_at_max_layer": bool(
                    config.get("truncate_at_max_layer", False)
                ),
                "device": str(device),
                "elapsed_sec": time.time() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
