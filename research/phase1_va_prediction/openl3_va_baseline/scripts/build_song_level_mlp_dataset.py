from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")
LABEL_START_SEC = 15.0
FRAME_HOP_SEC = 0.5


def load_split_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    baseline_root = Path(__file__).resolve().parents[1]
    project_root = baseline_root.parents[2]
    manifest_path = (
        project_root
        / "research"
        / "phase1b_bert_deam_va_training"
        / "data"
        / "manifests"
        / "split_manifest.csv"
    )
    feature_dir = baseline_root / "artifacts" / "openl3_2hz_features"
    label_dir = baseline_root / "artifacts" / "labels_2hz"
    output_dir = (
        baseline_root / "artifacts" / "datasets" / "mlp_song_level_aligned15"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_split_manifest(manifest_path)
    parts = {
        split: {"X": [], "Y": [], "song_id": [], "frame_index": []}
        for split in SPLITS
    }
    song_audit = []
    split_song_ids = {split: set() for split in SPLITS}

    for row in sorted(rows, key=lambda item: int(item["song_id"])):
        song_id = int(row["song_id"])
        split = row["split"]
        if split not in parts:
            raise ValueError(f"Unsupported split {split!r} for song {song_id}.")
        feature_path = feature_dir / f"{song_id}.npy"
        label_path = label_dir / f"{song_id}.npy"
        if not feature_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(
                f"Missing OpenL3 feature or VA label for song {song_id}."
            )

        features = np.load(feature_path).astype(np.float32)
        labels = np.load(label_path).astype(np.float32)
        start_index = int(round(LABEL_START_SEC / FRAME_HOP_SEC))
        aligned_frames = min(len(labels), len(features) - start_index)
        if aligned_frames <= 0:
            raise ValueError(
                f"No OpenL3 frames at or after {LABEL_START_SEC}s for song {song_id}."
            )
        features = features[start_index : start_index + aligned_frames]
        labels = labels[:aligned_frames]
        finite_mask = np.isfinite(features).all(axis=1) & np.isfinite(labels).all(
            axis=1
        )
        dropped_invalid = int(len(finite_mask) - finite_mask.sum())
        features = features[finite_mask]
        labels = labels[finite_mask]
        if len(features) == 0:
            raise ValueError(f"No finite aligned frames for song {song_id}.")

        parts[split]["X"].append(features)
        parts[split]["Y"].append(labels)
        parts[split]["song_id"].append(
            np.full(len(features), song_id, dtype=np.int32)
        )
        parts[split]["frame_index"].append(
            np.arange(len(features), dtype=np.int32)
        )
        split_song_ids[split].add(song_id)
        song_audit.append(
            {
                "song_id": song_id,
                "split": split,
                "feature_frames_raw": int(np.load(feature_path, mmap_mode="r").shape[0]),
                "label_frames_raw": int(np.load(label_path, mmap_mode="r").shape[0]),
                "feature_start_sec": LABEL_START_SEC,
                "feature_end_sec": LABEL_START_SEC
                + (aligned_frames - 1) * FRAME_HOP_SEC,
                "aligned_finite_frames": int(len(features)),
                "dropped_invalid_frames": dropped_invalid,
            }
        )

    overlaps = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlaps[f"{left}_{right}"] = sorted(
                split_song_ids[left] & split_song_ids[right]
            )
    if any(overlaps.values()):
        raise ValueError(f"Song leakage detected: {overlaps}")

    summary = {
        "dataset_id": "openl3_deam_mlp_song_level_aligned15_v2",
        "manifest": str(manifest_path.relative_to(project_root)),
        "label_range": "[-1, 1]",
        "label_start_sec": LABEL_START_SEC,
        "frame_hop_sec": FRAME_HOP_SEC,
        "alignment": (
            "OpenL3 timestamp index 30 (15.0 s) onward paired with labels "
            "starting at sample_15000ms"
        ),
        "splits": {},
        "song_overlap": overlaps,
    }
    for split in SPLITS:
        split_parts = parts[split]
        features = np.concatenate(split_parts["X"], axis=0).astype(np.float32)
        labels = np.concatenate(split_parts["Y"], axis=0).astype(np.float32)
        song_ids = np.concatenate(split_parts["song_id"], axis=0)
        frame_indices = np.concatenate(split_parts["frame_index"], axis=0)
        np.save(output_dir / f"X_{split}.npy", features)
        np.save(output_dir / f"Y_{split}.npy", labels)
        np.save(output_dir / f"song_id_{split}.npy", song_ids)
        np.save(output_dir / f"frame_index_{split}.npy", frame_indices)
        summary["splits"][split] = {
            "songs": len(split_song_ids[split]),
            "frames": int(len(features)),
            "feature_dim": int(features.shape[1]),
            "finite_features": bool(np.isfinite(features).all()),
            "finite_labels": bool(np.isfinite(labels).all()),
        }

    train_features = np.load(output_dir / "X_train.npy", mmap_mode="r")
    mean = np.asarray(train_features.mean(axis=0, keepdims=True), dtype=np.float32)
    std = np.asarray(
        train_features.std(axis=0, keepdims=True) + 1e-8, dtype=np.float32
    )
    np.savez(output_dir / "feature_norm.npz", mean=mean, std=std)

    with (output_dir / "song_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(song_audit[0]))
        writer.writeheader()
        writer.writerows(song_audit)
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
