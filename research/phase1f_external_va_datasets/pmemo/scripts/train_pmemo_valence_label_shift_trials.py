#!/usr/bin/env python3
"""Train Valence models with shifted dynamic labels on the bottleneck split.

Post-hoc temporal shifts improved Valence, so this script tests whether moving
the alignment into training helps the model itself. It creates shifted copies
of the PMEmo dynamic labels and trains the historical hidden6+harmonic BiLSTM
family on split_02 first.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_harmonic_valence_fusion import DEFAULT_LIBROSA_DATASET
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows, resolve_device
from train_pmemo_valence_multisplit_best import (
    DEFAULT_INDEX,
    DEFAULT_LABEL_DIR,
    DEFAULT_MERT_DIR,
    BestConfig,
    make_split_index,
    read_song_target_means,
    train_one_split,
)


OUT_DIR = ROOT / "outputs" / "valence_label_shift_trials_20260708"
SPLIT_ID = 2
SEED = 202

CURRENT_MODEL_BEST = {
    "model_id": "pmemo_valence_multisplit_best_20260627",
    "split_02_rmse": 0.23205068707466125,
    "split_02_mae": 0.1822001338005066,
    "split_02_pcc": 0.7219235532295507,
    "split_02_ccc": 0.7097179959426364,
}

CURRENT_PROJECT_BEST_SPLIT2 = {
    "model_id": "valence_extra_wide_rmse_shift_smoothing_20260708",
    "split_02_rmse": 0.19413441783123941,
    "split_02_mae": 0.15004043091703112,
    "split_02_pcc": 0.7979784711508939,
    "split_02_ccc": 0.7779712051447267,
    "note": "prediction artifact reference, not pure training checkpoint",
}

BASE_CONFIG = BestConfig(
    trial_id="h6_harm_h80_ext115_clip008_ccc006_do025",
    proj_dim=320,
    hidden_dim=80,
    dropout=0.25,
    learning_rate=3e-4,
    weight_decay=3e-4,
    batch_size=20,
    frame_ccc_weight=0.06,
    clip_mse_weight=0.08,
    extreme_weight=1.15,
    clip_std_weight=0.0,
    patience=8,
    epochs=60,
)


def shift_1d(values: np.ndarray, frames: int) -> np.ndarray:
    if frames == 0 or len(values) <= 1:
        return values.copy()
    out = np.empty_like(values)
    if frames > 0:
        out[:-frames] = values[frames:]
        out[-frames:] = values[-1]
    else:
        k = -frames
        out[:k] = values[0]
        out[k:] = values[:-k]
    return out


def build_shifted_label_dir(source_dir: Path, out_dir: Path, shift_frames: int, train_music_ids: set[int]) -> Path:
    label_dir = out_dir / f"labels_shift_{shift_frames:+03d}"
    label_dir.mkdir(parents=True, exist_ok=True)
    done = label_dir / "_DONE.json"
    if done.exists():
        meta = json.loads(done.read_text(encoding="utf-8"))
        if meta.get("train_only_shift") is True:
            return label_dir
        shutil.rmtree(label_dir)
        label_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(source_dir.glob("*.npz")):
        sid = int(src.stem)
        data = np.load(src)
        arrays = {key: data[key] for key in data.files}
        y = arrays["Y"].astype(np.float32).copy()
        if sid in train_music_ids:
            y[:, 0] = shift_1d(y[:, 0], shift_frames)
        arrays["Y"] = y
        np.savez(label_dir / src.name, **arrays)
    done.write_text(
        json.dumps(
            {
                "source": str(source_dir),
                "shift_frames": int(shift_frames),
                "shift_seconds": float(shift_frames * 0.5),
                "train_only_shift": True,
                "train_song_count": len(train_music_ids),
                "meaning": "Only train-split Valence labels are shifted; validation and test labels remain original for comparable metric reporting.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return label_dir


def run_one(shift_frames: int, split_index: pd.DataFrame, device) -> dict[str, Any]:
    train_music_ids = set(split_index.loc[split_index["split"] == "train", "musicId"].astype(int).tolist())
    label_dir = build_shifted_label_dir(DEFAULT_LABEL_DIR, OUT_DIR / "shifted_labels", shift_frames, train_music_ids)
    trial_id = f"{BASE_CONFIG.trial_id}_labelshift_{shift_frames:+03d}"
    config = BestConfig(**{**asdict(BASE_CONFIG), "trial_id": trial_id})
    trial_out = OUT_DIR / trial_id
    trial_out.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        out_dir=trial_out,
        mert_dir=DEFAULT_MERT_DIR,
        label_dir=label_dir,
        librosa_dataset=DEFAULT_LIBROSA_DATASET,
    )
    row = train_one_split(SPLIT_ID, SEED, split_index, args, device, config)
    row["label_shift_frames"] = int(shift_frames)
    row["label_shift_seconds"] = float(shift_frames * 0.5)
    row["trial_id"] = trial_id
    return row


def metrics_from_row(row: dict[str, Any]) -> dict[str, float]:
    return {m: float(row[f"test_{m}"]) for m in ["rmse", "mae", "pcc", "ccc"]}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_index = pd.read_csv(DEFAULT_INDEX)
    song_stats = read_song_target_means(base_index, DEFAULT_LABEL_DIR)
    split_index = make_split_index(song_stats, SEED)
    device = resolve_device("auto")

    rows = []
    for shift_frames in [-24, -18, -12, -6, 0, 6]:
        rows.append(run_one(shift_frames, split_index, device))

    comparison = pd.DataFrame(rows).sort_values(["test_rmse", "test_ccc"], ascending=[True, False])
    comparison.to_csv(OUT_DIR / "comparison.csv", index=False)
    best = comparison.iloc[0].to_dict()
    best_metrics = metrics_from_row(best)
    summary = {
        "created_at": "2026-07-08",
        "purpose": "screen training-time Valence label shifts on bottleneck split_02",
        "split_scope": "split_02 only; expand to all splits only if this improves materially",
        "base_config": asdict(BASE_CONFIG),
        "current_model_best_reference": CURRENT_MODEL_BEST,
        "current_project_best_split2_reference": CURRENT_PROJECT_BEST_SPLIT2,
        "shift_candidates_frames": [-24, -18, -12, -6, 0, 6],
        "trials": rows,
        "best_trial": best,
        "best_trial_gap_rows": threshold_rows(best_metrics),
        "delta_vs_current_model_best_split2": {
            "rmse": best_metrics["rmse"] - CURRENT_MODEL_BEST["split_02_rmse"],
            "mae": best_metrics["mae"] - CURRENT_MODEL_BEST["split_02_mae"],
            "pcc": best_metrics["pcc"] - CURRENT_MODEL_BEST["split_02_pcc"],
            "ccc": best_metrics["ccc"] - CURRENT_MODEL_BEST["split_02_ccc"],
        },
        "delta_vs_current_project_best_split2": {
            "rmse": best_metrics["rmse"] - CURRENT_PROJECT_BEST_SPLIT2["split_02_rmse"],
            "mae": best_metrics["mae"] - CURRENT_PROJECT_BEST_SPLIT2["split_02_mae"],
            "pcc": best_metrics["pcc"] - CURRENT_PROJECT_BEST_SPLIT2["split_02_pcc"],
            "ccc": best_metrics["ccc"] - CURRENT_PROJECT_BEST_SPLIT2["split_02_ccc"],
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
