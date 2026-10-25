#!/usr/bin/env python3
"""Train an audio-extractable PMEmo Arousal model.

Unlike the OpenSMILE baseline, this model uses features we can extract from any
MP3 with librosa, so it can be packaged for complete-song inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "pmemo2019_dynamic_frame_manifest_normalized.csv"
DEFAULT_OUT = ROOT / "outputs" / "arousal_fullsong_librosa"

DEFAULT_CONFIG = {
    "sample_rate": 22050,
    "n_fft": 1024,
    "hop_length": 512,
    "n_mfcc": 20,
    "pool_context_sec": 2.0,
    "tempogram_bpms": [40, 60, 80, 100, 120, 160, 200, 240],
    "seed": 42,
    "batch_size": 24,
    "epochs": 120,
    "patience": 14,
    "learning_rate": 8e-4,
    "weight_decay": 1e-4,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.25,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pearson(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 2 or np.std(pred) < 1e-8 or np.std(target) < 1e-8:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def ccc(pred: np.ndarray, target: np.ndarray) -> float:
    mean_x = float(np.mean(pred))
    mean_y = float(np.mean(target))
    var_x = float(np.var(pred))
    var_y = float(np.var(target))
    cov = float(np.mean((pred - mean_x) * (target - mean_y)))
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    return float(2 * cov / denom) if denom > 1e-12 else 0.0


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "pcc": pearson(pred, target),
        "ccc": ccc(pred, target),
    }


def split_song_ids(song_ids: list[int], seed: int) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    ids = np.asarray(sorted(song_ids), dtype=np.int64)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    split = {}
    for sid in ids[:n_train]:
        split[int(sid)] = "train"
    for sid in ids[n_train : n_train + n_val]:
        split[int(sid)] = "val"
    for sid in ids[n_train + n_val :]:
        split[int(sid)] = "test"
    return split


def pool_at_targets(feature: np.ndarray, feature_times: np.ndarray, target_times: np.ndarray, context_sec: float) -> np.ndarray:
    half = context_sec / 2.0
    pooled = []
    for target in target_times:
        mask = (feature_times >= target - half) & (feature_times < target + half)
        if not np.any(mask):
            idx = int(np.argmin(np.abs(feature_times - target)))
            values = feature[:, idx : idx + 1]
        else:
            values = feature[:, mask]
        pooled.append(np.concatenate([values.mean(axis=1), values.std(axis=1)]))
    return np.asarray(pooled, dtype=np.float32)


def extract_librosa_features(audio_path: Path, target_times: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    sr = int(config["sample_rate"])
    n_fft = int(config["n_fft"])
    hop = int(config["hop_length"])
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    if len(audio) == 0:
        raise RuntimeError(f"Empty audio: {audio_path}")
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    mag = np.abs(stft)
    harmonic = librosa.effects.harmonic(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=int(config["n_mfcc"]), n_fft=n_fft, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(S=mag, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=mag, sr=sr)
    flatness = librosa.feature.spectral_flatness(S=mag)
    contrast = librosa.feature.spectral_contrast(S=mag, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop)
    chroma = librosa.feature.chroma_stft(S=mag, sr=sr, hop_length=hop)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    if tonnetz.shape[1] != mag.shape[1]:
        tonnetz = librosa.util.fix_length(tonnetz, size=mag.shape[1], axis=1)
    rms = librosa.feature.rms(S=mag, frame_length=n_fft)
    onset = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)[None, :]
    base = np.concatenate([mfcc, centroid, bandwidth, flatness, contrast, zcr, chroma, tonnetz, rms, onset], axis=0).astype(np.float32)
    frame_times = librosa.frames_to_time(np.arange(base.shape[1]), sr=sr, hop_length=hop)
    pooled = pool_at_targets(base, frame_times, target_times, float(config["pool_context_sec"]))
    tempogram = librosa.feature.tempogram(onset_envelope=onset.squeeze(0), sr=sr, hop_length=hop)
    bpm_values = librosa.tempo_frequencies(tempogram.shape[0], sr=sr, hop_length=hop)
    tempo_rows = []
    for bpm in config["tempogram_bpms"]:
        idx = int(np.argmin(np.abs(bpm_values - float(bpm))))
        tempo_rows.append(tempogram[idx])
    tempo = pool_at_targets(np.asarray(tempo_rows, dtype=np.float32), frame_times, target_times, float(config["pool_context_sec"]))
    return np.concatenate([pooled, tempo], axis=1).astype(np.float32)


def build_dataset(manifest_path: Path, out_dir: Path, config: dict[str, Any], force: bool) -> dict[str, Any]:
    dataset_dir = out_dir / "dataset"
    seq_dir = dataset_dir / "sequence"
    seq_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    split_map = split_song_ids(sorted(manifest["musicId"].unique().astype(int).tolist()), int(config["seed"]))
    rows = []
    for idx, (music_id, group) in enumerate(manifest.groupby("musicId", sort=True), start=1):
        split = split_map[int(music_id)]
        path = seq_dir / f"{int(music_id)}.npz"
        if not path.exists() or force:
            group = group.sort_values("time_sec")
            target_times = group["time_sec"].to_numpy(dtype=np.float32)
            audio_path = Path(str(group["audio_path"].iloc[0]))
            X = extract_librosa_features(audio_path, target_times, config)
            y = group["arousal_deam_scale"].to_numpy(dtype=np.float32)
            finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
            if not finite.all():
                X = X[finite]
                y = y[finite]
                target_times = target_times[finite]
            np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.float32), time_sec=target_times, music_id=int(music_id), split=split, audio_path=str(audio_path))
        data = np.load(path)
        rows.append({"musicId": int(music_id), "split": split, "frames": int(len(data["y"])), "path": str(path)})
        if idx % 50 == 0:
            print(f"features prepared {idx}/{manifest['musicId'].nunique()}", flush=True)
    index = pd.DataFrame(rows)
    index.to_csv(dataset_dir / "sequence_index.csv", index=False)
    train_rows = index[index["split"] == "train"]
    train_x = [np.load(row["path"])["X"].astype(np.float32) for row in train_rows.to_dict("records")]
    train_concat = np.concatenate(train_x, axis=0)
    mean = train_concat.mean(axis=0).astype(np.float32)
    std = (train_concat.std(axis=0) + 1e-8).astype(np.float32)
    np.savez(dataset_dir / "feature_norm.npz", mean=mean, std=std)
    summary = {
        "songs": int(len(index)),
        "frames": int(index["frames"].sum()),
        "feature_dim": int(train_concat.shape[1]),
        "splits": {
            split: {
                "songs": int((index["split"] == split).sum()),
                "frames": int(index[index["split"] == split]["frames"].sum()),
            }
            for split in ["train", "val", "test"]
        },
    }
    (dataset_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


class SequenceDataset(Dataset):
    def __init__(self, index: pd.DataFrame, split: str, mean: np.ndarray, std: np.ndarray):
        self.rows = index[index["split"] == split].sort_values("musicId").to_dict("records")
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        data = np.load(self.rows[idx]["path"])
        X = ((data["X"].astype(np.float32) - self.mean) / self.std).astype(np.float32)
        y = data["y"].astype(np.float32)
        return torch.from_numpy(X), torch.from_numpy(y)


def collate(batch):
    lengths = [x.shape[0] for x, _ in batch]
    max_len = max(lengths)
    feat_dim = batch[0][0].shape[1]
    X = torch.zeros(len(batch), max_len, feat_dim)
    y = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (xi, yi) in enumerate(batch):
        X[i, : xi.shape[0]] = xi
        y[i, : yi.shape[0]] = yi
        mask[i, : xi.shape[0]] = True
    return X, y, mask


class ArousalBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.out(y).squeeze(-1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds, targets = [], []
    for X, y, mask in loader:
        X = X.to(device)
        mask = mask.to(device)
        pred = model(X, mask).detach().cpu()
        preds.append(pred[mask.cpu()].numpy())
        targets.append(y[mask.cpu()].numpy())
    pred_np = np.concatenate(preds)
    target_np = np.concatenate(targets)
    return {"metrics": metrics(pred_np, target_np), "pred": pred_np, "target": target_np}


@dataclass
class Result:
    trial_id: str
    best_epoch: int
    val_rmse: float
    test_rmse: float
    test_mae: float
    test_pcc: float
    test_ccc: float
    checkpoint: str


def train(out_dir: Path, config: dict[str, Any], device: torch.device) -> Result:
    dataset_dir = out_dir / "dataset"
    index = pd.read_csv(dataset_dir / "sequence_index.csv")
    norm = np.load(dataset_dir / "feature_norm.npz")
    mean, std = norm["mean"], norm["std"]
    loaders = {
        split: DataLoader(SequenceDataset(index, split, mean, std), batch_size=int(config["batch_size"]), shuffle=(split == "train"), collate_fn=collate)
        for split in ["train", "val", "test"]
    }
    input_dim = int(mean.shape[0])
    model = ArousalBiLSTM(input_dim, int(config["hidden_dim"]), int(config["num_layers"]), float(config["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    ckpt = out_dir / "arousal_bilstm_best.pt"
    best = float("inf")
    best_epoch = 0
    bad = 0
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses = []
        for X, y, mask in loaders["train"]:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(X, mask)
            loss = ((pred - y) ** 2)[mask].mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, loaders["val"], device)["metrics"]
        row = {"epoch": epoch, "train_mse": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(f"arousal_bilstm epoch={epoch:03d} train_mse={row['train_mse']:.5f} val_rmse={val['rmse']:.5f} val_pcc={val['pcc']:.3f} val_ccc={val['ccc']:.3f}", flush=True)
        if val["rmse"] < best:
            best = val["rmse"]
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "config": config, "input_dim": input_dim}, ckpt)
        else:
            bad += 1
            if bad >= int(config["patience"]):
                break
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    split_metrics = {split: evaluate(model, loaders[split], device)["metrics"] for split in ["train", "val", "test"]}
    summary = {
        "trial_id": "pmemo_librosa_arousal_bilstm",
        "model_type": "arousal_bilstm",
        "label_scale": "minus1_1",
        "feature_type": "librosa_handcrafted_extractable_from_audio",
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt),
        **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
    }
    pd.DataFrame(history).to_csv(out_dir / "arousal_bilstm_history.csv", index=False)
    (out_dir / "arousal_bilstm_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return Result("pmemo_librosa_arousal_bilstm", best_epoch, summary["val_rmse"], summary["test_rmse"], summary["test_mae"], summary["test_pcc"], summary["test_ccc"], str(ckpt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()
    config = dict(DEFAULT_CONFIG)
    set_seed(int(config["seed"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = build_dataset(args.manifest, args.out_dir, config, args.force_features)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)
    result = train(args.out_dir, config, device)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
