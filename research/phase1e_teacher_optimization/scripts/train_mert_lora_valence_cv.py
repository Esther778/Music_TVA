from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from cross_validate_mert_layer_mix import (
    INDEX,
    ROOT,
    macro_metrics,
    window_metrics,
)
from train_global_local_temporal_cv import correlation_loss, loss_function
from train_valence_harmonic_context import scalar_metrics


PROJECT_ROOT = ROOT.parents[1]
MANIFEST = (
    PROJECT_ROOT
    / "research/phase1b_bert_deam_va_training/data/manifests/split_manifest.csv"
)
FOLD_MANIFEST = ROOT / "data/five_fold_manifest.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(alpha) / rank
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(values) + self.scale * self.up(
            self.down(self.dropout(values))
        )


def install_lora(model: nn.Module, config: dict) -> list[str]:
    installed = []
    layers = [int(index) for index in config["lora_layers"]]
    targets = [str(name) for name in config["lora_targets"]]
    for layer_index in layers:
        layer = model.encoder.layers[layer_index]
        for target in targets:
            base = getattr(layer.attention, target)
            if not isinstance(base, nn.Linear):
                raise TypeError(
                    f"encoder.layers.{layer_index}.attention.{target} "
                    f"is {type(base).__name__}, expected Linear"
                )
            setattr(
                layer.attention,
                target,
                LoRALinear(
                    base,
                    int(config["lora_rank"]),
                    float(config["lora_alpha"]),
                    float(config["lora_dropout"]),
                ),
            )
            installed.append(
                f"encoder.layers.{layer_index}.attention.{target}"
            )
    return installed


class SongAudioDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        sample_rate: int,
        chunk_frames: int,
        context_sec: float,
        audio_chunk_sec: float | None = None,
        max_frames: int | None = None,
    ):
        self.rows = rows.sort_values("song_id").to_dict("records")
        self.sample_rate = sample_rate
        self.chunk_frames = chunk_frames
        self.context_sec = context_sec
        self.audio_chunk_sec = audio_chunk_sec
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict:
        row = self.rows[item]
        song_id = int(row["song_id"])
        with np.load(row["sequence_path"]) as data:
            times = data["time_sec"].astype(np.float32)
            target = data["Y"][:, :1].astype(np.float32)
        if self.max_frames is not None and len(target) > self.max_frames:
            start = random.randint(0, len(target) - self.max_frames)
            stop = start + self.max_frames
            times = times[start:stop]
            target = target[start:stop]
        duration = float(librosa.get_duration(path=row["audio_path"]))
        chunks = []
        chunk_times = []
        starts = []
        durations = []
        if self.audio_chunk_sec is not None:
            chunk_ids = np.floor(times / self.audio_chunk_sec).astype(int)
            groups = [
                np.where(chunk_ids == chunk_id)[0]
                for chunk_id in np.unique(chunk_ids)
            ]
        else:
            groups = [
                np.arange(frame_start, min(frame_start + self.chunk_frames, len(target)))
                for frame_start in range(0, len(target), self.chunk_frames)
            ]
        for group in groups:
            target_times = times[group]
            if self.audio_chunk_sec is not None:
                audio_start = (
                    math.floor(float(target_times[0]) / self.audio_chunk_sec)
                    * self.audio_chunk_sec
                )
                audio_end = min(duration, audio_start + self.audio_chunk_sec)
            else:
                audio_start = max(
                    0.0, float(target_times[0]) - self.context_sec
                )
                audio_end = min(
                    duration,
                    float(target_times[-1]) + 0.5 + self.context_sec,
                )
            audio, _ = librosa.load(
                row["audio_path"],
                sr=self.sample_rate,
                mono=True,
                offset=audio_start,
                duration=audio_end - audio_start,
            )
            chunks.append(audio.astype(np.float32))
            chunk_times.append(target_times)
            starts.append(audio_start)
            durations.append(len(audio) / self.sample_rate)
        return {
            "song_id": song_id,
            "chunks": chunks,
            "chunk_times": chunk_times,
            "audio_starts": starts,
            "audio_durations": durations,
            "target": target,
        }


class AudioCollator:
    def __init__(
        self,
        processor: Wav2Vec2FeatureExtractor,
        chunk_batch_size: int,
    ):
        self.processor = processor
        self.chunk_batch_size = chunk_batch_size

    def __call__(self, batch: list[dict]) -> dict:
        if len(batch) != 1:
            raise ValueError(
                "Raw-audio MERT adaptation currently requires batch_size=1 "
                "song; use gradient accumulation for a larger effective batch."
            )
        item = batch[0]
        mert_batches = []
        for start in range(0, len(item["chunks"]), self.chunk_batch_size):
            stop = min(start + self.chunk_batch_size, len(item["chunks"]))
            chunks = item["chunks"][start:stop]
            processed = self.processor(
                chunks,
                sampling_rate=self.processor.sampling_rate,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            mert_batches.append(
                {
                    "input_values": processed["input_values"],
                    "attention_mask": processed["attention_mask"],
                    "input_lengths": torch.tensor(
                        [len(chunk) for chunk in chunks], dtype=torch.long
                    ),
                    "chunk_times": item["chunk_times"][start:stop],
                    "audio_starts": item["audio_starts"][start:stop],
                    "audio_durations": item["audio_durations"][start:stop],
                }
            )
        return {
            "mert_batches": mert_batches,
            "target": torch.from_numpy(item["target"]).unsqueeze(0),
            "song_id": item["song_id"],
        }


class GlobalLocalHead(nn.Module):
    def __init__(self, input_dim: int, config: dict):
        super().__init__()
        projection = int(config["projection_dim"])
        hidden = int(config["hidden_dim"])
        dropout = float(config["dropout"])
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Linear(projection, 1)
        self.global_head = nn.Sequential(
            nn.Linear(projection, projection // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection // 2, 1),
        )
        self.local_temporal = nn.LSTM(
            projection, hidden, batch_first=True, bidirectional=True
        )
        self.local_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )
        self.residual_logit = nn.Parameter(torch.tensor(-0.5))

    def forward(self, features: torch.Tensor) -> dict:
        projected = self.project(features)
        weights = torch.softmax(
            self.attention(projected).squeeze(-1), dim=1
        ).unsqueeze(-1)
        global_embedding = (weights * projected).sum(dim=1)
        global_prediction = self.global_head(global_embedding)
        centered = projected - global_embedding.unsqueeze(1)
        local_hidden, _ = self.local_temporal(centered)
        local_raw = self.local_head(local_hidden)
        local_prediction = local_raw - local_raw.mean(dim=1, keepdim=True)
        residual_scale = torch.sigmoid(self.residual_logit)
        prediction = (
            global_prediction.unsqueeze(1)
            + residual_scale * local_prediction
        )
        return {
            "prediction": prediction,
            "global": global_prediction,
            "global_embedding": global_embedding,
            "local": local_prediction,
            "residual_scale": residual_scale,
            "attention": weights,
        }


class MERTLoRAValence(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.mert = AutoModel.from_pretrained(
            config["model_name"],
            trust_remote_code=True,
            local_files_only=True,
        )
        truncate_layers = int(config["truncate_layers"])
        self.mert.encoder.layers = self.mert.encoder.layers[:truncate_layers]
        self.mert.config.num_hidden_layers = truncate_layers
        self.mert.config.layerdrop = 0.0
        for parameter in self.mert.parameters():
            parameter.requires_grad = False
        self.lora_modules = install_lora(self.mert, config)
        self.head = GlobalLocalHead(
            int(self.mert.config.hidden_size), config
        )
        norm_path = PROJECT_ROOT / config["feature_norm"]
        with np.load(norm_path) as data:
            feature_mean = data["mean"].astype(np.float32)
            feature_std = data["std"].astype(np.float32)
        self.register_buffer(
            "feature_mean", torch.from_numpy(feature_mean), persistent=True
        )
        self.register_buffer(
            "feature_std", torch.from_numpy(feature_std), persistent=True
        )
        checkpoint_path = PROJECT_ROOT / config["head_init_checkpoint"]
        head_state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        self.head.load_state_dict(head_state, strict=True)
        self.head_init_checkpoint = str(checkpoint_path)
        self.train_head = bool(config.get("train_head", True))
        if not self.train_head:
            for parameter in self.head.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the frozen pretrained path deterministic. LoRA dropout is zero
        # in the initial experiment, so this does not disable trainable noise.
        self.mert.eval()
        for name, module in self.mert.named_modules():
            if isinstance(module, LoRALinear):
                module.train(mode)
        self.head.train(mode if self.train_head else False)
        return self

    @staticmethod
    def pool_targets(
        hidden: torch.Tensor,
        audio_start: float,
        audio_duration: float,
        target_times: np.ndarray,
    ) -> torch.Tensor:
        hidden_times = audio_start + torch.arange(
            len(hidden), device=hidden.device, dtype=hidden.dtype
        ) * (audio_duration / len(hidden))
        pooled = []
        for target_time in target_times:
            center = float(target_time)
            selected = (hidden_times >= center - 0.25) & (
                hidden_times < center + 0.25
            )
            if bool(selected.any()):
                pooled.append(hidden[selected].mean(dim=0))
            else:
                nearest = torch.argmin(
                    torch.abs(hidden_times - center)
                )
                pooled.append(hidden[nearest])
        return torch.stack(pooled)

    def forward(self, batch: dict) -> dict:
        chunks = []
        for mert_batch in batch["mert_batches"]:
            outputs = self.mert(
                input_values=mert_batch["input_values"],
                attention_mask=mert_batch["attention_mask"],
                output_hidden_states=False,
            )
            hidden = outputs.last_hidden_state
            hidden_lengths = self.mert._get_feat_extract_output_lengths(
                mert_batch["input_lengths"]
            )
            for index in range(len(mert_batch["chunk_times"])):
                valid = hidden[index, : int(hidden_lengths[index])]
                chunks.append(
                    self.pool_targets(
                        valid,
                        float(mert_batch["audio_starts"][index]),
                        float(mert_batch["audio_durations"][index]),
                        mert_batch["chunk_times"][index],
                    )
                )
        sequence = torch.cat(chunks, dim=0).unsqueeze(0)
        sequence = (sequence - self.feature_mean) / self.feature_std
        return self.head(sequence)


def split_parameters(model: MERTLoRAValence):
    lora, head = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".down." in name or ".up." in name:
            lora.append(parameter)
        else:
            head.append(parameter)
    return lora, head


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = dict(batch)
    moved["target"] = batch["target"].to(device)
    moved["mert_batches"] = []
    for mert_batch in batch["mert_batches"]:
        moved_mert = dict(mert_batch)
        for key in ("input_values", "attention_mask", "input_lengths"):
            moved_mert[key] = mert_batch[key].to(device)
        moved["mert_batches"].append(moved_mert)
    return moved


@torch.no_grad()
def evaluate(
    model: MERTLoRAValence,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
):
    model.eval()
    predictions, targets, song_ids = [], [], []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        moved = move_batch(batch, device)
        output = model(moved)
        prediction = output["prediction"][0].cpu().numpy().reshape(-1)
        target = batch["target"][0].numpy().reshape(-1)
        predictions.append(prediction)
        targets.append(target)
        song_ids.append(
            np.full(len(target), int(batch["song_id"]), dtype=np.int32)
        )
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    expanded_ids = np.concatenate(song_ids)
    micro = scalar_metrics(prediction, target)
    macro = macro_metrics(prediction, target, expanded_ids)
    window5 = window_metrics(prediction, target, expanded_ids, 10)
    window10 = window_metrics(prediction, target, expanded_ids, 20)
    centered_prediction, centered_target = [], []
    mean_prediction, mean_target = [], []
    first_prediction, first_target = [], []
    for song_id in np.unique(expanded_ids):
        mask = expanded_ids == song_id
        pred_song = prediction[mask]
        target_song = target[mask]
        centered_prediction.append(pred_song - pred_song.mean())
        centered_target.append(target_song - target_song.mean())
        mean_prediction.append(float(pred_song.mean()))
        mean_target.append(float(target_song.mean()))
        first_prediction.append(np.diff(pred_song))
        first_target.append(np.diff(target_song))
    within = scalar_metrics(
        np.concatenate(centered_prediction), np.concatenate(centered_target)
    )
    means = scalar_metrics(
        np.asarray(mean_prediction), np.asarray(mean_target)
    )
    first = scalar_metrics(
        np.concatenate(first_prediction), np.concatenate(first_target)
    )
    metrics = {
        **{f"micro_{key}": value for key, value in micro.items()},
        **{f"macro_{key}": value for key, value in macro.items()},
        **{f"window5_{key}": value for key, value in window5.items()},
        **{f"window10_{key}": value for key, value in window10.items()},
        **{f"within_{key}": value for key, value in within.items()},
        **{f"song_mean_{key}": value for key, value in means.items()},
        **{f"first_difference_{key}": value for key, value in first.items()},
    }
    return metrics, prediction, target, expanded_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/mert_lora_valence_fold0.json",
    )
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--limit-train-songs", type=int)
    parser.add_argument("--limit-held-songs", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate the zero-initialized LoRA model and pretrained head.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"]) + args.fold
    set_seed(seed)
    device = torch.device(args.device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
        local_files_only=True,
    )

    index = pd.read_csv(INDEX)
    manifest = pd.read_csv(MANIFEST)[["song_id", "audio_path"]]
    folds = pd.read_csv(FOLD_MANIFEST)[["song_id", "fold"]]
    rows = (
        index[index["split"] == "train"]
        .merge(manifest, on="song_id")
        .merge(folds, on="song_id")
    )
    train_rows = rows[rows["fold"] != args.fold].sort_values("song_id")
    held_rows = rows[rows["fold"] == args.fold].sort_values("song_id")
    if args.limit_train_songs:
        train_rows = train_rows.head(args.limit_train_songs)
    if args.limit_held_songs:
        held_rows = held_rows.head(args.limit_held_songs)

    datasets = {
        "train": SongAudioDataset(
            train_rows,
            int(processor.sampling_rate),
            int(config["chunk_frames"]),
            float(config["context_sec"]),
            float(config["audio_chunk_sec"]),
            int(config["max_train_frames"]),
        ),
        "held": SongAudioDataset(
            held_rows,
            int(processor.sampling_rate),
            int(config["chunk_frames"]),
            float(config["context_sec"]),
            float(config["audio_chunk_sec"]),
            None,
        ),
    }
    collator = AudioCollator(
        processor, int(config["mert_chunk_batch_size"])
    )
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=1,
            shuffle=name == "train",
            collate_fn=collator,
            num_workers=0,
        )
        for name, dataset in datasets.items()
    }

    model = MERTLoRAValence(config).to(device)
    lora_parameters, head_parameters = split_parameters(model)
    parameter_groups = [
        {
            "params": lora_parameters,
            "lr": float(config["lora_learning_rate"]),
        }
    ]
    if head_parameters:
        parameter_groups.append(
            {
                "params": head_parameters,
                "lr": float(config["head_learning_rate"]),
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=float(config["weight_decay"])
    )
    accumulation = int(config["gradient_accumulation"])
    output_name = f"mert_lora_valence_fold{args.fold}{args.output_suffix}"
    output_dir = ROOT / "outputs" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best.pt"
    epochs = args.epochs or int(config["epochs"])
    parameter_summary = {
        "all_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "lora_parameters": sum(p.numel() for p in lora_parameters),
        "head_parameters": sum(p.numel() for p in head_parameters),
    }
    print(
        json.dumps(
            {
                "fold": args.fold,
                "device": str(device),
                "train_songs": len(train_rows),
                "held_songs": len(held_rows),
                "lora_modules": model.lora_modules,
                "head_init_checkpoint": model.head_init_checkpoint,
                **parameter_summary,
                "test_used": False,
                "legacy_validation_used": False,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.eval_only:
        held_metrics, _, _, _ = evaluate(
            model,
            loaders["held"],
            device,
            args.max_eval_batches,
        )
        print(
            json.dumps(
                {
                    "eval_only": True,
                    "held_songs_evaluated": (
                        min(len(held_rows), args.max_eval_batches)
                        if args.max_eval_batches is not None
                        else len(held_rows)
                    ),
                    **held_metrics,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    history = []
    best_rmse, best_epoch, bad_epochs = float("inf"), 0, 0
    training_started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        epoch_started = time.time()
        train_steps = 0
        for batch_index, batch in enumerate(loaders["train"]):
            if (
                args.max_train_steps is not None
                and train_steps >= args.max_train_steps
            ):
                break
            moved = move_batch(batch, device)
            output = model(moved)
            mask = torch.ones(
                moved["target"].shape[:2],
                dtype=torch.bool,
                device=device,
            )
            loss = loss_function(
                output, moved["target"], mask, config["loss"]
            )
            (loss / accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            train_steps += 1
            if train_steps % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [*lora_parameters, *head_parameters],
                    float(config["grad_clip"]),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if train_steps == 1 or train_steps % 25 == 0:
                print(
                    f"epoch={epoch:03d} step={train_steps:04d} "
                    f"loss={losses[-1]:.6f} "
                    f"elapsed={time.time() - epoch_started:.1f}s",
                    flush=True,
                )
        if train_steps % accumulation:
            torch.nn.utils.clip_grad_norm_(
                [*lora_parameters, *head_parameters],
                float(config["grad_clip"]),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        held_metrics, _, _, _ = evaluate(
            model,
            loaders["held"],
            device,
            args.max_eval_batches,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_steps": train_steps,
            "epoch_sec": time.time() - epoch_started,
            **held_metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"rmse={held_metrics['micro_rmse']:.6f} "
            f"pcc={held_metrics['micro_pcc']:.6f} "
            f"ccc={held_metrics['micro_ccc']:.6f} "
            f"within_pcc={held_metrics['within_pcc']:.6f} "
            f"epoch_sec={row['epoch_sec']:.1f}",
            flush=True,
        )
        if held_metrics["micro_rmse"] < best_rmse:
            best_rmse = held_metrics["micro_rmse"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["patience"]):
                break

    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    held_metrics, prediction, target, song_ids = evaluate(
        model,
        loaders["held"],
        device,
        args.max_eval_batches,
    )
    summary = {
        "fold": args.fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "train_songs": len(train_rows),
        "held_songs": len(held_rows),
        "limited_run": any(
            value is not None
            for value in (
                args.limit_train_songs,
                args.limit_held_songs,
                args.max_train_steps,
                args.max_eval_batches,
            )
        ),
        "elapsed_sec": time.time() - training_started,
        "lora_modules": model.lora_modules,
        "head_init_checkpoint": model.head_init_checkpoint,
        **parameter_summary,
        "checkpoint": str(checkpoint),
        "test_used": False,
        "legacy_validation_used": False,
        "config": config,
        **held_metrics,
    }
    np.save(output_dir / "held_pred.npy", prediction.astype(np.float32))
    np.save(output_dir / "held_target.npy", target.astype(np.float32))
    np.save(output_dir / "held_song_ids.npy", song_ids.astype(np.int32))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
