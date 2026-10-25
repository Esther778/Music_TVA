from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_dual_scale_full_song_va import (
    INDEX,
    PROTOCOL,
    ROOT,
    FullSongDataset,
    build_norm,
    make_loader,
    metrics,
    set_seed,
    window_metrics,
)


class ValenceGlobalLocal(nn.Module):
    def __init__(self, dimensions: tuple[int, int, int], config: dict):
        super().__init__()
        projection = int(config["projection_dim"])
        dropout = float(config["dropout"])
        self.projectors = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(dimension),
                nn.Linear(dimension, projection),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for dimension in dimensions
        )
        self.source_gate = nn.Linear(projection * 3, 3)
        self.attention = nn.Linear(projection, 1)
        self.global_head = nn.Sequential(
            nn.LayerNorm(projection * 3),
            nn.Linear(projection * 3, projection),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection, 1),
        )
        hidden = int(config["hidden_dim"])
        self.local_lstm = nn.LSTM(
            projection,
            hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.local_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )
        self.residual_logit = nn.Parameter(torch.tensor(-0.5))

    def forward(self, inputs, mask):
        projected = [
            projector(values)
            for projector, values in zip(self.projectors, inputs)
        ]
        joined = torch.cat(projected, dim=-1)
        source_weights = torch.softmax(self.source_gate(joined), dim=-1)
        fused = (
            source_weights.unsqueeze(-1)
            * torch.stack(projected, dim=-2)
        ).sum(dim=-2)

        attention_scores = self.attention(fused).squeeze(-1)
        attention_scores = attention_scores.masked_fill(
            ~mask, torch.finfo(attention_scores.dtype).min
        )
        attention = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
        attended = (attention * fused).sum(dim=1)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean = (fused * mask.unsqueeze(-1)).sum(dim=1) / denominator
        centered = (fused - mean.unsqueeze(1)) * mask.unsqueeze(-1)
        variance = (centered**2).sum(dim=1) / denominator
        std = torch.sqrt(variance.clamp_min(1e-8))
        global_prediction = self.global_head(
            torch.cat([attended, mean, std], dim=-1)
        )

        local_hidden, _ = self.local_lstm(centered)
        local_raw = self.local_head(local_hidden) * mask.unsqueeze(-1)
        local_mean = local_raw.sum(dim=1, keepdim=True) / (
            mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1)
        )
        local = (local_raw - local_mean) * mask.unsqueeze(-1)
        residual_scale = torch.sigmoid(self.residual_logit)
        prediction = (
            global_prediction.unsqueeze(1) + residual_scale * local
        )
        return {
            "prediction": prediction,
            "global": global_prediction,
            "local": local,
            "residual_scale": residual_scale,
            "source_weights": source_weights,
        }


def correlation_loss(prediction, target, ccc=False):
    pred_centered = prediction - prediction.mean()
    target_centered = target - target.mean()
    covariance = (pred_centered * target_centered).mean()
    if ccc:
        score = 2.0 * covariance / (
            pred_centered.var(unbiased=False)
            + target_centered.var(unbiased=False)
            + (prediction.mean() - target.mean()) ** 2
            + 1e-8
        )
    else:
        score = covariance / torch.sqrt(
            (pred_centered**2).mean()
            * (target_centered**2).mean()
            + 1e-8
        )
    return 1.0 - score


def moving_average_target(values: torch.Tensor, frames: int) -> torch.Tensor:
    if frames <= 1 or len(values) <= 1:
        return values
    frames = min(frames, len(values))
    if frames % 2 == 0:
        frames -= 1
    if frames <= 1:
        return values
    padding = frames // 2
    series = values.T.unsqueeze(0)
    series = F.pad(series, (padding, padding), mode="replicate")
    smoothed = F.avg_pool1d(series, kernel_size=frames, stride=1)
    return smoothed.squeeze(0).T


def loss_function(output, target, mask, config):
    losses = []
    weights = config["loss"]
    target_smoothing_frames = int(weights.get("target_smoothing_frames", 1))
    for index in range(len(target)):
        truth = target[index][mask[index], :1]
        training_truth = moving_average_target(
            truth, target_smoothing_frames
        )
        prediction = output["prediction"][index][mask[index]]
        true_mean = training_truth.mean(dim=0)
        true_local = training_truth - true_mean
        predicted_local = (
            output["residual_scale"] * output["local"][index][mask[index]]
        )
        loss = (
            float(weights["frame_huber"])
            * F.huber_loss(prediction, training_truth, delta=0.15)
            + float(weights["frame_mse"])
            * F.mse_loss(prediction, training_truth)
            + float(weights["frame_ccc"])
            * correlation_loss(prediction, training_truth, ccc=True)
            + float(weights["global_mse"])
            * F.mse_loss(output["global"][index], true_mean)
            + float(weights["local_mse"])
            * F.mse_loss(predicted_local, true_local)
            + float(weights["local_pearson"])
            * correlation_loss(predicted_local, true_local)
        )
        if len(training_truth) > 1:
            loss = loss + float(weights["derivative"]) * F.mse_loss(
                prediction[1:] - prediction[:-1],
                training_truth[1:] - training_truth[:-1],
            )
        loss = loss + float(weights["variance"]) * (
            prediction.std(unbiased=False)
            - training_truth.std(unbiased=False)
        ) ** 2
        losses.append(loss)
    return torch.stack(losses).mean()


def expanded_song_ids(rows):
    values = []
    for row in rows:
        with np.load(row["sequence_path"]) as data:
            values.append(
                np.full(
                    len(data["Y"]),
                    int(row["song_id"]),
                    dtype=np.int32,
                )
            )
    return np.concatenate(values)


def valence_metrics(prediction, target):
    combined_pred = np.column_stack([prediction, prediction])
    combined_target = np.column_stack([target, target])
    both = metrics(combined_pred, combined_target)
    return {
        "rmse": both["rmse_valence"],
        "mae": both["mae_valence"],
        "pcc": both["pcc_valence"],
        "ccc": both["ccc_valence"],
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    expanded_ids = []
    source_weights = []
    for layer6, final, hand, target, mask, _, batch_song_ids, _ in loader:
        output = model(
            (layer6.to(device), final.to(device), hand.to(device)),
            mask.to(device),
        )
        device_mask = mask.to(device)
        predictions.append(
            output["prediction"][device_mask].cpu().numpy().reshape(-1)
        )
        targets.append(target[mask][:, 0].numpy().reshape(-1))
        for batch_index, song_id in enumerate(batch_song_ids):
            expanded_ids.extend([int(song_id)] * int(mask[batch_index].sum()))
        source_weights.append(
            output["source_weights"][device_mask]
            .mean(dim=0)
            .cpu()
            .numpy()
        )
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    song_ids = np.asarray(expanded_ids, dtype=np.int32)
    frame = valence_metrics(prediction, target)
    window_results = {}
    for name, frames in (("window5", 10), ("window10", 20)):
        p_windows, t_windows = [], []
        for song_id in np.unique(song_ids):
            selected = song_ids == song_id
            p_song = prediction[selected]
            t_song = target[selected]
            for start in range(0, len(p_song), frames):
                stop = min(start + frames, len(p_song))
                p_windows.append(p_song[start:stop].mean())
                t_windows.append(t_song[start:stop].mean())
        window_results[name] = valence_metrics(
            np.asarray(p_windows), np.asarray(t_windows)
        )
    mean_pred, mean_target = [], []
    centered_pred, centered_target = [], []
    for song_id in np.unique(song_ids):
        selected = song_ids == song_id
        p_song = prediction[selected]
        t_song = target[selected]
        mean_pred.append(p_song.mean())
        mean_target.append(t_song.mean())
        centered_pred.append(p_song - p_song.mean())
        centered_target.append(t_song - t_song.mean())
    song_mean = valence_metrics(
        np.asarray(mean_pred), np.asarray(mean_target)
    )
    within = valence_metrics(
        np.concatenate(centered_pred), np.concatenate(centered_target)
    )
    result = {
        **{f"frame_{key}": value for key, value in frame.items()},
        **{
            f"{level}_{key}": value
            for level, values in window_results.items()
            for key, value in values.items()
        },
        **{f"song_mean_{key}": value for key, value in song_mean.items()},
        **{f"within_{key}": value for key, value in within.items()},
        "source_weight_layer6": float(
            np.asarray(source_weights)[:, 0].mean()
        ),
        "source_weight_final": float(
            np.asarray(source_weights)[:, 1].mean()
        ),
        "source_weight_handcrafted": float(
            np.asarray(source_weights)[:, 2].mean()
        ),
    }
    return result, prediction, target, song_ids


def train_stage(
    stage,
    model,
    train_loader,
    train_sampler,
    validation_loader,
    device,
    config,
    output_dir,
    epoch_override,
):
    stage_config = config[stage]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage_config["learning_rate"]),
        weight_decay=float(stage_config["weight_decay"]),
    )
    epochs = epoch_override or int(stage_config["epochs"])
    checkpoint = output_dir / f"{stage}_best.pt"
    history = []
    best_rmse = float("inf")
    best_epoch = 0
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        model.train()
        train_sampler.set_epoch(epoch)
        losses = []
        for layer6, final, hand, target, mask, _, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                (layer6.to(device), final.to(device), hand.to(device)),
                mask.to(device),
            )
            loss = loss_function(
                output, target.to(device), mask.to(device), config
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{stage}: non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["grad_clip"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation, _, _, _ = evaluate(
            model, validation_loader, device
        )
        row = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "epoch_sec": time.time() - epoch_started,
            **validation,
        }
        history.append(row)
        print(
            f"stage={stage} epoch={epoch:03d} "
            f"loss={row['train_loss']:.6f} "
            f"v_rmse={validation['frame_rmse']:.6f} "
            f"v_mae={validation['frame_mae']:.6f} "
            f"v_pcc={validation['frame_pcc']:.6f} "
            f"v_ccc={validation['frame_ccc']:.6f} "
            f"mean_rmse={validation['song_mean_rmse']:.6f} "
            f"within_rmse={validation['within_rmse']:.6f} "
            f"sec={row['epoch_sec']:.1f}",
            flush=True,
        )
        score = validation["frame_rmse"]
        if not np.isfinite(score):
            raise RuntimeError(f"{stage}: non-finite validation RMSE")
        if score < best_rmse:
            best_rmse = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= int(stage_config["patience"]):
                break
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    return history, best_epoch, best_rmse, checkpoint


def combined_metrics(
    valence_prediction,
    valence_target,
    song_ids,
    transfer_dir: Path,
):
    arousal_prediction = np.load(transfer_dir / "validation_pred.npy")[:, 1]
    transfer_target = np.load(transfer_dir / "validation_target.npy")
    transfer_ids = np.load(transfer_dir / "validation_song_ids.npy")
    selected = np.isin(transfer_ids, np.unique(song_ids))
    arousal_prediction = arousal_prediction[selected]
    transfer_target = transfer_target[selected]
    transfer_ids = transfer_ids[selected]
    aligned_arousal = []
    aligned_target = []
    transfer_positions = {}
    counters = {}
    for index, song_id in enumerate(transfer_ids):
        song_id = int(song_id)
        frame_index = counters.get(song_id, 0)
        transfer_positions[(song_id, frame_index)] = index
        counters[song_id] = frame_index + 1
    counters = {}
    for song_id in song_ids:
        song_id = int(song_id)
        frame_index = counters.get(song_id, 0)
        transfer_index = transfer_positions[(song_id, frame_index)]
        aligned_arousal.append(arousal_prediction[transfer_index])
        aligned_target.append(transfer_target[transfer_index])
        counters[song_id] = frame_index + 1
    arousal_prediction = np.asarray(aligned_arousal)
    transfer_target = np.asarray(aligned_target)
    if not np.allclose(valence_target, transfer_target[:, 0], atol=1e-5):
        raise RuntimeError("Validation target mismatch.")
    prediction = np.column_stack([valence_prediction, arousal_prediction])
    target = transfer_target
    frame = metrics(prediction, target)
    window5 = window_metrics(prediction, target, song_ids, 10)
    window10 = window_metrics(prediction, target, song_ids, 20)
    return {
        **{f"frame_{key}": value for key, value in frame.items()},
        **{f"window5_{key}": value for key, value in window5.items()},
        **{f"window10_{key}": value for key, value in window10.items()},
    }, prediction, target


def optional_combined_metrics(
    outer_fold: int,
    prediction: np.ndarray,
    target: np.ndarray,
    song_ids: np.ndarray,
):
    transfer_dir = ROOT / f"outputs/bilstm_full_song_transfer_fold{outer_fold}"
    required = (
        transfer_dir / "validation_pred.npy",
        transfer_dir / "validation_target.npy",
        transfer_dir / "validation_song_ids.npy",
    )
    if not all(path.is_file() for path in required):
        return None, None, None, str(transfer_dir)
    hybrid, hybrid_prediction, hybrid_target = combined_metrics(
        prediction, target, song_ids, transfer_dir
    )
    return hybrid, hybrid_prediction, hybrid_target, str(transfer_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/valence_global_local_full_song.json",
    )
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--finetune-epochs", type=int)
    parser.add_argument("--limit-short", type=int)
    parser.add_argument("--limit-full", type=int)
    parser.add_argument("--limit-validation", type=int)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument(
        "--normalization-source",
        choices=["all_train", "full_only"],
        default="all_train",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"]) + args.outer_fold
    set_seed(seed)
    device = torch.device(args.device)
    protocol = pd.read_csv(PROTOCOL)
    fold = protocol[protocol["outer_fold"] == args.outer_fold].copy()
    index = pd.read_csv(INDEX)[["song_id", "frames", "sequence_path"]]
    fold = fold.drop(columns=["sequence_path"]).merge(index, on="song_id")
    short_rows = fold[
        (fold["role"] == "train") & (fold["domain"] == "short_clip")
    ].copy()
    full_rows = fold[
        (fold["role"] == "train") & (fold["domain"] == "full_song")
    ].copy()
    validation_rows = fold[fold["role"] == "validation"].copy()
    test_rows = fold[fold["role"] == "test"].copy()
    if args.limit_short:
        short_rows = short_rows.head(args.limit_short)
    if args.limit_full:
        full_rows = full_rows.head(args.limit_full)
    if args.limit_validation:
        validation_rows = validation_rows.head(args.limit_validation)
    overlap = (
        set(short_rows.song_id) | set(full_rows.song_id)
    ) & (set(validation_rows.song_id) | set(test_rows.song_id))
    if overlap:
        raise RuntimeError(f"Train/held-out overlap: {sorted(overlap)}")

    output_dir = ROOT / "outputs" / (
        f"{config['trial_id']}_fold{args.outer_fold}{args.output_suffix}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    training_rows = (
        full_rows
        if args.normalization_source == "full_only"
        else pd.concat([short_rows, full_rows], ignore_index=True)
    )
    norms = build_norm(training_rows, output_dir / "feature_norm.npz")
    datasets = {
        "short": FullSongDataset(short_rows, norms),
        "full": FullSongDataset(full_rows, norms),
        "validation": FullSongDataset(validation_rows, norms),
    }
    loaders, samplers = {}, {}
    for stage, dataset in datasets.items():
        if stage == "short":
            batch_size = int(config["pretrain"]["batch_size"])
        elif stage == "full":
            batch_size = int(config["finetune"]["batch_size"])
        else:
            batch_size = 1
        loaders[stage], samplers[stage] = make_loader(
            dataset, batch_size, stage != "validation", seed
        )
    sample = datasets["short"][0]
    model = ValenceGlobalLocal(
        (sample[0].shape[1], sample[1].shape[1], sample[2].shape[1]),
        config,
    ).to(device)
    if args.initial_checkpoint:
        model.load_state_dict(
            torch.load(
                args.initial_checkpoint,
                map_location=device,
                weights_only=True,
            )
        )
    started = time.time()
    initial, _, _, _ = evaluate(model, loaders["validation"], device)
    print(
        json.dumps(
            {
                "device": str(device),
                "short_pretrain_songs": len(short_rows),
                "full_finetune_songs": len(full_rows),
                "validation_full_songs": len(validation_rows),
                "locked_test_full_songs": len(test_rows),
            "test_used": False,
            "parameters": sum(p.numel() for p in model.parameters()),
            "initial_checkpoint": (
                str(args.initial_checkpoint)
                if args.initial_checkpoint
                else None
            ),
            "initial_validation": initial,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.evaluate_checkpoint:
        model.load_state_dict(
            torch.load(
                args.evaluate_checkpoint,
                map_location=device,
                weights_only=True,
            )
        )
        final, prediction, target, song_ids = evaluate(
            model, loaders["validation"], device
        )
        hybrid, hybrid_prediction, hybrid_target, transfer_dir = optional_combined_metrics(
            args.outer_fold,
            prediction, target, song_ids
        )
        summary = {
            "trial_id": config["trial_id"],
            "outer_fold": args.outer_fold,
            "seed": seed,
            "elapsed_sec": time.time() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
            "short_pretrain_songs": len(short_rows),
            "full_finetune_songs": len(full_rows),
            "validation_full_songs": len(validation_rows),
            "locked_test_full_songs": len(test_rows),
            "test_used": False,
            "evaluation_only": True,
            "evaluated_checkpoint": str(args.evaluate_checkpoint),
            "initial_checkpoint": (
                str(args.initial_checkpoint)
                if args.initial_checkpoint
                else None
            ),
            "config": config,
            "finetuned_valence_validation": final,
            "hybrid_validation": hybrid,
            "hybrid_transfer_dir": transfer_dir,
        }
        np.save(output_dir / "valence_validation_pred.npy", prediction)
        np.save(output_dir / "validation_target.npy", target)
        np.save(output_dir / "validation_song_ids.npy", song_ids)
        if hybrid_prediction is not None:
            np.save(output_dir / "hybrid_validation_pred.npy", hybrid_prediction)
            np.save(output_dir / "hybrid_validation_target.npy", hybrid_target)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return
    if args.skip_pretrain:
        pretrain = ([], 0, None, None)
        pretrained = initial
    else:
        pretrain = train_stage(
            "pretrain",
            model,
            loaders["short"],
            samplers["short"],
            loaders["validation"],
            device,
            config,
            output_dir,
            args.pretrain_epochs,
        )
        pretrained, _, _, _ = evaluate(model, loaders["validation"], device)
    finetune = train_stage(
        "finetune",
        model,
        loaders["full"],
        samplers["full"],
        loaders["validation"],
        device,
        config,
        output_dir,
        args.finetune_epochs,
    )
    final, prediction, target, song_ids = evaluate(
        model, loaders["validation"], device
    )
    hybrid, hybrid_prediction, hybrid_target, transfer_dir = optional_combined_metrics(
        args.outer_fold,
        prediction, target, song_ids
    )
    summary = {
        "trial_id": config["trial_id"],
        "outer_fold": args.outer_fold,
        "seed": seed,
        "elapsed_sec": time.time() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "short_pretrain_songs": len(short_rows),
        "full_finetune_songs": len(full_rows),
        "validation_full_songs": len(validation_rows),
        "locked_test_full_songs": len(test_rows),
        "test_used": False,
        "initial_checkpoint": (
            str(args.initial_checkpoint) if args.initial_checkpoint else None
        ),
        "config": config,
        "skip_pretrain": args.skip_pretrain,
        "normalization_source": args.normalization_source,
        "pretrain_best_epoch": pretrain[1],
        "pretrain_best_rmse": pretrain[2],
        "pretrain_checkpoint": str(pretrain[3]) if pretrain[3] else None,
        "finetune_best_epoch": finetune[1],
        "finetune_best_rmse": finetune[2],
        "finetune_checkpoint": str(finetune[3]),
        "initial_validation": initial,
        "pretrained_validation": pretrained,
        "finetuned_valence_validation": final,
        "hybrid_validation": hybrid,
        "hybrid_transfer_dir": transfer_dir,
    }
    np.save(output_dir / "valence_validation_pred.npy", prediction)
    np.save(output_dir / "validation_target.npy", target)
    np.save(output_dir / "validation_song_ids.npy", song_ids)
    if hybrid_prediction is not None:
        np.save(output_dir / "hybrid_validation_pred.npy", hybrid_prediction)
        np.save(output_dir / "hybrid_validation_target.npy", hybrid_target)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    history = pretrain[0] + finetune[0]
    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
