from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d

from train_bilstm_full_song_transfer import (
    INDEX,
    PROTOCOL,
    ROOT,
    SongDataset,
    TransferBiLSTM,
    build_normalization,
    evaluate as evaluate_bilstm,
    make_loader as make_bilstm_loader,
    metrics,
)
from train_valence_global_local_full_song import (
    FullSongDataset,
    ValenceGlobalLocal,
    build_norm,
    evaluate as evaluate_valence,
    make_loader as make_valence_loader,
)


FROZEN_POSTPROCESS = {
    "postprocess_id": "conservative_v2_no_isotonic",
    "selected_on": "fold0_validation",
    "test_used_for_selection": False,
    "valence_smoothing": {
        "method": "moving_average",
        "frames": 321,
        "seconds": 160.5,
    },
    "arousal_smoothing": {
        "method": "gaussian",
        "sigma_frames": 14,
        "sigma_seconds": 7.0,
    },
    "valence_calibration": {
        "method": "quadratic",
        "a2": -0.7599359910982995,
        "a1": 1.1917104763060982,
        "bias": 0.027344647777076514,
    },
    "arousal_calibration": {
        "method": "quadratic",
        "a2": 0.28838856178322847,
        "a1": 1.1163182091498336,
        "bias": -0.018169279634738793,
    },
}


def moving_average(series: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 1:
        return series.copy()
    if frames % 2 == 0:
        frames += 1
    half = frames // 2
    padded = np.pad(series, (half, half), mode="edge")
    kernel = np.ones(frames, dtype=np.float64) / frames
    return np.convolve(padded, kernel, mode="valid")


def apply_frozen_postprocess(
    prediction: np.ndarray,
    song_ids: np.ndarray,
) -> np.ndarray:
    result = prediction.copy().astype(np.float64)
    for song_id in np.unique(song_ids):
        selected = song_ids == song_id
        result[selected, 0] = moving_average(
            prediction[selected, 0],
            FROZEN_POSTPROCESS["valence_smoothing"]["frames"],
        )
        result[selected, 1] = gaussian_filter1d(
            prediction[selected, 1],
            sigma=FROZEN_POSTPROCESS["arousal_smoothing"]["sigma_frames"],
            mode="nearest",
        )
    valence = result[:, 0]
    v_calibration = FROZEN_POSTPROCESS["valence_calibration"]
    result[:, 0] = (
        v_calibration["a2"] * valence**2
        + v_calibration["a1"] * valence
        + v_calibration["bias"]
    )
    arousal = result[:, 1]
    a_calibration = FROZEN_POSTPROCESS["arousal_calibration"]
    result[:, 1] = (
        a_calibration["a2"] * arousal**2
        + a_calibration["a1"] * arousal
        + a_calibration["bias"]
    )
    return result.astype(np.float32)


def align_arousal(
    valence_prediction: np.ndarray,
    valence_target: np.ndarray,
    valence_song_ids: np.ndarray,
    arousal_prediction: np.ndarray,
    arousal_target: np.ndarray,
    arousal_song_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions: dict[tuple[int, int], int] = {}
    counters: dict[int, int] = {}
    for index, song_id in enumerate(arousal_song_ids):
        song_id = int(song_id)
        frame_index = counters.get(song_id, 0)
        positions[(song_id, frame_index)] = index
        counters[song_id] = frame_index + 1

    aligned_arousal = []
    aligned_target = []
    counters = {}
    for song_id in valence_song_ids:
        song_id = int(song_id)
        frame_index = counters.get(song_id, 0)
        index = positions[(song_id, frame_index)]
        aligned_arousal.append(arousal_prediction[index, 1])
        aligned_target.append(arousal_target[index])
        counters[song_id] = frame_index + 1

    aligned_arousal = np.asarray(aligned_arousal, dtype=np.float32)
    aligned_target = np.asarray(aligned_target, dtype=np.float32)
    if not np.allclose(valence_target, aligned_target[:, 0], atol=1e-5):
        raise RuntimeError("Valence and arousal test targets are misaligned.")
    prediction = np.column_stack(
        [valence_prediction, aligned_arousal]
    ).astype(np.float32)
    return prediction, aligned_target


def main() -> None:
    device = torch.device("cpu")
    outer_fold = 0
    seed = 2026

    protocol = pd.read_csv(PROTOCOL)
    fold = protocol[protocol["outer_fold"] == outer_fold].copy()
    index = pd.read_csv(INDEX)[["song_id", "frames", "sequence_path"]]
    fold = fold.drop(columns=["sequence_path"]).merge(index, on="song_id")
    train_rows = fold[fold["role"] == "train"].copy()
    test_rows = fold[fold["role"] == "test"].copy()

    valence_output = ROOT / "outputs/valence_global_local_full_song_fold0"
    valence_config = json.loads(
        (ROOT / "configs/valence_global_local_full_song.json").read_text(
            encoding="utf-8"
        )
    )
    valence_norms = build_norm(
        train_rows,
        valence_output / "feature_norm.npz",
    )
    valence_dataset = FullSongDataset(test_rows, valence_norms)
    valence_loader, _ = make_valence_loader(
        valence_dataset, batch_size=1, shuffle=False, seed=seed
    )
    sample = valence_dataset[0]
    valence_model = ValenceGlobalLocal(
        (sample[0].shape[1], sample[1].shape[1], sample[2].shape[1]),
        valence_config,
    ).to(device)
    valence_model.load_state_dict(
        torch.load(
            valence_output / "finetune_best.pt",
            map_location=device,
            weights_only=True,
        )
    )
    valence_result, valence_pred, valence_target, valence_song_ids = (
        evaluate_valence(valence_model, valence_loader, device)
    )

    arousal_output = ROOT / "outputs/bilstm_full_song_transfer_fold0"
    arousal_config = json.loads(
        (ROOT / "configs/bilstm_full_song_transfer.json").read_text(
            encoding="utf-8"
        )
    )
    mean, std = build_normalization(
        train_rows,
        arousal_output / "feature_norm.npz",
    )
    arousal_dataset = SongDataset(test_rows, mean, std)
    arousal_loader = make_bilstm_loader(arousal_dataset, 1, False)
    arousal_model = TransferBiLSTM(768, arousal_config).to(device)
    arousal_model.load_state_dict(
        torch.load(
            arousal_output / "finetune_best.pt",
            map_location=device,
            weights_only=True,
        )
    )
    arousal_result, arousal_pred, arousal_target, arousal_song_ids = (
        evaluate_bilstm(arousal_model, arousal_loader, device)
    )

    raw_prediction, target = align_arousal(
        valence_pred,
        valence_target,
        valence_song_ids,
        arousal_pred,
        arousal_target,
        arousal_song_ids,
    )
    postprocessed = apply_frozen_postprocess(
        raw_prediction,
        valence_song_ids,
    )
    raw_metrics = metrics(raw_prediction, target)
    postprocessed_metrics = metrics(postprocessed, target)

    output_dir = ROOT / "outputs/fold0_locked_test_frozen_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "test_raw_pred.npy", raw_prediction)
    np.save(output_dir / "test_postprocessed_pred.npy", postprocessed)
    np.save(output_dir / "test_target.npy", target.astype(np.float32))
    np.save(output_dir / "test_song_ids.npy", valence_song_ids.astype(np.int32))
    test_rows[["song_id", "domain", "frames", "sequence_path"]].to_csv(
        output_dir / "test_manifest.csv",
        index=False,
    )
    summary = {
        "run_id": "fold0_locked_test_frozen_v2",
        "outer_fold": outer_fold,
        "test_used": True,
        "test_role": "locked_outer_test",
        "selection_used_test": False,
        "test_complete_songs": int(len(test_rows)),
        "test_song_ids": [int(value) for value in sorted(test_rows.song_id)],
        "valence_checkpoint": str(valence_output / "finetune_best.pt"),
        "arousal_checkpoint": str(arousal_output / "finetune_best.pt"),
        "frozen_postprocess": FROZEN_POSTPROCESS,
        "valence_raw_test": valence_result,
        "arousal_raw_test": arousal_result,
        "hybrid_raw_test": raw_metrics,
        "hybrid_frozen_postprocessed_test": postprocessed_metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
