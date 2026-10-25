from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_bilstm_full_song_transfer import metrics


def smooth_by_song(
    prediction: np.ndarray,
    song_ids: np.ndarray,
    frames_by_dim: tuple[int, int],
) -> np.ndarray:
    smoothed = prediction.copy()
    for dim, frames in enumerate(frames_by_dim):
        if frames <= 1:
            continue
        if frames % 2 == 0:
            frames -= 1
        kernel = np.ones(frames, dtype=np.float64) / frames
        half = frames // 2
        for song_id in np.unique(song_ids):
            selected = song_ids == song_id
            series = prediction[selected, dim]
            padded = np.pad(series, (half, half), mode="edge")
            smoothed[selected, dim] = np.convolve(
                padded, kernel, mode="valid"
            )
    return smoothed


def linear_calibrate(prediction: np.ndarray, target: np.ndarray):
    calibrated = prediction.copy()
    parameters = []
    for dim in range(prediction.shape[1]):
        x = prediction[:, dim].astype(np.float64)
        y = target[:, dim].astype(np.float64)
        variance = np.var(x)
        if variance <= 1e-12:
            scale = 1.0
            bias = float(y.mean() - x.mean())
        else:
            covariance = np.mean((x - x.mean()) * (y - y.mean()))
            scale = float(covariance / variance)
            bias = float(y.mean() - scale * x.mean())
        calibrated[:, dim] = scale * prediction[:, dim] + bias
        parameters.append({"scale": scale, "bias": bias})
    return calibrated, parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-frames",
        default="1,11,21,41,61,81,101,121,151,181",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    prediction = np.load(output_dir / "hybrid_validation_pred.npy")
    target = np.load(output_dir / "hybrid_validation_target.npy")
    song_ids = np.load(output_dir / "validation_song_ids.npy")
    candidates = [
        int(value)
        for value in args.candidate_frames.split(",")
        if value.strip()
    ]
    rows = []
    best = None
    best_prediction = None
    for valence_frames in candidates:
        for arousal_frames in candidates:
            smoothed = smooth_by_song(
                prediction,
                song_ids,
                (valence_frames, arousal_frames),
            )
            calibrated, calibration = linear_calibrate(smoothed, target)
            result = metrics(calibrated, target)
            row = {
                "valence_smoothing_frames": valence_frames,
                "arousal_smoothing_frames": arousal_frames,
                "frame_hop_sec": 0.5,
                "calibration": calibration,
                **result,
            }
            rows.append(row)
            if best is None or row["rmse"] < best["rmse"]:
                best = row
                best_prediction = calibrated
    summary = {
        "postprocess_id": "validation_song_bounded_smoothing_linear_calibration",
        "validation_selected": True,
        "test_used": False,
        "selection_metric": "mean frame RMSE",
        "candidate_frames": candidates,
        "best": best,
        "all_trials": rows,
    }
    (output_dir / "validation_postprocess_search.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(
        output_dir / "hybrid_validation_pred_postprocessed.npy",
        best_prediction.astype(np.float32),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
