from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from train_layer6_gated_fusion import LAYER6_ROOT, time_indices
from train_valence_harmonic_context import scalar_metrics


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
INDEX = (
    PROJECT_ROOT
    / "research/phase1b_bert_deam_va_training/data/datasets/mert_va/sequence_index.csv"
)
FOLD_MANIFEST = ROOT / "data/five_fold_manifest.csv"


def macro_metrics(
    prediction: np.ndarray, target: np.ndarray, song_ids: np.ndarray
) -> dict[str, float]:
    rows = []
    for song_id in np.unique(song_ids):
        mask = song_ids == song_id
        rows.append(scalar_metrics(prediction[mask], target[mask]))
    return {
        key: float(np.nanmean([row[key] for row in rows]))
        for key in ("rmse", "mae", "pcc", "ccc")
    }


def window_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    song_ids: np.ndarray,
    frames: int,
) -> dict[str, float]:
    pred_windows, target_windows = [], []
    for song_id in np.unique(song_ids):
        mask = song_ids == song_id
        pred_song = prediction[mask]
        target_song = target[mask]
        for start in range(0, len(pred_song), frames):
            stop = min(start + frames, len(pred_song))
            pred_windows.append(float(pred_song[start:stop].mean()))
            target_windows.append(float(target_song[start:stop].mean()))
    return scalar_metrics(np.asarray(pred_windows), np.asarray(target_windows))


def load_all():
    index = pd.read_csv(INDEX).set_index("song_id")
    manifest = pd.read_csv(FOLD_MANIFEST).sort_values("song_id")
    layer6_values, final_values, targets, songs, folds = [], [], [], [], []
    frame_counts = {}
    for row in manifest.itertuples():
        song_id = int(row.song_id)
        sequence_path = Path(index.loc[song_id, "sequence_path"])
        with np.load(sequence_path) as sequence:
            times = sequence["time_sec"].astype(np.float32)
            final = sequence["X"].astype(np.float32)
            target = sequence["Y"][:, 0].astype(np.float32)
        with np.load(LAYER6_ROOT / f"{song_id}.npz") as layer_file:
            layer_times = layer_file["time_sec"].astype(np.float32)
            layer6 = layer_file["embeddings"][:, 0].astype(np.float32)
        layer6 = layer6[time_indices(layer_times, times, song_id)]
        if len(layer6) != len(final) or len(final) != len(target):
            raise ValueError(f"Song {song_id}: alignment length mismatch.")
        layer6_values.append(layer6)
        final_values.append(final)
        targets.append(target)
        songs.append(np.full(len(target), song_id, dtype=np.int32))
        folds.append(np.full(len(target), int(row.fold), dtype=np.int8))
        frame_counts[song_id] = len(target)
    return (
        np.concatenate(layer6_values),
        np.concatenate(final_values),
        np.concatenate(targets),
        np.concatenate(songs),
        np.concatenate(folds),
        frame_counts,
    )


def main() -> None:
    layer6, final, target, song_ids, fold_ids, frame_counts = load_all()
    candidates = {
        "final": 0.0,
        "mix_25_layer6": 0.25,
        "mean_layer6_final": 0.5,
        "mix_75_layer6": 0.75,
        "layer6": 1.0,
    }
    rows = []
    oof_predictions = {
        candidate: np.empty_like(target) for candidate in candidates
    }
    for fold in range(5):
        train_mask = fold_ids != fold
        held_mask = fold_ids == fold
        mean6 = layer6[train_mask].mean(axis=0, keepdims=True)
        std6 = layer6[train_mask].std(axis=0, keepdims=True) + 1e-6
        mean_final = final[train_mask].mean(axis=0, keepdims=True)
        std_final = final[train_mask].std(axis=0, keepdims=True) + 1e-6
        train6 = (layer6[train_mask] - mean6) / std6
        held6 = (layer6[held_mask] - mean6) / std6
        train_final = (final[train_mask] - mean_final) / std_final
        held_final = (final[held_mask] - mean_final) / std_final
        train_songs = song_ids[train_mask]
        sample_weight = np.asarray(
            [1.0 / frame_counts[int(song)] for song in train_songs],
            dtype=np.float64,
        )
        sample_weight *= len(sample_weight) / sample_weight.sum()
        for name, layer6_weight in candidates.items():
            train_x = (
                layer6_weight * train6 + (1.0 - layer6_weight) * train_final
            )
            held_x = (
                layer6_weight * held6 + (1.0 - layer6_weight) * held_final
            )
            model = Ridge(alpha=100.0, solver="lsqr", max_iter=2000, tol=1e-5)
            model.fit(train_x, target[train_mask], sample_weight=sample_weight)
            prediction = model.predict(held_x).astype(np.float32)
            oof_predictions[name][held_mask] = prediction
            micro = scalar_metrics(prediction, target[held_mask])
            macro = macro_metrics(prediction, target[held_mask], song_ids[held_mask])
            window5 = window_metrics(
                prediction, target[held_mask], song_ids[held_mask], 10
            )
            window10 = window_metrics(
                prediction, target[held_mask], song_ids[held_mask], 20
            )
            rows.append(
                {
                    "fold": fold,
                    "candidate": name,
                    "layer6_weight": layer6_weight,
                    "held_songs": int(len(np.unique(song_ids[held_mask]))),
                    **{f"micro_{key}": value for key, value in micro.items()},
                    **{f"macro_{key}": value for key, value in macro.items()},
                    **{f"window5_{key}": value for key, value in window5.items()},
                    **{f"window10_{key}": value for key, value in window10.items()},
                }
            )
            print(
                f"fold={fold} candidate={name} "
                f"micro_rmse={micro['rmse']:.6f} "
                f"pcc={micro['pcc']:.6f} ccc={micro['ccc']:.6f}"
            )
        del train6, held6, train_final, held_final
    result_table = pd.DataFrame(rows)
    output_csv = ROOT / "outputs/mert_layer_mix_five_fold.csv"
    result_table.to_csv(output_csv, index=False)
    summary_rows = []
    for name in candidates:
        part = result_table[result_table["candidate"] == name]
        row = {"candidate": name, "layer6_weight": candidates[name]}
        for column in [
            "micro_rmse", "micro_mae", "micro_pcc", "micro_ccc",
            "macro_rmse", "macro_mae", "macro_pcc", "macro_ccc",
            "window5_rmse", "window5_pcc", "window5_ccc",
            "window10_rmse", "window10_pcc", "window10_ccc",
        ]:
            row[f"{column}_mean"] = float(part[column].mean())
            row[f"{column}_std"] = float(part[column].std(ddof=1))
        final_wins = int(
            (
                part.sort_values("fold")["micro_rmse"].to_numpy()
                < result_table[result_table["candidate"] == "final"]
                .sort_values("fold")["micro_rmse"]
                .to_numpy()
            ).sum()
        )
        row["folds_better_than_final"] = final_wins
        summary_rows.append(row)
        np.save(
            ROOT / f"outputs/oof_{name}_valence.npy",
            oof_predictions[name],
        )
    summary = {
        "dataset": "DEAM original train split only",
        "test_used": False,
        "legacy_validation_used": False,
        "folds": 5,
        "split_seed": 2026,
        "ridge_alpha": 100.0,
        "training_weighting": "equal total weight per song",
        "candidates": summary_rows,
    }
    (ROOT / "outputs/mert_layer_mix_five_fold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
