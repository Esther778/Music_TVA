from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_dual_scale_full_song_va import INDEX, PROTOCOL, ROOT
from train_valence_global_local_full_song import (
    FullSongDataset,
    ValenceGlobalLocal,
    build_norm,
    evaluate,
    make_loader,
    valence_metrics,
)


OUTPUT_DIR = ROOT / "outputs/valence_global_local_nested_cv"


def main() -> None:
    device = torch.device("cpu")
    config = json.loads(
        (ROOT / "configs/valence_global_local_full_song.json").read_text(
            encoding="utf-8"
        )
    )
    protocol = pd.read_csv(PROTOCOL)
    index = pd.read_csv(INDEX)[["song_id", "frames", "sequence_path"]]
    protocol = protocol.drop(columns=["sequence_path"]).merge(
        index, on="song_id"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    all_predictions = []
    all_targets = []
    all_song_ids = []
    for fold in sorted(protocol.outer_fold.unique()):
        output_dir = ROOT / f"outputs/valence_global_local_full_song_fold{fold}"
        checkpoint = output_dir / "finetune_best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        part = protocol[protocol.outer_fold == fold].copy()
        train_rows = part[part.role == "train"].copy()
        validation_rows = part[part.role == "validation"].copy()
        test_rows = part[part.role == "test"].copy()
        norms = build_norm(train_rows, output_dir / "feature_norm.npz")
        validation_dataset = FullSongDataset(validation_rows, norms)
        test_dataset = FullSongDataset(test_rows, norms)
        sample = validation_dataset[0]
        model = ValenceGlobalLocal(
            (sample[0].shape[1], sample[1].shape[1], sample[2].shape[1]),
            config,
        ).to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        validation_loader, _ = make_loader(
            validation_dataset, batch_size=1, shuffle=False, seed=2026 + int(fold)
        )
        test_loader, _ = make_loader(
            test_dataset, batch_size=1, shuffle=False, seed=2026 + int(fold)
        )
        validation_result, _, _, _ = evaluate(model, validation_loader, device)
        test_result, prediction, target, song_ids = evaluate(
            model, test_loader, device
        )
        fold_row = {
            "outer_fold": int(fold),
            "checkpoint": str(checkpoint),
            "train_full_songs": int(
                len(train_rows[train_rows.domain == "full_song"])
            ),
            "validation_full_songs": int(len(validation_rows)),
            "test_full_songs": int(len(test_rows)),
            "test_song_ids": [int(value) for value in sorted(test_rows.song_id)],
            **{
                f"validation_{key}": value
                for key, value in validation_result.items()
            },
            **{f"test_{key}": value for key, value in test_result.items()},
        }
        fold_rows.append(fold_row)
        all_predictions.append(prediction)
        all_targets.append(target)
        all_song_ids.append(song_ids)
        print(
            f"fold={fold} "
            f"validation_v_rmse={validation_result['frame_rmse']:.6f} "
            f"test_v_rmse={test_result['frame_rmse']:.6f} "
            f"test_v_pcc={test_result['frame_pcc']:.6f} "
            f"test_v_ccc={test_result['frame_ccc']:.6f}"
        )

    prediction = np.concatenate(all_predictions)
    target = np.concatenate(all_targets)
    song_ids = np.concatenate(all_song_ids)
    overall = valence_metrics(prediction, target)
    summary = {
        "experiment_id": "valence_global_local_nested_cv",
        "test_used_for_training_or_selection": False,
        "model": "Valence Global-Local full-song expert",
        "outer_test_songs": int(len(np.unique(song_ids))),
        "outer_test_frames": int(len(target)),
        "folds": fold_rows,
        "overall_outer_test": overall,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "fold_results.csv", index=False)
    np.save(OUTPUT_DIR / "outer_test_valence_pred.npy", prediction.astype(np.float32))
    np.save(OUTPUT_DIR / "outer_test_valence_target.npy", target.astype(np.float32))
    np.save(OUTPUT_DIR / "outer_test_song_ids.npy", song_ids.astype(np.int32))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
