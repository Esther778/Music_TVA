from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
INDEX = (
    PROJECT_ROOT
    / "research/phase1b_bert_deam_va_training/data/datasets/mert_va/sequence_index.csv"
)


def main() -> None:
    index = pd.read_csv(INDEX)
    rows = index[index["split"] == "train"].copy().sort_values("song_id")
    summaries = []
    for row in rows.itertuples():
        with np.load(row.sequence_path) as data:
            target = data["Y"].astype(np.float32)
        summaries.append(
            {
                "song_id": int(row.song_id),
                "frames": int(len(target)),
                "mean_valence": float(target[:, 0].mean()),
                "mean_arousal": float(target[:, 1].mean()),
            }
        )
    manifest = pd.DataFrame(summaries)
    manifest["valence_bin"] = pd.qcut(
        manifest["mean_valence"], q=5, labels=False
    ).astype(int)
    manifest["arousal_bin"] = pd.qcut(
        manifest["mean_arousal"], q=4, labels=False
    ).astype(int)
    manifest["stratum"] = manifest["valence_bin"].astype(str)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    manifest["fold"] = -1
    for fold, (_, held_out) in enumerate(
        splitter.split(manifest, manifest["stratum"])
    ):
        manifest.loc[held_out, "fold"] = fold
    output = ROOT / "data/five_fold_manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)

    audit = {"seed": 2026, "folds": {}}
    for fold in range(5):
        part = manifest[manifest["fold"] == fold]
        audit["folds"][str(fold)] = {
            "songs": int(len(part)),
            "frames": int(part["frames"].sum()),
            "long_songs": int((part["frames"] > 100).sum()),
            "mean_valence": float(part["mean_valence"].mean()),
            "std_valence": float(part["mean_valence"].std()),
            "mean_arousal": float(part["mean_arousal"].mean()),
            "std_arousal": float(part["mean_arousal"].std()),
        }
    audit["all_songs"] = int(len(manifest))
    audit["unique_songs"] = int(manifest["song_id"].nunique())
    audit["unassigned"] = int((manifest["fold"] < 0).sum())
    (ROOT / "outputs/five_fold_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
