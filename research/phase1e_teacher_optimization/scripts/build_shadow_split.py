from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
INDEX = (
    PROJECT_ROOT
    / "research/phase1b_bert_deam_va_training/data/datasets/mert_va/sequence_index.csv"
)


def quantile_bins(values: pd.Series, bins: int) -> pd.Series:
    return pd.qcut(values, q=bins, labels=False, duplicates="drop").astype(int)


def main() -> None:
    index = pd.read_csv(INDEX)
    train = index[index["split"] == "train"].copy().sort_values("song_id")
    summaries = []
    for row in train.itertuples():
        with np.load(row.sequence_path) as data:
            target = data["Y"].astype(np.float32)
        summaries.append(
            {
                "song_id": int(row.song_id),
                "mean_valence": float(target[:, 0].mean()),
                "mean_arousal": float(target[:, 1].mean()),
                "frames": int(len(target)),
            }
        )
    summary = pd.DataFrame(summaries)
    summary["valence_bin"] = quantile_bins(summary["mean_valence"], 5)
    summary["arousal_bin"] = quantile_bins(summary["mean_arousal"], 4)
    summary["stratum"] = (
        summary["valence_bin"].astype(str)
        + "_"
        + summary["arousal_bin"].astype(str)
    )
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=180, random_state=2026
    )
    train_indices, shadow_indices = next(
        splitter.split(summary, summary["stratum"])
    )
    summary["shadow_split"] = "train_sub"
    summary.loc[shadow_indices, "shadow_split"] = "shadow_val"
    output = ROOT / "data/shadow_split.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values("song_id").to_csv(output, index=False)

    audit = {}
    for split in ("train_sub", "shadow_val"):
        part = summary[summary["shadow_split"] == split]
        audit[split] = {
            "songs": int(len(part)),
            "frames": int(part["frames"].sum()),
            "long_songs": int((part["frames"] > 100).sum()),
            "mean_valence": float(part["mean_valence"].mean()),
            "std_valence": float(part["mean_valence"].std()),
            "mean_arousal": float(part["mean_arousal"].mean()),
            "std_arousal": float(part["mean_arousal"].std()),
            "valence_min": float(part["mean_valence"].min()),
            "valence_max": float(part["mean_valence"].max()),
        }
    audit["overlap"] = int(
        len(
            set(summary[summary["shadow_split"] == "train_sub"]["song_id"])
            & set(summary[summary["shadow_split"] == "shadow_val"]["song_id"])
        )
    )
    audit["seed"] = 2026
    (ROOT / "outputs/shadow_split_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
