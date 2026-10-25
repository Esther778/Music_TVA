from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
PHASE1B = PROJECT_ROOT / "research/phase1b_bert_deam_va_training"
SOURCE_MANIFEST = PHASE1B / "data/manifests/split_manifest.csv"
SEQUENCE_INDEX = PHASE1B / "data/datasets/mert_va/sequence_index.csv"
OUTPUT_DIR = ROOT / "data/full_song_protocol"
SEED = 2026
FOLDS = 5
SPLIT_SEARCH_TRIALS = 10000


def load_inventory() -> pd.DataFrame:
    manifest = pd.read_csv(SOURCE_MANIFEST)
    index = pd.read_csv(SEQUENCE_INDEX)[
        ["song_id", "frames", "sequence_path"]
    ]
    rows = manifest.merge(index, on="song_id", how="inner")
    if len(rows) != 1802:
        raise ValueError(f"Expected 1802 aligned songs, found {len(rows)}.")

    records = []
    for row in rows.sort_values("song_id").itertuples():
        song_id = int(row.song_id)
        is_full_song = 2001 <= song_id <= 2058
        with np.load(row.sequence_path) as data:
            target = data["Y"].astype(np.float32)
            times = data["time_sec"].astype(np.float32)
        audio_path = Path(row.audio_path)
        records.append(
            {
                "song_id": song_id,
                "domain": "full_song" if is_full_song else "short_clip",
                "audio_path": str(audio_path),
                "sequence_path": str(row.sequence_path),
                "audio_duration_sec": float(
                    librosa.get_duration(path=audio_path)
                ),
                "valid_label_frames": int(len(target)),
                "label_start_sec": float(times[0]),
                "label_end_sec": float(times[-1]),
                "valid_label_duration_sec": float(
                    times[-1] - times[0] + 0.5
                ),
                "mean_valence": float(target[:, 0].mean()),
                "std_valence": float(target[:, 0].std()),
                "mean_arousal": float(target[:, 1].mean()),
                "std_arousal": float(target[:, 1].std()),
                "source_legacy_split": str(row.split),
            }
        )
    return pd.DataFrame(records)


def assign_full_song_folds(full_songs: pd.DataFrame) -> pd.DataFrame:
    values = full_songs.copy().sort_values("song_id").reset_index(drop=True)
    values["valence_bin"] = pd.qcut(
        values["mean_valence"], q=2, labels=False
    ).astype(int)
    values["arousal_bin"] = pd.qcut(
        values["mean_arousal"], q=2, labels=False
    ).astype(int)
    values["va_quadrant"] = (
        values["valence_bin"] * 2 + values["arousal_bin"]
    ).astype(int)
    dummy = np.zeros(len(values))
    global_std = {
        column: max(float(values[column].std()), 1e-8)
        for column in (
            "mean_valence",
            "mean_arousal",
            "audio_duration_sec",
        )
    }
    best_score = float("inf")
    best_assignment = None
    best_seed = None
    for offset in range(SPLIT_SEARCH_TRIALS):
        candidate_seed = SEED + offset
        splitter = StratifiedKFold(
            n_splits=FOLDS,
            shuffle=True,
            random_state=candidate_seed,
        )
        assignment = np.full(len(values), -1, dtype=np.int8)
        for fold, (_, held_indices) in enumerate(
            splitter.split(dummy, values["va_quadrant"])
        ):
            assignment[held_indices] = fold
        candidate = values.assign(outer_fold=assignment)
        fold_stats = candidate.groupby("outer_fold").agg(
            mean_valence=("mean_valence", "mean"),
            mean_arousal=("mean_arousal", "mean"),
            audio_duration_sec=("audio_duration_sec", "mean"),
            frames=("valid_label_frames", "sum"),
        )
        score = 0.0
        for column in (
            "mean_valence",
            "mean_arousal",
            "audio_duration_sec",
        ):
            score += float(
                (fold_stats[column].std() / global_std[column]) ** 2
            )
        score += float(
            (
                fold_stats["frames"].std()
                / max(float(fold_stats["frames"].mean()), 1e-8)
            )
            ** 2
        )
        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()
            best_seed = candidate_seed
    values["outer_fold"] = best_assignment
    values["split_seed"] = best_seed
    values["balance_score"] = best_score
    if (values["outer_fold"] < 0).any():
        raise RuntimeError("Some complete songs were not assigned a fold.")
    return values


def build_pdmer_manifest(inventory: pd.DataFrame) -> pd.DataFrame:
    result = inventory[
        ["song_id", "domain", "audio_path", "sequence_path"]
    ].copy()
    result["role"] = np.where(
        result["domain"] == "short_clip", "train", "test"
    )
    result["protocol"] = "pdmer_short_to_full"
    return result[
        [
            "protocol",
            "song_id",
            "domain",
            "role",
            "audio_path",
            "sequence_path",
        ]
    ]


def build_nested_manifest(
    inventory: pd.DataFrame, full_folds: pd.DataFrame
) -> pd.DataFrame:
    fold_by_song = dict(
        zip(full_folds["song_id"], full_folds["outer_fold"])
    )
    records = []
    for outer_fold in range(FOLDS):
        validation_fold = (outer_fold + 1) % FOLDS
        for row in inventory.itertuples():
            if row.domain == "short_clip":
                role = "train"
                full_song_fold = np.nan
            else:
                full_song_fold = int(fold_by_song[int(row.song_id)])
                if full_song_fold == outer_fold:
                    role = "test"
                elif full_song_fold == validation_fold:
                    role = "validation"
                else:
                    role = "train"
            records.append(
                {
                    "outer_fold": outer_fold,
                    "validation_fold": validation_fold,
                    "song_id": int(row.song_id),
                    "domain": row.domain,
                    "role": role,
                    "full_song_fold": full_song_fold,
                    "audio_path": row.audio_path,
                    "sequence_path": row.sequence_path,
                }
            )
    return pd.DataFrame(records)


def summarize(
    inventory: pd.DataFrame,
    full_folds: pd.DataFrame,
    pdmer: pd.DataFrame,
    nested: pd.DataFrame,
) -> dict:
    full = inventory[inventory["domain"] == "full_song"]
    short = inventory[inventory["domain"] == "short_clip"]
    fold_rows = []
    for fold in range(FOLDS):
        held = full_folds[full_folds["outer_fold"] == fold]
        fold_rows.append(
            {
                "fold": fold,
                "songs": int(len(held)),
                "frames": int(held["valid_label_frames"].sum()),
                "mean_valence": float(held["mean_valence"].mean()),
                "std_mean_valence": float(held["mean_valence"].std()),
                "mean_arousal": float(held["mean_arousal"].mean()),
                "std_mean_arousal": float(held["mean_arousal"].std()),
                "mean_audio_duration_sec": float(
                    held["audio_duration_sec"].mean()
                ),
                "quadrant_counts": {
                    str(int(key)): int(value)
                    for key, value in held["va_quadrant"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
            }
        )

    nested_checks = []
    for outer_fold, part in nested.groupby("outer_fold"):
        role_sets = {
            role: set(group["song_id"].astype(int))
            for role, group in part.groupby("role")
        }
        overlap = {
            "train_validation": sorted(
                role_sets["train"] & role_sets["validation"]
            ),
            "train_test": sorted(
                role_sets["train"] & role_sets["test"]
            ),
            "validation_test": sorted(
                role_sets["validation"] & role_sets["test"]
            ),
        }
        counts = (
            part.groupby(["role", "domain"]).size().to_dict()
        )
        nested_checks.append(
            {
                "outer_fold": int(outer_fold),
                "song_counts": {
                    f"{role}:{domain}": int(count)
                    for (role, domain), count in counts.items()
                },
                "overlap": overlap,
                "all_roles_disjoint": not any(overlap.values()),
                "all_short_clips_train_only": bool(
                    (
                        part[part["domain"] == "short_clip"]["role"]
                        == "train"
                    ).all()
                ),
            }
        )

    return {
        "protocol_version": 1,
        "created_date": "2026-06-23",
        "seed": SEED,
        "fold_search_trials": SPLIT_SEARCH_TRIALS,
        "selected_fold_seed": int(full_folds["split_seed"].iloc[0]),
        "selected_balance_score": float(
            full_folds["balance_score"].iloc[0]
        ),
        "song_identity_rule": {
            "short_clip": "song_id <= 2000",
            "full_song": "2001 <= song_id <= 2058",
        },
        "inventory": {
            "total_songs": int(len(inventory)),
            "short_clips": int(len(short)),
            "full_songs": int(len(full)),
            "missing_audio_paths": int(
                sum(not Path(path).is_file() for path in inventory["audio_path"])
            ),
            "missing_sequence_paths": int(
                sum(
                    not Path(path).is_file()
                    for path in inventory["sequence_path"]
                )
            ),
            "short_clip_frames": int(
                short["valid_label_frames"].sum()
            ),
            "full_song_frames": int(
                full["valid_label_frames"].sum()
            ),
        },
        "pdmer_compatible": {
            "train_short_clips": int(
                len(pdmer[(pdmer["role"] == "train")])
            ),
            "test_full_songs": int(
                len(pdmer[(pdmer["role"] == "test")])
            ),
            "song_overlap": sorted(
                set(pdmer[pdmer["role"] == "train"]["song_id"])
                & set(pdmer[pdmer["role"] == "test"]["song_id"])
            ),
        },
        "full_song_outer_folds": fold_rows,
        "nested_protocol_checks": nested_checks,
        "legacy_split_note": (
            "source_legacy_split is retained for traceability only and must "
            "not be used by the full-song protocols."
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = load_inventory()
    full_inventory = inventory[
        inventory["domain"] == "full_song"
    ].copy()
    if len(full_inventory) != 58:
        raise ValueError(
            f"Expected 58 complete songs, found {len(full_inventory)}."
        )
    if len(inventory[inventory["domain"] == "short_clip"]) != 1744:
        raise ValueError("Expected 1744 short clips.")

    full_folds = assign_full_song_folds(full_inventory)
    pdmer = build_pdmer_manifest(inventory)
    nested = build_nested_manifest(inventory, full_folds)
    audit = summarize(inventory, full_folds, pdmer, nested)

    inventory.to_csv(OUTPUT_DIR / "song_inventory.csv", index=False)
    full_folds.to_csv(
        OUTPUT_DIR / "full_song_outer_folds.csv", index=False
    )
    pdmer.to_csv(
        OUTPUT_DIR / "pdmer_short_to_full_manifest.csv", index=False
    )
    nested.to_csv(
        OUTPUT_DIR / "nested_full_song_protocol.csv", index=False
    )
    (OUTPUT_DIR / "protocol_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
