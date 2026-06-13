# prepare_labels.py

from pathlib import Path
import numpy as np


def load_csv(csv_path: Path):
    labels = {}
    with csv_path.open("r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            sid = int(parts[0])
            labels[sid] = np.asarray(parts[1:], dtype=np.float32)
    return labels


def build_va(valence, arousal):
    va = {}
    for sid in sorted(valence.keys() & arousal.keys()):
        T = min(len(valence[sid]), len(arousal[sid]))
        va[sid] = np.stack([valence[sid][:T], arousal[sid][:T]], axis=-1)
    return va


def main():
    ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（scripts 的上一级）

    deam_ann_dir = (
        ROOT
        / "data"
        / "DEAM"
        / "annotations"
        / "annotations averaged per song"
        / "dynamic (per second annotations)"
    )

    valence_csv = deam_ann_dir / "valence.csv"
    arousal_csv = deam_ann_dir / "arousal.csv"

    out_dir = ROOT / "artifacts" / "labels_2hz"
    out_dir.mkdir(parents=True, exist_ok=True)

    valence = load_csv(valence_csv)
    arousal = load_csv(arousal_csv)
    va_labels = build_va(valence, arousal)

    for sid, va in va_labels.items():
        np.save(out_dir / f"{sid}.npy", va)

    print(f"[DONE] Saved {len(va_labels)} VA label files → {out_dir}")


if __name__ == "__main__":
    main()
