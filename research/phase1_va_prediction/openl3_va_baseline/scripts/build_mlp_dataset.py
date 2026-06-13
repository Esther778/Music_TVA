from pathlib import Path
import numpy as np


def common_song_ids(feature_dir: Path, label_dir: Path):
    feat_ids = {p.stem for p in feature_dir.glob("*.npy")}
    lab_ids = {p.stem for p in label_dir.glob("*.npy")}
    return sorted(feat_ids & lab_ids, key=lambda x: int(x))


def build_frame_level(song_ids, feature_dir: Path, label_dir: Path):
    X_list, Y_list = [], []
    for sid in song_ids:
        X = np.load(feature_dir / f"{sid}.npy")
        Y = np.load(label_dir / f"{sid}.npy")
        T = min(X.shape[0], Y.shape[0])
        X_list.append(X[-T:].astype(np.float32))
        Y_list.append(Y[:T].astype(np.float32))
    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)
    return X_all, Y_all


def split_frames(X, Y, seed=42, train=0.8, val=0.1):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X = X[idx]
    Y = Y[idx]

    n = len(X)
    n_train = int(n * train)
    n_val = int(n * val)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    X_test, Y_test = X[n_train + n_val:], Y[n_train + n_val:]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def save(out_dir: Path, name: str, X, Y):
    np.save(out_dir / f"X_{name}.npy", X)
    np.save(out_dir / f"Y_{name}.npy", Y)


def main():
    ROOT = Path(__file__).resolve().parents[1]
    feature_dir = ROOT / "artifacts" / "openl3_2hz_features"
    label_dir = ROOT / "artifacts" / "labels_2hz"
    out_dir = ROOT / "artifacts" / "datasets" / "mlp"
    out_dir.mkdir(parents=True, exist_ok=True)

    song_ids = common_song_ids(feature_dir, label_dir)
    X_all, Y_all = build_frame_level(song_ids, feature_dir, label_dir)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_frames(X_all, Y_all, seed=42)

    save(out_dir, "train", X_train, Y_train)
    save(out_dir, "val", X_val, Y_val)
    save(out_dir, "test", X_test, Y_test)

    print(f"[DONE] Saved MLP dataset → {out_dir}")
    print(f"       Frames: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")


if __name__ == "__main__":
    main()
