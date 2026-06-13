from pathlib import Path
import numpy as np

SEQ_LEN = 60
SEED = 42


def common_song_ids(feature_dir: Path, label_dir: Path):
    feat_ids = {p.stem for p in feature_dir.glob("*.npy")}
    lab_ids = {p.stem for p in label_dir.glob("*.npy")}
    return sorted(feat_ids & lab_ids, key=lambda x: int(x))


def build_seq_dataset(song_ids, feature_dir: Path, label_dir: Path, seq_len: int):
    X_list, Y_list = [], []

    for sid in song_ids:
        X = np.load(feature_dir / f"{sid}.npy")
        Y = np.load(label_dir / f"{sid}.npy")

        T = min(X.shape[0], Y.shape[0])
        if T < seq_len:
            continue

        X_list.append(X[-seq_len:].astype(np.float32))
        Y_list.append(Y[-seq_len:].astype(np.float32))

    X_all = np.stack(X_list, axis=0)
    Y_all = np.stack(Y_list, axis=0)
    return X_all, Y_all


def split_by_song(X, Y, seed=42, train=0.8, val=0.1):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    X = X[idx]
    Y = Y[idx]

    n = X.shape[0]
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
    out_dir = ROOT / "artifacts" / "datasets" / "seq"
    out_dir.mkdir(parents=True, exist_ok=True)

    song_ids = common_song_ids(feature_dir, label_dir)
    X_all, Y_all = build_seq_dataset(song_ids, feature_dir, label_dir, SEQ_LEN)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_by_song(X_all, Y_all, seed=SEED)

    save(out_dir, "train", X_train, Y_train)
    save(out_dir, "val", X_val, Y_val)
    save(out_dir, "test", X_test, Y_test)

    print(f"[DONE] Saved SEQ dataset → {out_dir}")
    print(f"       Seqs: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")


if __name__ == "__main__":
    main()
