from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_history(path: Path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def plot_loss(history, out_dir: Path):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("MLP Training / Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    out_path = out_dir / "mlp_loss_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rmse(history, out_dir: Path):
    val_rmse = history["val_rmse"]
    epochs = np.arange(1, len(val_rmse) + 1)

    plt.figure()
    plt.plot(epochs, val_rmse)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (Valence + Arousal)")
    plt.title("MLP Validation RMSE")
    plt.grid(True, linestyle="--", alpha=0.5)

    out_path = out_dir / "mlp_val_rmse_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pearson(history, out_dir: Path):
    pv = history["val_pearson_v"]
    pa = history["val_pearson_a"]
    epochs = np.arange(1, len(pv) + 1)

    plt.figure()
    plt.plot(epochs, pv, label="Pearson(V)")
    plt.plot(epochs, pa, label="Pearson(A)")
    plt.xlabel("Epoch")
    plt.ylabel("Pearson r")
    plt.title("MLP Validation Pearson Correlation")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    out_path = out_dir / "mlp_val_pearson_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ROOT = Path(__file__).resolve().parents[1]
    history_path = ROOT / "outputs" / "results" / "mlp" / "mlp_history.npz"
    out_dir = ROOT / "outputs" / "plots" / "mlp"
    out_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(history_path)

    plot_loss(history, out_dir)
    plot_rmse(history, out_dir)
    plot_pearson(history, out_dir)

    print(f"[DONE] Saved MLP training plots → {out_dir}")


if __name__ == "__main__":
    main()
