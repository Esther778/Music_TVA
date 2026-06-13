from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 项目根目录
    ROOT = Path(__file__).resolve().parents[1]

    # 路径（统一命名）
    history_path = ROOT / "outputs" / "results" / "transformer" / "transformer_history.npz"
    out_dir = ROOT / "outputs" / "plots" / "transformer"
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = np.load(history_path)
    print("[INFO] keys in history:", hist.files)

    train_loss = hist["train_loss"]
    val_loss = hist["val_loss"]
    val_rmse = hist["val_rmse"]
    val_pv = hist["val_pearson_v"]
    val_pa = hist["val_pearson_a"]

    epochs = np.arange(1, len(train_loss) + 1)

    # -------- 1. Loss 曲线 --------
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transformer Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "transformer_loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------- 2. RMSE 曲线 --------
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_rmse, label="Val RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Transformer Validation RMSE")
    plt.grid(True)
    plt.savefig(out_dir / "transformer_val_rmse_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------- 3. Pearson 曲线 --------
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_pv, label="Pearson Valence")
    plt.plot(epochs, val_pa, label="Pearson Arousal")
    plt.xlabel("Epoch")
    plt.ylabel("Pearson r")
    plt.title("Transformer Validation Pearson Correlation")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "transformer_val_pearson_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[DONE] Saved Transformer training plots → {out_dir}")


if __name__ == "__main__":
    main()
