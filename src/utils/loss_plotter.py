import matplotlib.pyplot as plt
import numpy as np


def plot_loss_history(
    loss_history_file: str,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot train/validation loss curves from a checkpoint loss history .npy file.

    Expected array shape is (N, 3) with columns:
    - epoch index
    - train loss
    - validation loss (can contain NaN values)
    """
    history = np.load(loss_history_file)

    if history.ndim != 2 or history.shape[1] != 3:
        raise ValueError(
            "Loss history must have shape (N, 3) with [epoch, train_loss, val_loss]."
        )

    epochs = history[:, 0]
    train_loss = history[:, 1]
    val_loss = history[:, 2]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)

    valid_val_mask = ~np.isnan(val_loss)
    if np.any(valid_val_mask):
        ax.plot(
            epochs[valid_val_mask],
            val_loss[valid_val_mask],
            label="Validation Loss",
            linewidth=2,
        )

    ax.set_title("Loss Evolution During Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
