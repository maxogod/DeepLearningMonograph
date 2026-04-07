import matplotlib.pyplot as plt
import numpy as np
from src.utils import logger


log = logger.get_logger()


def plot_learning_rate_evolution(
    learning_rate: float,
    eta_min_lr: float,
    num_epochs: int,
    save_path: str | None = None,
    show: bool = True,
) -> np.ndarray:
    """Plot cosine annealing learning rate values used across training epochs."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be greater than 0")

    epochs = np.arange(1, num_epochs + 1)
    t = np.arange(0, num_epochs, dtype=np.float64)
    lrs = eta_min_lr + 0.5 * (learning_rate - eta_min_lr) * (
        1.0 + np.cos(np.pi * t / num_epochs)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, lrs, color="tab:blue", linewidth=2)
    ax.set_title("Cosine Annealing Learning Rate Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        log.info(f"Saved LR evolution plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return lrs
