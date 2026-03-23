from typing import Optional
import numpy as np
import torch
import matplotlib.pyplot as plt


class InferencePlotter:
    def plot_prediction(
        self,
        volume: np.ndarray,
        seg_onehot: np.ndarray,
        preds: torch.Tensor,
        slice_idx: Optional[int] = None,
    ) -> None:
        preds_np = preds.squeeze(0).detach().cpu().numpy()

        if slice_idx is None:
            slice_idx = volume.shape[2] // 2

        t1ce_slice = volume[:, :, slice_idx, 0]
        t2_slice = volume[:, :, slice_idx, 1]
        flair_slice = volume[:, :, slice_idx, 2]

        seg_slice = seg_onehot[:, :, slice_idx, :]
        gt_seg_img = self._seg_to_color(seg_slice)

        pred_slice = preds_np[slice_idx, :, :]
        pred_seg_img = self._class_indices_to_color(pred_slice)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(t1ce_slice, cmap="gray")
        axes[0].set_title("T1CE")
        axes[0].axis("off")

        axes[1].imshow(t2_slice, cmap="gray")
        axes[1].set_title("T2")
        axes[1].axis("off")

        axes[2].imshow(flair_slice, cmap="gray")
        axes[2].set_title("FLAIR")
        axes[2].axis("off")

        axes[3].imshow(gt_seg_img)
        axes[3].set_title("Ground Truth")
        axes[3].axis("off")

        axes[4].imshow(pred_seg_img)
        axes[4].set_title("Prediction")
        axes[4].axis("off")

        plt.tight_layout()
        plt.show()

    def _seg_to_color(self, seg_onehot_slice: np.ndarray) -> np.ndarray:
        h, w, _ = seg_onehot_slice.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        colors = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}

        for c, color in colors.items():
            mask = seg_onehot_slice[:, :, c] > 0
            color_img[mask] = color

        return color_img

    def _class_indices_to_color(self, class_slice: np.ndarray) -> np.ndarray:
        h, w = class_slice.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        colors = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}

        for c, color in colors.items():
            mask = class_slice == c
            color_img[mask] = color

        return color_img
