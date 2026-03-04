import os
import random
import numpy as np
import matplotlib.pyplot as plt
from src.utils import file_operations


def main():
    train_path = "data/preproc/train"
    volume_files = [f for f in os.listdir(train_path) if f.endswith("_volume.npy")]

    if not volume_files:
        print("No volume files found")
        return

    random_file = random.choice(volume_files)
    index = random_file.split("_")[0]
    volume_path = os.path.join(train_path, f"{index}_volume.npy")
    seg_path = os.path.join(train_path, f"{index}_seg.npy")

    volume = file_operations.load_npy(volume_path)
    seg = file_operations.load_npy(seg_path)

    print(f"Loaded case {index}, volume shape: {volume.shape}, seg shape: {seg.shape}")

    # Choose middle slice
    slice_idx = volume.shape[2] // 2
    vol_slice = volume[:, :, slice_idx, :]  # (128,128,3)
    seg_slice = seg[:, :, slice_idx, :]  # (128,128,4)

    # Create seg image
    seg_img = np.zeros((128, 128, 3), dtype=np.uint8)
    # Colors: class 1: red, 2: green, 3: blue, 0: black (background)
    colors = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}
    for c in [1, 2, 3]:
        mask = seg_slice[:, :, c] > 0
        seg_img[mask] = colors[c]

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(vol_slice[:, :, 0], cmap="gray")
    axes[0].set_title("T1CE")
    axes[1].imshow(vol_slice[:, :, 1], cmap="gray")
    axes[1].set_title("T2")
    axes[2].imshow(vol_slice[:, :, 2], cmap="gray")
    axes[2].set_title("FLAIR")
    axes[3].imshow(seg_img)
    axes[3].set_title("Segmentation")

    plt.show()


if __name__ == "__main__":
    main()
