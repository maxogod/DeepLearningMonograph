# Brain Tumor Segmentation Models & Analysis

Application of deep learning techniques and models to the semantic segmentation of tumors in brain tissue
based on RMI 3D volumes.

## How to setup environment

- First install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
- Then create the environment with its dependencies as follows

```bash
# Create the environment
conda env create -f ./environment.yaml
conda activate brats

# Update after changing environment.yaml
conda env update --name brats --file environment.yml --prune
```

## How to run

```bash
python -m src.main
```

## Model description

- 3D U-Net

## Problem description

- Each brain has 4 files (volumes) of different types of RMIs therefore we have 4 Channels of data for one brain.
    - T1
    - T1CE
    - T2
    - FLAIR (T2 Fluid attenuated Inversion Recovery)
- Images were segmented manually and approved by neuro-radiologists.
- Labels in the segmentaion volume are (excluding label 3 which has no data):
    - 0: unlabeled (non useful area)
    - 1: NCR/NET (necrotic & non-enhancing tumor)
    - 2: ED (edema)
    - 4: ET (enhancing tumor)

- Dataset: [[Brats 2020 with .nii (42.8GB)]](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

## Proposed solution

- Scale volumes (voxels have uneven very low and high values and they vary between volumes)
- Stack volumes into 1 volume 3 channels (T1CE, T2, FLAIR)
- Move label 4 to label 3
- Crop 128x128x128 (remove as much air as possible)
- Drop volumes where anotated data is low
- Save as npy files
- Split train/val datasets
- Lazy loader to get volumes as a generator
- Train in batches and validate
