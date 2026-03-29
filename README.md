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
conda env update --name brats --file environment.yaml --prune
```

## How to run

Open the `config.yaml` file and change the program settings according to whats needed.

**Most important settings:**
- preprocess(`False/True`): indicates whether to start preprocessing the raw data to generate the `data/preproc` folder (must be done once)
- train(`False/True`): indicates whether to start training with the settings or not
- evaluate(`False/True`): indicates whether to evaluate the model with validation data and print metrics
- predict(`False/True`): indicates whether to infer (and plot) a segmentation on a given RMI
- plot_loss(`False/True`): indicates whether to plot the evolution of train/val loss during training of specified model

Finally run the program

```bash
python -m src.main
# If predict is True add a path to a raw RMI folder to use infer the segmentation and plot it
# e.g. python -m src.main data/raw/training/BraTS20_Training_324
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

### Dataset

- Visit: [[Brats 2020 with .nii (42.8GB)]](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

> The dataset contains 369 cases of the described problem above. <br/>
<small>Entry number 355 has a bad file name! rename accordingly</small>

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
