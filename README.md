# Weed detection notebook

This repository contains a single Kaggle-oriented notebook, `weed-detection.ipynb`. It does **not** include the image dataset, COCO annotations, trained weights, or a reproducible environment lockfile. No model-performance claim is made here; the notebook has not been run against the unavailable dataset as part of this maintenance change.

## Expected inputs

The notebook assumes a Kaggle runtime and these paths:

- Images under `/kaggle/input/weed-detection/train` and `/kaggle/input/weed-detection/test`.
- A COCO annotation file at `/kaggle/working/_annotations.coco.json` (the notebook does not download or create this source annotation file).

Before running, attach the matching dataset in Kaggle and place the annotation file at the expected working path. Confirm that the annotation file refers to the same training images. These paths and the Kaggle runtime are assumptions, not data supplied by this repository.

## Running

Open the notebook in a Kaggle notebook session with the expected dataset attached, then run cells in order. The training cell installs the currently available `ultralytics` package and downloads/loads the YOLOv8 `yolov8s.pt` starting weights as needed. Other imports used include Pillow, Matplotlib, tqdm, OpenCV, Albumentations, scikit-learn, and PyYAML; versions are not pinned, so exact runtime reproducibility is not guaranteed. Training can require substantial time, compute, and network access.

The notebook's setup cell now preserves existing files in `/kaggle/working` instead of recursively deleting that entire directory. It still writes generated artifacts to Kaggle working-directory paths; inspect the cells before running them in any other environment.

## Data split and outputs

The split now happens on original images first, then generated `aug_<number>_<original filename>` images follow their original into either train or validation. This prevents the same source image (or its augmentation) from appearing on both sides of the split. Inputs are sorted before applying the fixed split seed for more stable grouping.

Saved notebook outputs and execution counts were cleared because they were stale, large, and tied to an unavailable dataset/runtime. Run the notebook yourself to generate new results. No training, validation, or accuracy result has been verified by this repository change.

## Local checks

Run the standard-library regression and notebook-syntax checks with:

```bash
python3 -m unittest discover -s tests -v
```

These checks cover notebook source syntax, preservation of the working directory, and grouping of augmentations with their source images. They do not validate the dataset, model training, or predictive performance.
