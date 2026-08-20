# OSWRIDK

Open-Set Offline Writer Recognition via Isolation Distributional Kernel-Based Anomaly Detection.

## Overview

This repository implements the OSWRIDK framework. The pipeline consists of four stages:

1. **Preprocessing** — color inversion (`P_R = 255 - P`) followed by sliding-window cropping of a `256 x 256` patch; the single patch with the densest foreground-ink area is retained as the local view.
2. **Feature extraction** — a Dual Branch Network (DBN) with a ResNet50 backbone extracts global (full-image) and local (patch) features, which are concatenated into one representation.
3. **Anomaly detection** — an Isolation Distributional Kernel (IDK) maps each known writer class to a kernel mean embedding and scores a sample by its average Isolation Kernel similarity to that class.
4. **Open-set decision** — a class-specific threshold `mu` is derived from leave-one-out intra-class similarities, the global rejection threshold is `tau = min(mu)`, and a sample is rejected as out-of-distribution when its maximum class similarity falls below `tau`.

The IDK here replaces the original ranking-based decision rule with the threshold mechanism described in the paper.

## Layout

```
config.py            hyperparameters
preprocess.py        color inversion and patch cropping
preprocess_variants.py  binarization, CutMix and PixMix ablations
dbn.py               dual branch network
idk.py               isolation kernel / isolation distributional kernel
metrics.py           AUROC, AUPR, FPR95, F1
dataset.py           data loading
train.py             DBN training (SGD + early stopping)
evaluate.py          open-set evaluation pipeline
main.py              command-line entry point
baselines/           classical anomaly detectors and the anomaly benchmark
```

## Data layout

Place each dataset under `data/<dataset>/` with the following structure:

```
data/<dataset>/
├── train/           one subfolder per writer, containing its training images
│   ├── writer_000/
│   └── ...
├── test/            one subfolder per writer, containing its test images
└── ood/             out-of-distribution images (arbitrary nesting)
```

Writer subfolders are sorted lexicographically to assign class labels; the same writers must appear in `train/` and `test/`.

## Installation

```
pip install -r requirements.txt
```

## Usage

Train the DBN:

```
python main.py --dataset HWDB --train
```

Evaluate the open-set pipeline (prints AUROC, AUPR, FPR95 and F1):

```
python main.py --dataset HWDB --evaluate
```

Both can be run together, and `--model` overrides the default weight path.

## Baselines

The anomaly detectors used in the detector comparison are implemented in `baselines/`:

- `INNEDetector` — isolation via nearest-neighbour ensembles
- `IsolationForestDetector` — scikit-learn Isolation Forest
- `DBSCANDetector` — DBSCAN with a large neighbourhood radius
- `KMeansDetector` — distance to a single cluster centroid
- `SVMDetector` — one-class SVM
- `logit_norm.py` — Logit-Norm OOD scoring from classifier logits

`baselines/anomaly_benchmark.py` reproduces the detector comparison: for each of 100 randomly selected categories, 3 images from other categories are inserted as anomalies, and Precision / Recall / F1 are reported.

## Preprocessing ablation

`preprocess_variants.py` provides the alternative preprocessing methods compared in the ablation study: `binarize` (Otsu thresholding), `cutmix` and `pixmix`. They are applied to the training images before the standard inversion + patch-cropping pipeline; each variant requires retraining the DBN for a fair comparison.

## Key hyperparameters

- `PSI_RATIO` (`psi / m`) — the fraction of class points drawn per partitioning, default `1/3`.
- `T` — number of partitionings, default `200`.
- `PATIENCE`, `DELTA` — DBN early stopping parameters, defaults `2` and `0.05`.
