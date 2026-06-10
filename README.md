# Dog Breed Classification with MLP-Mixer

An image classification project for recognizing dog breeds from portrait images. The project compares classical machine learning baselines with an improved MLP-Mixer model, then provides a local web application for real-image inference.

![Dog breed samples](docs/assets/dog_breed_samples.jpg)

## Overview

The task is supervised image classification:

- **Input:** a dog image.
- **Output:** predicted dog breed.
- **Classes:** 10 visually diverse dog breeds.
- **Main model:** MLP-Mixer, an all-MLP vision architecture that mixes information across image patches and feature channels.

The project includes preprocessing, model training, evaluation, hyperparameter sweeps, checkpoint saving, and a browser-based prediction app.

## Supported Classes

The `diverse10` preset contains:

| # | Breed |
|---|---|
| 1 | Afghan Hound |
| 2 | Basset Hound |
| 3 | Bull Terrier |
| 4 | Chihuahua |
| 5 | Chow Chow |
| 6 | Dalmatian |
| 7 | Great Dane |
| 8 | Greyhound |
| 9 | Pembroke Welsh Corgi |
| 10 | Poodle |

## Results

The final evaluation uses top-k accuracy. Top-1 requires the highest-confidence prediction to be correct; top-3 counts a sample as correct if the true label appears among the three most confident predictions.

| Model | Best setting | Test Top-1 | Test Top-3 |
|---|---:|---:|---:|
| KNN | `k = 1` | 43.50% | 57.33% |
| Linear SVM | `C = 10` | 40.00% | 64.83% |
| Decision Tree | `max_thresholds = 3` | 30.00% | 50.17% |
| MLP-Mixer | improved training recipe | **73.32%** | **86.99%** |

Validation result for the best MLP-Mixer checkpoint:

| Model | Validation Top-1 | Validation Top-3 |
|---|---:|---:|
| MLP-Mixer | 73.46% | 87.12% |

The full project report is available at [docs/report.pdf](docs/report.pdf).

## Method

### Data Collection and Cleaning

Images were collected for 10 dog breeds. Noisy samples such as memes, cartoons, game images, large landscape scenes, and images containing irrelevant subjects were removed. The cleaned dataset keeps approximately 1000 portrait-style images per class.

### Preprocessing

`preprocess_images.py` converts image folders into a flat CSV dataset:

1. Correct image orientation using EXIF metadata.
2. Convert images to RGB.
3. Center-crop and resize to a square image.
4. Convert pixels to numeric arrays.
5. Flatten each image into a feature vector.
6. Normalize pixel values to `[0, 1]`.

### MLP-Mixer Improvements

The final MLP-Mixer implementation includes:

- Patch projection with LayerNorm.
- Token-mixing MLP for communication across image patches.
- Channel-mixing MLP for feature interaction inside each patch.
- Positional embedding.
- Residual connections.
- Learned token pooling.
- DropPath and LayerScale.
- AdamW with proper no-decay parameter groups.
- Warmup + cosine learning-rate schedule.
- Gradient clipping.
- EMA model evaluation.
- Mixup.
- Spatial augmentation: random crop, horizontal flip, and random erasing.

## Repository Structure

```text
.
├── class_presets.py
├── data_utils.py
├── metrics.py
├── preprocess_images.py
├── merge_dog_datasets.py
├── train.py
├── train_library_mlp.py
├── train_knn.py
├── train_svm.py
├── train_decision_tree.py
├── predict_app.py
├── model/
│   ├── base.py
│   ├── knn.py
│   ├── svm.py
│   ├── decision_tree.py
│   └── mlp.py
├── scripts/
│   └── export_app_bundle.py
└── docs/
    ├── APP.md
    ├── DATASET.md
    ├── GITHUB.md
    ├── TRAINING.md
    ├── report.pdf
    └── assets/
        └── dog_breed_samples.jpg
```

Additional documentation:

- [Dataset guide](docs/DATASET.md)
- [Training guide](docs/TRAINING.md)
- [Prediction app guide](docs/APP.md)
- [GitHub checklist](docs/GITHUB.md)

## Installation

```bash
conda create -n dog python=3.11
conda activate dog
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build that matches your NVIDIA driver:

```text
https://pytorch.org/get-started/locally/
```

## Dataset Format

The expected image dataset layout is:

```text
data/merged_dog_dataset_v2/
  Afghan Hound/
    image_001.jpg
  Basset Hound/
    image_001.jpg
  ...
```

The raw dataset and generated CSV files are intentionally not included in this repository.

## Preprocess Images

Create a normalized RGB CSV from image folders:

```bash
python preprocess_images.py \
  --data-dir data/merged_dog_dataset_v2 \
  --output data/dog_dataset_32_new.csv \
  --image-size 64
```

Create a 96x96 version for higher-resolution experiments:

```bash
python preprocess_images.py \
  --data-dir data/merged_dog_dataset_v2 \
  --output data/dog_dataset_96.csv \
  --image-size 96
```

Limit each class to 1000 images:

```bash
python preprocess_images.py \
  --data-dir data/merged_dog_dataset_v2 \
  --output data/dog_dataset_64_1000_each.csv \
  --image-size 64 \
  --limit-per-class 1000
```

## Train the Best MLP-Mixer

```bash
python train.py \
  --model mlp \
  --mlp-type mixer \
  --optimizer adamw \
  --data-path data/dog_dataset_32_new.csv \
  --class-preset diverse10 \
  --device cuda \
  --no-scaler \
  --epochs 200 \
  --patience 40 \
  --warmup-epochs 10 \
  --batch-size 128 \
  --hidden-size 256 \
  --depth 8 \
  --patch-size 4 \
  --token-mlp-size 256 \
  --channel-mlp-size 1024 \
  --learning-rate 5e-4 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --label-smoothing 0.05 \
  --feature-noise 0.005 \
  --feature-drop 0.01 \
  --ema-decay 0.99 \
  --mixup-alpha 0.1 \
  --grad-clip 1.0 \
  --drop-path 0.05 \
  --layer-scale 0.1 \
  --crop-padding 4 \
  --hflip-prob 0.5 \
  --erase-prob 0.25 \
  --erase-scale 0.18 \
  --checkpoint checkpoint/mlp_mixer_diverse10_best.pt
```

## Train Other Models

Original flattened MLP:

```bash
python train.py \
  --model mlp \
  --mlp-type original \
  --data-path data/dog_dataset_32_new.csv \
  --class-preset diverse10 \
  --device cuda \
  --no-scaler
```

KNN sweep:

```bash
python train_knn.py \
  --data-path data/dog_dataset_32_new.csv \
  --class-preset diverse10 \
  --device cuda \
  --no-scaler
```

SVM sweep:

```bash
python train_svm.py \
  --data-path data/dog_dataset_32_new.csv \
  --class-preset diverse10 \
  --device cuda \
  --no-scaler
```

Decision Tree sweep:

```bash
python train_decision_tree.py \
  --data-path data/dog_dataset_32_new.csv \
  --class-preset diverse10 \
  --pca-components 100
```

## Prediction App

Run the local web app after training or downloading a checkpoint:

```bash
python predict_app.py \
  --checkpoint checkpoint/mlp_mixer_diverse10_best.pt \
  --device cuda \
  --sample-data-dir data/merged_dog_dataset_v2 \
  --host 127.0.0.1 \
  --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The app supports selecting, dragging, dropping, or pasting an external image. It shows top-k predictions, confidence scores, sample images, and breed information links.

## Export a Standalone App Bundle

```bash
python scripts/export_app_bundle.py \
  --output-dir dog_mlp_app_bundle \
  --checkpoint checkpoint/mlp_mixer_diverse10_best.pt \
  --label-file artifacts/selected_classes.txt \
  --sample-data-dir data/merged_dog_dataset_v2
```

This creates a small folder that can be copied to another computer for inference without the full training dataset.

## Version-Control Policy

Large generated files are excluded from Git:

```text
data/
artifacts/
checkpoint/
*.csv
*.pkl
*.pt
*.pth
```

If trained checkpoints need to be shared, use GitHub Releases, Google Drive, Hugging Face, or Git LFS.

## References

- Tolstikhin, I. O., et al. "MLP-Mixer: An all-MLP Architecture for Vision." NeurIPS, 2021.
- Loshchilov, I., and Hutter, F. "Decoupled Weight Decay Regularization." arXiv:1711.05101, 2017.
