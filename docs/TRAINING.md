# Training Guide

## Best MLP-Mixer

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

## Original MLP

```bash
python train.py \
  --model mlp \
  --mlp-type original \
  --data-path data/dog_dataset_32_new.csv \
  --class-preset diverse10 \
  --device cuda \
  --no-scaler \
  --epochs 100 \
  --batch-size 512 \
  --hidden-size 256 \
  --depth 5 \
  --learning-rate 0.01 \
  --dropout 0.3 \
  --weight-decay 0.001
```

## Library MLP-Mixer

```bash
python train_library_mlp.py \
  --data-path data/dog_dataset_32_new.csv \
  --class-preset diverse10 \
  --architecture mixer \
  --device cuda \
  --no-scaler
```

## KNN Sweep

```bash
python train_knn.py \
  --data-path data/dog_dataset_32_new.csv \
  --class-preset diverse10 \
  --device cuda \
  --no-scaler \
  --batch-size 64
```

## SVM C Sweep

```bash
python train_svm.py \
  --data-path data/dog_dataset_32_new.csv \
  --class-preset diverse10 \
  --device cuda \
  --no-scaler \
  --epochs 100 \
  --batch-size 256
```

## Decision Tree Sweep

```bash
python train_decision_tree.py \
  --data-path data/dog_dataset_32_new.csv \
  --class-preset diverse10 \
  --pca-components 100 \
  --max-depth 8 \
  --max-threshold-values 3 5 10 20
```

PCA is recommended for decision trees because raw pixel vectors are high-dimensional.
