# Prediction App

The local app loads a trained MLP-Mixer checkpoint and predicts dog breeds from external images.

## Run

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

## Features

- Click to select an image.
- Drag and drop an image.
- Paste image from clipboard with `Ctrl+V`.
- Preview uploaded image.
- Show top-k predictions with confidence bars.
- Show sample image and breed information links.

## Minimal App Bundle

For running on another computer, create a small bundle containing:

```text
dog_mlp_app_bundle/
  predict_app.py
  checkpoint/mlp_mixer_diverse10_best.pt
  artifacts/selected_classes.txt
  model/
  sample_images/
  requirements_app.txt
  README.md
```

The bundle does not need the full training dataset.
