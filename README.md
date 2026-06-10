# Dog Breed Classification With MLP-Mixer

Project phân loại giống chó từ ảnh, tập trung vào việc so sánh các mô hình truyền thống với MLP tự code và MLP-Mixer. Dataset không được đưa trực tiếp lên GitHub vì dung lượng lớn; repo chỉ lưu code, script train, app inference và hướng dẫn tái tạo dữ liệu.

![Dog breed samples](docs/assets/dog_breed_samples.jpg)

## Highlights

- Preprocess ảnh folder thành CSV pixel normalized.
- Train/test nhiều mô hình: KNN, SVM, Decision Tree, MLP tự code và MLP-Mixer.
- MLP-Mixer giữ cấu trúc ảnh qua patch, không dùng convolution.
- Metrics: top-1, top-3, top-5 accuracy.
- App web local để upload/kéo-thả/paste ảnh và dự đoán giống chó bằng checkpoint đã train.

## Project Structure

```text
.
├── class_presets.py              # Preset 5/10 giống chó
├── data_utils.py                 # Load CSV, split train/val/test, scaler/PCA
├── metrics.py                    # Top-k metrics
├── preprocess_images.py          # Convert image folders -> CSV
├── train.py                      # Trainer chung cho KNN/SVM/Tree/MLP/Mixer
├── train_library_mlp.py          # PyTorch MLP/Mixer training script nâng cao
├── train_knn.py                  # Sweep KNN k
├── train_svm.py                  # Sweep SVM C
├── train_decision_tree.py        # Sweep Decision Tree thresholds
├── predict_app.py                # Local web app inference
├── scripts/
│   └── export_app_bundle.py      # Export minimal app bundle
├── model/
│   ├── base.py
│   ├── knn.py
│   ├── svm.py
│   ├── decision_tree.py
│   └── mlp.py                   # Original MLP + MLP-Mixer
└── docs/assets/
    └── dog_breed_samples.jpg
```

More docs:

- [Dataset guide](docs/DATASET.md)
- [Training guide](docs/TRAINING.md)
- [Prediction app guide](docs/APP.md)
- [GitHub checklist](docs/GITHUB.md)

## What To Push To GitHub

Nên push:

```text
.gitignore
.gitattributes
.env.example
README.md
requirements.txt
class_presets.py
data_utils.py
metrics.py
preprocess_images.py
merge_dog_datasets.py
train.py
train_library_mlp.py
train_knn.py
train_svm.py
train_decision_tree.py
predict_app.py
model/
docs/
scripts/
```

Không nên push:

```text
data/
artifacts/
checkpoint/
__pycache__/
dog_mlp_app_bundle/
dog_mlp_app_bundle.zip
*.csv
*.pkl
*.pt
*.pth
```

Lý do: `data/` có thể lên tới hàng GB, checkpoint/artifacts là output train, không nên để GitHub thường lưu. Nếu cần chia sẻ model đã train, dùng GitHub Releases, Google Drive, Hugging Face, hoặc Git LFS.

## Setup

```bash
conda create -n dog python=3.11
conda activate dog
pip install -r requirements.txt
```

Nếu dùng GPU, cài PyTorch CUDA phù hợp với driver:

```text
https://pytorch.org/get-started/locally/
```

## Dataset Layout

Dataset ảnh cần có dạng mỗi class là một folder:

```text
data/merged_dog_dataset_v2/
  Afghan Hound/
    image_001.jpg
  Basset Hound/
    image_001.jpg
  ...
```

Preset `diverse10` hiện gồm:

```text
Afghan Hound
Basset Hound
Bull Terrier
Chihuahua
Chow Chow
Dalmatian
Great Dane
Greyhound
Pembroke Welsh Corgi
Poodle
```

## Preprocess Images To CSV

Tạo CSV ảnh `64x64x3`, normalized về `[0, 1]`:

```bash
python preprocess_images.py \
  --data-dir data/merged_dog_dataset_v2 \
  --output data/dog_dataset_32_new.csv \
  --image-size 64
```

Tạo bản `96x96x3` để thử tăng accuracy:

```bash
python preprocess_images.py \
  --data-dir data/merged_dog_dataset_v2 \
  --output data/dog_dataset_96.csv \
  --image-size 96
```

Giới hạn mỗi class 1000 ảnh:

```bash
python preprocess_images.py \
  --data-dir data/merged_dog_dataset_v2 \
  --output data/dog_dataset_64_1000_each.csv \
  --image-size 64 \
  --limit-per-class 1000
```

## Train Best MLP-Mixer

Lệnh train MLP-Mixer mạnh nhất trong `train.py`:

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

Các cơ chế đang dùng trong MLP-Mixer:

- Patch projection với LayerNorm.
- Token mixing và channel mixing.
- Positional embedding.
- Learned token pooling.
- DropPath + LayerScale.
- AdamW với param groups không decay bias/norm.
- Warmup LR + cosine decay.
- Gradient clipping.
- EMA model.
- Mixup.
- Spatial augmentation: crop, horizontal flip, random erasing.

## Train Original MLP

MLP nguyên bản flatten ảnh thành vector, không giữ cấu trúc không gian:

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

## Train Library MLP-Mixer

`train_library_mlp.py` là bản PyTorch-library đầy đủ hơn, có DataLoader, AMP, EMA và checkpoint riêng:

```bash
python train_library_mlp.py \
  --data-path data/dog_dataset_32_new.csv \
  --class-preset diverse10 \
  --architecture mixer \
  --device cuda \
  --no-scaler
```

## Other Models

KNN sweep:

```bash
python train_knn.py \
  --data-path data/dog_dataset_32_new.csv \
  --class-preset diverse10 \
  --device cuda \
  --no-scaler \
  --batch-size 64
```

SVM C sweep:

```bash
python train_svm.py \
  --data-path data/dog_dataset_32_new.csv \
  --class-preset diverse10 \
  --device cuda \
  --no-scaler \
  --epochs 100 \
  --batch-size 256
```

Decision Tree sweep:

```bash
python train_decision_tree.py \
  --data-path data/dog_dataset_32_new.csv \
  --class-preset diverse10 \
  --pca-components 100 \
  --max-depth 8 \
  --max-threshold-values 3 5 10 20
```

## Run Prediction App

Sau khi có checkpoint:

```bash
python predict_app.py \
  --checkpoint checkpoint/mlp_mixer_diverse10_best.pt \
  --device cuda \
  --sample-data-dir data/merged_dog_dataset_v2 \
  --host 127.0.0.1 \
  --port 8000
```

Mở:

```text
http://127.0.0.1:8000
```

App hỗ trợ:

- Click chọn ảnh.
- Kéo-thả ảnh vào khung.
- Paste ảnh bằng `Ctrl+V`.
- Preview ảnh upload.
- Hiển thị top-k nhãn, xác suất, ảnh mẫu và link thông tin giống chó.

## Export App Bundle

Nếu muốn chạy app trên máy cá nhân mà không copy toàn bộ project, tạo/copy folder bundle gồm:

```bash
python scripts/export_app_bundle.py \
  --output-dir dog_mlp_app_bundle \
  --checkpoint checkpoint/mlp_mixer_diverse10_best.pt \
  --label-file artifacts/selected_classes.txt \
  --sample-data-dir data/merged_dog_dataset_v2
```

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

Folder này chỉ khoảng vài chục MB nếu mỗi class chỉ giữ một ảnh mẫu.

## Notes

- Không push `data/`, `checkpoint/`, `artifacts/` lên GitHub thường.
- Nếu muốn public checkpoint, dùng GitHub Release hoặc Hugging Face.
- MLP-Mixer yêu cầu raw normalized pixels, vì vậy khi train Mixer cần `--no-scaler` và không dùng PCA.
- Với ảnh nhỏ `64x64`, CNN vẫn có lợi thế inductive bias. MLP-Mixer cải thiện bằng patch/token mixing nhưng không phải convolution.
