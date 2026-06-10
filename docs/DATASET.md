# Dataset Guide

Dataset không đi kèm repo vì dung lượng lớn. Project kỳ vọng dữ liệu ảnh có cấu trúc mỗi giống chó là một thư mục.

## Folder Layout

```text
data/merged_dog_dataset_v2/
  Afghan Hound/
    image_001.jpg
    image_002.jpg
  Basset Hound/
    image_001.jpg
  ...
```

## Diverse10 Classes

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

## Convert Images To CSV

64x64 RGB:

```bash
python preprocess_images.py \
  --data-dir data/merged_dog_dataset_v2 \
  --output data/dog_dataset_32_new.csv \
  --image-size 64
```

96x96 RGB:

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

## CSV Format

```text
feature_0,feature_1,...,feature_N,label
0.23,0.18,...,0.91,Poodle
```

Pixel values are normalized to `[0, 1]`.
