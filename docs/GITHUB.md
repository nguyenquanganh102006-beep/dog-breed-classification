# GitHub Checklist

## Commit These

```bash
git add \
  .gitattributes \
  .gitignore \
  .env.example \
  README.md \
  requirements.txt \
  class_presets.py \
  data_utils.py \
  metrics.py \
  preprocess_images.py \
  merge_dog_datasets.py \
  train.py \
  train_library_mlp.py \
  train_knn.py \
  train_svm.py \
  train_decision_tree.py \
  predict_app.py \
  model \
  docs \
  scripts
```

## Do Not Commit These

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

## First Push

```bash
git commit -m "Add dog breed classification project"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

If GitHub already has a commit:

```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

Use force push only if the remote repo is disposable:

```bash
git push -u origin main --force
```
