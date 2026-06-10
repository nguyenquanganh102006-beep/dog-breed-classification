from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    num_classes: int


def load_and_prepare_data(
    data_path: Path,
    label_column: str,
    test_size: float,
    val_size: float,
    random_state: int,
    use_scaler: bool,
    pca_components: int,
    artifacts_dir: Path,
    num_classes: int = 0,
    selected_classes: list[str] | None = None,
) -> DatasetSplits:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test-size and val-size must be positive and sum to less than 1.")

    dataframe = pd.read_csv(data_path).dropna()
    if label_column not in dataframe.columns:
        raise ValueError(f"Missing label column: {label_column}")

    available_classes = sorted(dataframe[label_column].unique())
    if num_classes < 0:
        raise ValueError("num-classes cannot be negative.")
    if num_classes > len(available_classes):
        raise ValueError(
            f"num-classes cannot exceed the {len(available_classes)} available classes."
        )
    if selected_classes and num_classes > 0:
        raise ValueError("Use either selected-classes or num-classes, not both.")
    if selected_classes:
        missing_classes = sorted(set(selected_classes) - set(available_classes))
        if missing_classes:
            raise ValueError(f"Classes not found in dataset: {', '.join(missing_classes)}")
        dataframe = dataframe[dataframe[label_column].isin(selected_classes)].copy()
        print(
            f"Selected {len(selected_classes)} classes: "
            f"{', '.join(map(str, selected_classes))}"
        )
    elif num_classes > 0:
        first_classes = available_classes[:num_classes]
        dataframe = dataframe[dataframe[label_column].isin(first_classes)].copy()
        print(f"Selected {num_classes} classes: {', '.join(map(str, first_classes))}")

    X = dataframe.drop(columns=[label_column]).to_numpy(dtype=np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(dataframe[label_column].to_numpy())

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_temp,
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, artifacts_dir / "label_encoder.pkl")
    (artifacts_dir / "selected_classes.txt").write_text(
        "\n".join(map(str, encoder.classes_)) + "\n",
        encoding="utf-8",
    )

    if use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        joblib.dump(scaler, artifacts_dir / "scaler.pkl")

    if pca_components > 0:
        components = min(pca_components, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=components, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        joblib.dump(pca, artifacts_dir / "pca.pkl")
        explained = pca.explained_variance_ratio_.sum() * 100
        print(f"PCA: {components} components, explained variance: {explained:.2f}%")

    return DatasetSplits(
        X_train=np.asarray(X_train, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.int64),
        X_val=np.asarray(X_val, dtype=np.float32),
        y_val=np.asarray(y_val, dtype=np.int64),
        X_test=np.asarray(X_test, dtype=np.float32),
        y_test=np.asarray(y_test, dtype=np.int64),
        num_classes=len(encoder.classes_),
    )
