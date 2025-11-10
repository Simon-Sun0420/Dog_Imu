#!/usr/bin/env python3
"""
Python re-implementation of the Matlab analysis pipeline that accompanies
“Automatic behaviour recognition in accelerometer data from dogs”.

Steps mirrored:
1. Load the processed_dog_features.pkl produced by process_dog_data.py
   (equivalent to CalcFeatures_Acc_Gyro.m + CollectFeatures.m).
2. Rank features with ReliefF for k ∈ {3,5,9,13,17,21} and keep the union
   of the best 20 unique variables per sensor.
3. Run sequential forward selection (with Leave-One-Dog-Out CV) separately
   for LDA, QDA, and RBF-SVM classifiers, identical to SelectFeatures_DA_SVM.m.
4. Evaluate each classifier per sensor with Leave-One-Dog-Out CV, report
   accuracy/F1, confusion matrices, and per-dog correct rates (as in
   ClassifyDogs_*.m).
"""

import warnings

warnings.filterwarnings("ignore")

import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

RESULTS_DIR = Path("results")
K_VALUES = [3, 5, 9, 13, 17, 21]
MAX_RELIEF_VARS = 20


# ---------------------------------------------------------------------------
# ReliefF implementation
# ---------------------------------------------------------------------------
def relieff(X: np.ndarray, y: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Compute ReliefF weights for multi-class problems."""
    n_samples, n_features = X.shape
    classes, counts = np.unique(y, return_counts=True)
    class_prob = {cls: cnt / n_samples for cls, cnt in zip(classes, counts)}

    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, n_samples), metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)

    weights = np.zeros(n_features, dtype=np.float64)
    for i in range(n_samples):
        Xi = X[i]
        yi = y[i]
        hits = []
        misses: Dict[int, List[int]] = {cls: [] for cls in classes if cls != yi}

        for neighbor_idx in indices[i][1:]:
            cls = y[neighbor_idx]
            if cls == yi and len(hits) < n_neighbors:
                hits.append(neighbor_idx)
            elif cls != yi and len(misses[cls]) < n_neighbors:
                misses[cls].append(neighbor_idx)
            if len(hits) == n_neighbors and all(len(m) == n_neighbors for m in misses.values()):
                break

        if not hits:
            continue

        hit_diff = np.abs(Xi - X[hits]).mean(axis=0)
        weights -= hit_diff / (n_neighbors * n_samples)

        denom = max(1e-9, 1.0 - class_prob[yi])
        for cls, miss_list in misses.items():
            if not miss_list:
                continue
            miss_diff = np.abs(Xi - X[miss_list]).mean(axis=0)
            weights += (class_prob[cls] / denom) * miss_diff / (n_neighbors * n_samples)
    return weights


def build_relief_pool(X: np.ndarray, y_encoded: np.ndarray) -> Tuple[List[int], Dict[int, np.ndarray]]:
    """Rank features for each k and return the pooled top variables."""
    rankings = {}
    pooled: List[int] = []
    for k in K_VALUES:
        weights = relieff(X, y_encoded, k)
        order = np.argsort(weights)[::-1]
        rankings[k] = order
        for idx in order[:MAX_RELIEF_VARS]:
            if idx not in pooled:
                pooled.append(int(idx))
    return pooled, rankings


# ---------------------------------------------------------------------------
# Sequential forward selection with LODO CV
# ---------------------------------------------------------------------------
def cross_val_predict_indices(
    estimator, X: np.ndarray, y: np.ndarray, groups: np.ndarray, feature_idx: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stacked true/predicted labels and dog IDs for a feature subset."""
    logo = LeaveOneGroupOut()
    true_labels, pred_labels, dog_list = [], [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        model = clone(estimator)
        model.fit(X[train_idx][:, feature_idx], y[train_idx])
        preds = model.predict(X[test_idx][:, feature_idx])
        true_labels.extend(y[test_idx])
        pred_labels.extend(preds)
        dog_list.extend(groups[test_idx])
    return np.asarray(true_labels), np.asarray(pred_labels), np.asarray(dog_list)


def evaluate_subset(estimator, X: np.ndarray, y: np.ndarray, groups: np.ndarray, subset: Sequence[int]) -> float:
    """Accuracy of a feature subset under Leave-One-Dog-Out CV."""
    if not subset:
        return 0.0
    true_labels, pred_labels, _ = cross_val_predict_indices(estimator, X, y, groups, subset)
    return accuracy_score(true_labels, pred_labels)


def sequential_forward_selection(
    estimator, X: np.ndarray, y: np.ndarray, groups: np.ndarray, candidates: List[int]
) -> List[int]:
    """Greedy forward selection with early stop when accuracy saturates."""
    remaining = candidates.copy()
    selected: List[int] = []
    best_score = 0.0

    while remaining:
        subset_best_score = best_score
        subset_best_feature = None
        for idx in remaining:
            trial_subset = selected + [idx]
            score = evaluate_subset(estimator, X, y, groups, trial_subset)
            if score > subset_best_score + 1e-4:
                subset_best_score = score
                subset_best_feature = idx
        if subset_best_feature is None:
            break
        selected.append(subset_best_feature)
        remaining.remove(subset_best_feature)
        best_score = subset_best_score
    if not selected:
        selected = remaining[:5]  # fallback
    return selected


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def save_confusion_matrix(sensor: str, model_name: str, classes: List[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"{sensor} - {model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)
    plt.tight_layout()
    outfile = RESULTS_DIR / f"{sensor.lower()}_{model_name.lower()}_confusion.png"
    plt.savefig(outfile)
    plt.close()


def save_report(sensor: str, model_name: str, classes: List[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    report = classification_report(y_true, y_pred, labels=classes)
    outfile = RESULTS_DIR / f"{sensor.lower()}_{model_name.lower()}_report.txt"
    with open(outfile, "w") as fh:
        fh.write(report)


def per_dog_accuracy(y_true: np.ndarray, y_pred: np.ndarray, dog_ids: np.ndarray) -> Dict[int, float]:
    dog_rates: Dict[int, float] = {}
    for dog in np.unique(dog_ids):
        mask = dog_ids == dog
        if mask.any():
            dog_rates[int(dog)] = accuracy_score(y_true[mask], y_pred[mask])
    return dog_rates


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def load_processed_data(path: str) -> Dict:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    return data


def zscore_matrix(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (matrix - mean) / std


def run_sensor_pipeline(sensor: str, sensor_df: pd.DataFrame, labels: pd.Series, dog_ids: pd.Series) -> None:
    print(f"\n=== Sensor: {sensor} ===")
    feature_names = sensor_df.columns.tolist()
    X = sensor_df.to_numpy(dtype=np.float32)
    Xz = zscore_matrix(X)
    y = labels.to_numpy()
    groups = dog_ids.to_numpy()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    relief_pool, rankings = build_relief_pool(Xz, y_encoded)
    print(f"ReliefF candidate pool size: {len(relief_pool)}")

    models = {
        # "LDA": LinearDiscriminantAnalysis(),
        # "QDA": QuadraticDiscriminantAnalysis(),
        "SVM": SVC(kernel="rbf", gamma="scale", C=1.0, decision_function_shape="ovo"),
    }

    for model_name, estimator in models.items():
        print(f"\nSelecting features for {model_name} ...")
        selected_idx = sequential_forward_selection(estimator, Xz, y, groups, relief_pool)
        selected_names = [feature_names[i] for i in selected_idx]
        print(f"Selected {len(selected_idx)} features: {selected_names}")

        y_true, y_pred, dog_list = cross_val_predict_indices(estimator, Xz, y, groups, selected_idx)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        print(f"{sensor} - {model_name}: Acc={acc:.3f}, F1_macro={f1:.3f}")

        classes = label_encoder.classes_.tolist()
        save_confusion_matrix(sensor, model_name, classes, y_true, y_pred)
        save_report(sensor, model_name, classes, y_true, y_pred)

        dog_rates = per_dog_accuracy(y_true, y_pred, dog_list)
        print("Per-dog accuracy:")
        for dog, rate in sorted(dog_rates.items()):
            print(f"  Dog {dog}: {rate:.3f}")


def main():
    data = load_processed_data("processed_dog_features.pkl")
    labels = pd.Series(data["labels"])
    dog_ids = pd.Series(data["dog_ids"])
    sensors = data["sensors"]

    for sensor_name, sensor_df in sensors.items():
        if isinstance(sensor_df, pd.DataFrame):
            df = sensor_df
        else:
            df = pd.DataFrame(sensor_df, columns=data["feature_columns"])
        run_sensor_pipeline(sensor_name, df, labels, dog_ids)


if __name__ == "__main__":
    main()
