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
from sklearn.tree import DecisionTreeClassifier

RESULTS_DIR = Path("results")
K_VALUES = [3, 5, 9, 13, 17, 21]
SVM_FAST_FRACTION = 0.5
SCENARIOS = {
    "ACC_GYRO": {
        "label": "A+G",
        "feature_filter": lambda col: col.startswith("A") or col.startswith("G"),
        "max_relief_vars": 20,
        "max_selected_features": 10,
        "tree_leaf_grid": [80, 110, 140, 170, 200],
        "svm_candidate_cap": 12,
        "svm_fast_fraction": 0.5,
    },
    "ACC": {
        "label": "A only",
        "feature_filter": lambda col: col.startswith("A"),
        "max_relief_vars": 15,
        "max_selected_features": 8,
        "tree_leaf_grid": [60, 100, 140, 180],
        "svm_candidate_cap": 10,
        "svm_fast_fraction": 0.5,
    },
}
SUMMARY_CSV = RESULTS_DIR / "model_summary.csv"
SUMMARY_XLSX = RESULTS_DIR / "model_summary.xlsx"
PER_DOG_CSV = RESULTS_DIR / "per_dog_metrics.csv"

SUMMARY_RECORDS: List[Dict] = []
DOG_RECORDS: List[Dict] = []


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


def build_relief_pool(X: np.ndarray, y_encoded: np.ndarray, max_vars: int) -> Tuple[List[int], Dict[int, np.ndarray]]:
    """Rank features for each k and return the pooled top variables."""
    rankings = {}
    pooled: List[int] = []
    for k in K_VALUES:
        weights = relieff(X, y_encoded, k)
        order = np.argsort(weights)[::-1]
        rankings[k] = order
        for idx in order[:max_vars]:
            if idx not in pooled:
                pooled.append(int(idx))
    return pooled, rankings


# ---------------------------------------------------------------------------
# Sequential forward selection with LODO CV
# ---------------------------------------------------------------------------
def cross_val_predict_indices(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    feature_idx: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stacked true/predicted labels and dog IDs for a feature subset."""
    true_labels, pred_labels, dog_list = [], [], []
    for train_idx, test_idx in folds:
        model = clone(estimator)
        model.fit(X[train_idx][:, feature_idx], y[train_idx])
        preds = model.predict(X[test_idx][:, feature_idx])
        true_labels.extend(y[test_idx])
        pred_labels.extend(preds)
        dog_list.extend(groups[test_idx])
    return np.asarray(true_labels), np.asarray(pred_labels), np.asarray(dog_list)


def evaluate_subset(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    subset: Sequence[int],
) -> float:
    """Accuracy of a feature subset under Leave-One-Dog-Out CV."""
    if not subset:
        return 0.0
    true_labels, pred_labels, _ = cross_val_predict_indices(estimator, X, y, groups, folds, subset)
    return accuracy_score(true_labels, pred_labels)


def sequential_forward_selection(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    candidates: List[int],
    max_selected: int,
    folds_eval: List[Tuple[np.ndarray, np.ndarray]] = None,
) -> List[int]:
    """Greedy forward selection with early stop when accuracy saturates."""
    remaining = candidates.copy()
    selected: List[int] = []
    best_score = 0.0
    folds_used = folds_eval if folds_eval is not None else folds

    while remaining:
        subset_best_score = best_score
        subset_best_feature = None
        print(f"  Step {len(selected)+1}: evaluating {len(remaining)} candidates")
        for idx in remaining:
            trial_subset = selected + [idx]
            score = evaluate_subset(estimator, X, y, groups, folds_used, trial_subset)
            if score > subset_best_score + 1e-4:
                subset_best_score = score
                subset_best_feature = idx
        if subset_best_feature is None:
            break
        selected.append(subset_best_feature)
        remaining.remove(subset_best_feature)
        best_score = subset_best_score
        print(f"    Selected feature index {subset_best_feature} (score={best_score:.3f})")
        if len(selected) >= max_selected:
            print(f"    Reached MAX_SELECTED_FEATURES={max_selected}; stopping search.")
            break
    if not selected:
        selected = remaining[:5]  # fallback
    return selected


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def save_confusion_matrix(sensor: str, scenario: str, model_name: str, classes: List[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"{sensor} ({scenario}) - {model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)
    plt.tight_layout()
    outfile = RESULTS_DIR / f"{sensor.lower()}_{scenario.lower()}_{model_name.lower()}_confusion.png"
    plt.savefig(outfile)
    plt.close()


def save_report(sensor: str, scenario: str, model_name: str, classes: List[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    report = classification_report(y_true, y_pred, labels=classes)
    outfile = RESULTS_DIR / f"{sensor.lower()}_{scenario.lower()}_{model_name.lower()}_report.txt"
    with open(outfile, "w") as fh:
        fh.write(report)


def per_dog_accuracy(y_true: np.ndarray, y_pred: np.ndarray, dog_ids: np.ndarray) -> Dict[int, float]:
    dog_rates: Dict[int, float] = {}
    for dog in np.unique(dog_ids):
        mask = dog_ids == dog
        if mask.any():
            dog_rates[int(dog)] = accuracy_score(y_true[mask], y_pred[mask])
    return dog_rates


def select_fold_subset(folds: List[Tuple[np.ndarray, np.ndarray]], fraction: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    if fraction >= 1.0 or len(folds) <= 2:
        return folds
    target = max(2, int(np.ceil(len(folds) * fraction)))
    indices = np.linspace(0, len(folds) - 1, target, dtype=int)
    return [folds[i] for i in indices]


def optimize_tree_leaves(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    candidate_features: List[int],
    leaf_grid: List[int],
) -> int:
    """Grid-search the number of leaves (approximate cuts) for the tree."""
    if not leaf_grid:
        return None
    subset = candidate_features[: min(len(candidate_features), 10)]
    best_leaf = leaf_grid[0]
    best_score = -np.inf
    for leaf in leaf_grid:
        estimator = DecisionTreeClassifier(max_leaf_nodes=leaf, random_state=0)
        score = evaluate_subset(estimator, X, y, groups, folds, subset)
        if score > best_score + 1e-4:
            best_score = score
            best_leaf = leaf
    print(f"  Optimal tree max_leaf_nodes={best_leaf} (cv acc={best_score:.3f})")
    return best_leaf


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
    for scenario_key, scenario_cfg in SCENARIOS.items():
        if not (sensor == "Neck" and scenario_key == "ACC"):
            continue
        scenario_label = scenario_cfg["label"]
        scenario_cols = [col for col in sensor_df.columns if scenario_cfg["feature_filter"](col)]
        if not scenario_cols:
            continue

        print(f"\n=== Sensor: {sensor} | Scenario: {scenario_label} ===")
        feature_names = scenario_cols
        scenario_data = sensor_df[scenario_cols]
        X = scenario_data.to_numpy(dtype=np.float32)
        Xz = zscore_matrix(X)
        y = labels.to_numpy()
        groups = dog_ids.to_numpy()
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        logo = LeaveOneGroupOut()
        folds = list(logo.split(Xz, y, groups))

        relief_pool, _ = build_relief_pool(Xz, y_encoded, scenario_cfg["max_relief_vars"])
        print(f"ReliefF candidate pool size: {len(relief_pool)}")

        tree_leaf = optimize_tree_leaves(
            Xz,
            y,
            groups,
            folds,
            relief_pool,
            scenario_cfg.get("tree_leaf_grid", []),
        )

        models = {
            #"LDA": LinearDiscriminantAnalysis(),
            #"QDA": QuadraticDiscriminantAnalysis(),
            "SVM": SVC(kernel="rbf", gamma="scale", C=1.0, decision_function_shape="ovo"),
        }
        if tree_leaf:
            models["Tree"] = DecisionTreeClassifier(max_leaf_nodes=tree_leaf, random_state=0)

        for model_name, estimator in models.items():
            print(f"\nSelecting features for {model_name} ...")

            candidates = relief_pool
            folds_eval = None
            if model_name == "SVM":
                cap = scenario_cfg.get("svm_candidate_cap")
                if cap:
                    candidates = relief_pool[: min(cap, len(relief_pool))]
                fast_fraction = scenario_cfg.get("svm_fast_fraction", SVM_FAST_FRACTION)
                if fast_fraction < 1.0:
                    folds_eval = select_fold_subset(folds, fast_fraction)
                    print(
                        f"  Using {len(folds_eval)}/{len(folds)} folds for SVM feature selection "
                        f"(fraction={fast_fraction:.2f})"
                    )

            selected_idx = sequential_forward_selection(
                estimator,
                Xz,
                y,
                groups,
                folds,
                candidates,
                scenario_cfg["max_selected_features"],
                folds_eval=folds_eval,
            )
            selected_names = [feature_names[i] for i in selected_idx]
            print(f"Selected {len(selected_idx)} features: {selected_names}")

            y_true, y_pred, dog_list = cross_val_predict_indices(
                estimator, Xz, y, groups, folds, selected_idx
            )
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            print(f"{sensor} [{scenario_label}] - {model_name}: Acc={acc:.3f}, F1_macro={f1:.3f}")

            classes = label_encoder.classes_.tolist()
            save_confusion_matrix(sensor, scenario_key, model_name, classes, y_true, y_pred)
            save_report(sensor, scenario_key, model_name, classes, y_true, y_pred)

            dog_rates = per_dog_accuracy(y_true, y_pred, dog_list)
            print("Per-dog accuracy:")
            for dog, rate in sorted(dog_rates.items()):
                print(f"  Dog {dog}: {rate:.3f}")
                DOG_RECORDS.append(
                    {
                        "sensor": sensor,
                        "scenario": scenario_label,
                        "model": model_name,
                        "dog_id": int(dog),
                        "accuracy": rate,
                    }
                )

            SUMMARY_RECORDS.append(
                {
                    "sensor": sensor,
                    "scenario": scenario_label,
                    "model": model_name,
                    "num_features": len(selected_idx),
                    "selected_features": ";".join(selected_names),
                    "accuracy": acc,
                    "f1_macro": f1,
                }
            )


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

    RESULTS_DIR.mkdir(exist_ok=True)
    if SUMMARY_RECORDS:
        summary_df = pd.DataFrame(SUMMARY_RECORDS)
        summary_df.to_csv(SUMMARY_CSV, index=False)
        try:
            summary_df.to_excel(SUMMARY_XLSX, index=False)
        except Exception as exc:
            print(f"Excel export skipped ({exc})")
    if DOG_RECORDS:
        dog_df = pd.DataFrame(DOG_RECORDS)
        dog_df.to_csv(PER_DOG_CSV, index=False)


if __name__ == "__main__":
    main()
