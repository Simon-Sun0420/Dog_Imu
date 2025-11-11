#!/usr/bin/env python3
"""
Neural-network baseline for the dog behaviour dataset.

The script mirrors the classical pipeline by:
* loading processed_dog_features.pkl (Back & Neck sensors)
* evaluating both feature scenarios (ACC+GYRO and ACC-only) via Leave-One-Dog-Out
* logging the overall accuracy/F1 plus per-dog accuracies into the shared
  results/model_summary.csv and results/per_dog_metrics.csv files so the NN can
  be compared directly with the SVM/LDA/QDA/Tree runs.

Example usage:
    python nn_dog_classifier.py --data processed_dog_features.pkl
    python nn_dog_classifier.py --data processed_dog_features.pkl --sensor Back
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

RESULTS_DIR = Path("results")
SUMMARY_CSV = RESULTS_DIR / "model_summary.csv"
SUMMARY_XLSX = RESULTS_DIR / "model_summary.xlsx"
PER_DOG_CSV = RESULTS_DIR / "per_dog_metrics.csv"

SCENARIOS = {
    "ACC_GYRO": {
        "label": "A+G",
        "feature_filter": lambda col: col.startswith("A") or col.startswith("G"),
    },
    "ACC": {
        "label": "A only",
        "feature_filter": lambda col: col.startswith("A"),
    },
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_from_processed(path: Path, sensor: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    sensor = sensor.capitalize()
    if sensor not in payload["sensors"]:
        raise ValueError(f"Sensor '{sensor}' not found in {path}. Available: {list(payload['sensors'].keys())}")
    sensor_frame = payload["sensors"][sensor]
    if isinstance(sensor_frame, pd.DataFrame):
        X = sensor_frame.to_numpy(dtype=np.float32)
    else:
        X = np.asarray(sensor_frame, dtype=np.float32)
    y = np.asarray(payload["labels"])
    groups = np.asarray(payload["dog_ids"])
    return X, y, groups


def load_from_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "dog_id" not in df or "label" not in df:
        raise ValueError("CSV file must include 'dog_id' and 'label' columns.")
    feature_cols = [c for c in df.columns if c not in {"dog_id", "test_num", "window_start", "window_end", "label"}]
    if not feature_cols:
        raise ValueError("No feature columns found in CSV.")
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    groups = df["dog_id"].to_numpy()
    return X, y, groups


def load_dataset(path: str, sensor: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{path} not found.")
    if path_obj.suffix == ".pkl":
        return load_from_processed(path_obj, sensor)
    return load_from_csv(path_obj)


# ---------------------------------------------------------------------------
# Torch dataset/model utilities
# ---------------------------------------------------------------------------
class DogDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class DogNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(preds)


# ---------------------------------------------------------------------------
# Main training loop (LODO evaluation)
# ---------------------------------------------------------------------------
def run_lodo_training(X: np.ndarray, y: np.ndarray, groups: np.ndarray, args) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    logo = LeaveOneGroupOut()

    all_true, all_pred = [], []
    all_dogs = []
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y_enc, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_dataset = DogDataset(X_train, y_train)
        test_dataset = DogDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        model = DogNet(
            input_dim=X.shape[1],
            hidden_dim=args.hidden_dim,
            num_classes=len(label_encoder.classes_),
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            if args.verbose and (epoch % max(1, args.log_interval) == 0):
                print(f"[Fold {fold:02d}] Epoch {epoch:03d} - loss={loss:.4f}")

        true_fold, pred_fold = evaluate(model, test_loader, device)
        acc = accuracy_score(true_fold, pred_fold)
        f1 = f1_score(true_fold, pred_fold, average="macro")
        test_dog = np.unique(groups[test_idx])[0]
        print(f"[Fold {fold:02d}] Dog {test_dog} - Acc={acc:.3f}, F1_macro={f1:.3f}")
        all_true.append(true_fold)
        all_pred.append(pred_fold)
        all_dogs.append(groups[test_idx])

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    all_dogs = np.concatenate(all_dogs)
    overall_acc = accuracy_score(all_true, all_pred)
    overall_f1 = f1_score(all_true, all_pred, average="macro")
    per_dog = {}
    for dog in np.unique(all_dogs):
        mask = all_dogs == dog
        per_dog[int(dog)] = accuracy_score(all_true[mask], all_pred[mask])
    print(f"\nOverall: Acc={overall_acc:.3f}, F1_macro={overall_f1:.3f}")
    return {
        "accuracy": overall_acc,
        "f1_macro": overall_f1,
        "per_dog": per_dog,
    }


def parse_list_arg(value: str, default: List[str], transform=lambda x: x) -> List[str]:
    if value is None:
        return default
    parts = [transform(item.strip()) for item in value.split(",") if item.strip()]
    if not parts:
        return default
    if len(parts) == 1 and parts[0].lower() == "all":
        return default
    return parts


def normalize_sensor_list(value: str, available: List[str]) -> List[str]:
    default = available
    requested = parse_list_arg(value, default, lambda s: s.capitalize())
    return [sensor for sensor in requested if sensor in available]


def normalize_scenario_list(value: str) -> List[str]:
    default = list(SCENARIOS.keys())
    requested = parse_list_arg(
        value,
        default,
        lambda s: s.replace("+", "_").replace(" ", "_").upper(),
    )
    valid = []
    for key in requested:
        if key in SCENARIOS:
            valid.append(key)
    return valid if valid else default


def append_results(summary_rows: List[Dict], dog_rows: List[Dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        if SUMMARY_CSV.exists():
            existing = pd.read_csv(SUMMARY_CSV)
            summary_df = pd.concat([existing, summary_df], ignore_index=True)
        summary_df.to_csv(SUMMARY_CSV, index=False)
        try:
            summary_df.to_excel(SUMMARY_XLSX, index=False)
        except Exception as exc:
            print(f"[WARN] Excel export skipped: {exc}")
    if dog_rows:
        dog_df = pd.DataFrame(dog_rows)
        if PER_DOG_CSV.exists():
            existing = pd.read_csv(PER_DOG_CSV)
            dog_df = pd.concat([existing, dog_df], ignore_index=True)
        dog_df.to_csv(PER_DOG_CSV, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Neural network baseline for dog behaviour recognition.")
    parser.add_argument("--data", default="processed_dog_features.pkl", help="Path to processed .pkl or raw CSV file.")
    parser.add_argument("--sensor", default="Back,Neck", help="Comma-separated sensors (Back,Neck) or 'all'.")
    parser.add_argument("--scenarios", default="ACC_GYRO,ACC", help="Comma-separated scenarios (ACC_GYRO,ACC) or 'all'.")
    parser.add_argument("--model-name", default="NN", help="Name written to the summary CSV.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per fold.")
    parser.add_argument("--hidden-dim", type=int, default=128, dest="hidden_dim", help="Hidden layer size.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--batch-size", type=int, default=128, dest="batch_size", help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay", help="Adam weight decay.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--log-interval", type=int, default=10, dest="log_interval", help="Epoch interval for logging.")
    parser.add_argument("--verbose", action="store_true", help="Print training loss during epochs.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    summary_rows = []
    dog_rows = []

    if data_path.suffix == ".pkl":
        with open(data_path, "rb") as fh:
            payload = pickle.load(fh)
        sensors_available = list(payload["sensors"].keys())
        sensors_to_run = normalize_sensor_list(args.sensor, sensors_available)
        scenario_keys = normalize_scenario_list(args.scenarios)
        labels = np.asarray(payload["labels"])
        groups = np.asarray(payload["dog_ids"])

        for sensor in sensors_to_run:
            sensor_df = payload["sensors"][sensor]
            if not isinstance(sensor_df, pd.DataFrame):
                sensor_df = pd.DataFrame(sensor_df, columns=payload["feature_columns"])
            for scenario_key in scenario_keys:
                scenario_cfg = SCENARIOS[scenario_key]
                scenario_cols = [col for col in sensor_df.columns if scenario_cfg["feature_filter"](col)]
                if not scenario_cols:
                    continue
                X = sensor_df[scenario_cols].to_numpy(dtype=np.float32)
                print(f"\nSensor={sensor} Scenario={scenario_cfg['label']} Features={len(scenario_cols)} Samples={X.shape[0]}")
                metrics = run_lodo_training(X, labels, groups, args)

                summary_rows.append(
                    {
                        "sensor": sensor,
                        "scenario": scenario_cfg["label"],
                        "model": args.model_name,
                        "num_features": len(scenario_cols),
                        "selected_features": ";".join(scenario_cols),
                        "accuracy": metrics["accuracy"],
                        "f1_macro": metrics["f1_macro"],
                    }
                )
                for dog, acc in metrics["per_dog"].items():
                    dog_rows.append(
                        {
                            "sensor": sensor,
                            "scenario": scenario_cfg["label"],
                            "model": args.model_name,
                            "dog_id": dog,
                            "accuracy": acc,
                        }
                    )
    else:
        X, y, groups = load_dataset(args.data, None)
        print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features from {args.data}")
        metrics = run_lodo_training(X, y, groups, args)
        summary_rows.append(
            {
                "sensor": Path(args.data).stem,
                "scenario": "CSV",
                "model": args.model_name,
                "num_features": X.shape[1],
                "selected_features": ";".join([f"f{i}" for i in range(X.shape[1])]),
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
            }
        )
        for dog, acc in metrics["per_dog"].items():
            dog_rows.append(
                {
                    "sensor": Path(args.data).stem,
                    "scenario": "CSV",
                    "model": args.model_name,
                    "dog_id": dog,
                    "accuracy": acc,
                }
            )

    append_results(summary_rows, dog_rows)


if __name__ == "__main__":
    main()
