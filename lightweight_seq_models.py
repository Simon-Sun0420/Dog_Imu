#!/usr/bin/env python3
"""
Lightweight sequence models (TCN / BiLSTM / ResNet1D) trained on the raw
DogMoveData.csv windows (200×6, 50 % overlap). The script mirrors the
evaluation flow used by other models: Leave-One-Dog-Out cross-validation,
class-weighted loss, confusion matrices, classification reports, and
per-dog accuracy logs. Each run trains ONE model (selected via --model),
so you can queue multiple runs overnight.

Outputs are written to results_raw/<run_name>/model_summary.csv,
results_raw/<run_name>/per_dog_metrics.csv, plus confusion/report files
for every sensor/scenario/model combination.
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

WINDOW_LEN = 200
HOP = 100
BEHAVIOR_COLUMNS = ["Behavior_1", "Behavior_2", "Behavior_3"]
TARGET_BEHAVIORS = [
    "Walking",
    "Standing",
    "Lying_chest",
    "Trotting",
    "Sitting",
    "Galloping",
    "Sniffing",
]

SCENARIOS = {
    "ACC_GYRO": {
        "label": "A+G",
        "channels": ["A_x", "A_y", "A_z", "G_x", "G_y", "G_z"],
    },
    "ACC": {
        "label": "A only",
        "channels": ["A_x", "A_y", "A_z"],
    },
}


def sanitize_behavior(label: Optional[str]) -> Optional[str]:
    if label is None or pd.isna(label):
        return None
    label = label.strip()
    if not label or label == "<undefined>":
        return None
    return label.replace(" ", "_").replace("<", "LT").replace(">", "GT")


def collect_behavior_names(df: pd.DataFrame) -> List[str]:
    names = set()
    for col in BEHAVIOR_COLUMNS:
        if col in df:
            names.update(sanitize_behavior(v) for v in df[col].dropna().unique())
    names.discard(None)
    return sorted(names)


def compute_behavior_percentages(segment: pd.DataFrame, vocab: List[str]) -> Dict[str, float]:
    counts = {beh: 0 for beh in vocab}
    for col in BEHAVIOR_COLUMNS:
        if col not in segment:
            continue
        vc = segment[col].value_counts()
        for beh, cnt in vc.items():
            if beh in counts:
                counts[beh] += int(cnt)
    return {beh: 100.0 * counts[beh] / WINDOW_LEN for beh in counts}


def build_windows(df: pd.DataFrame, sensor: str, scenario_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scenario = SCENARIOS[scenario_key]
    if sensor.lower() == "back":
        acc = ["ABack_x", "ABack_y", "ABack_z"]
        gyro = ["GBack_x", "GBack_y", "GBack_z"]
    else:
        acc = ["ANeck_x", "ANeck_y", "ANeck_z"]
        gyro = ["GNeck_x", "GNeck_y", "GNeck_z"]
    mapping = {
        "A_x": acc[0],
        "A_y": acc[1],
        "A_z": acc[2],
        "G_x": gyro[0],
        "G_y": gyro[1],
        "G_z": gyro[2],
    }
    cols = [mapping[ch] for ch in scenario["channels"]]

    vocab = sorted(set(collect_behavior_names(df)).union(TARGET_BEHAVIORS))
    windows, labels, dogs = [], [], []
    for (dog, test), group in df.groupby(["DogID", "TestNum"]):
        group = group.sort_values("t_sec")
        if len(group) < WINDOW_LEN:
            continue
        for start in range(0, len(group) - WINDOW_LEN + 1, HOP):
            segment = group.iloc[start : start + WINDOW_LEN]
            percentages = compute_behavior_percentages(segment, vocab)
            valid = [k for k, v in percentages.items() if v >= 75.0 and k in TARGET_BEHAVIORS]
            if len(valid) != 1:
                continue
            windows.append(segment[cols].to_numpy(dtype=np.float32))
            labels.append(valid[0].replace("_", " "))
            dogs.append(int(dog))
    if not windows:
        return np.empty((0, WINDOW_LEN, len(cols))), np.array([]), np.array([])
    return np.stack(windows), np.array(labels), np.array(dogs)


def standardize(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (test - mean) / std


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")
        padding = dilation * ((kernel_size - 1) // 2)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        out = self.net(x)
        out = out + self.downsample(x)
        return nn.functional.leaky_relu(out, 0.01)


class TCNModel(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        channels = [input_channels, 64, 128, 128]
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(
                TCNBlock(channels[i], channels[i + 1], kernel_size=3, dilation=2 ** i, dropout=dropout)
            )
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        return self.head(x)


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.head(out)


class ResNet1DBlock(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.01),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.dropout(out)
        return nn.functional.leaky_relu(out + x, 0.01)


class ResNet1DModel(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, base_channels: int = 64, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.01),
        )
        self.blocks = nn.Sequential(
            ResNet1DBlock(base_channels, dropout),
            ResNet1DBlock(base_channels, dropout),
            ResNet1DBlock(base_channels, dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_channels, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.blocks(self.stem(x))
        return self.head(x)


MODEL_FACTORY = {
    "TCN": lambda inp, cls, args: TCNModel(inp, cls, dropout=args.dropout),
    "BiLSTM": lambda inp, cls, args: BiLSTMModel(inp, cls, hidden=args.hidden_dim, num_layers=2, dropout=args.dropout),
    "ResNet1D": lambda inp, cls, args: ResNet1DModel(inp, cls, base_channels=args.hidden_dim, dropout=args.dropout),
}


def save_confusion(sensor, scenario_key, model_name, classes, y_true, y_pred, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"{sensor} ({scenario_key}) - {model_name} Confusion")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / f"{sensor.lower()}_{scenario_key.lower()}_{model_name.lower()}_confusion.png")
    plt.close()


def save_report(sensor, scenario_key, model_name, classes, y_true, y_pred, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, labels=classes)
    (out_dir / f"{sensor.lower()}_{scenario_key.lower()}_{model_name.lower()}_report.txt").write_text(report)


def run_lodo_model(
    model_name: str,
    windows: np.ndarray,
    labels: np.ndarray,
    dogs: np.ndarray,
    num_classes: int,
    args,
):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logo = LeaveOneGroupOut()

    class_counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / np.maximum(class_counts, 1)
    weights = weights / weights.sum() * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    all_true, all_pred, all_dogs = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(windows, labels, dogs), start=1):
        X_train, X_test = standardize(windows[train_idx], windows[test_idx])
        y_train, y_test = labels[train_idx], labels[test_idx]

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train)),
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test)),
            batch_size=args.batch_size,
            shuffle=False,
        )

        model = MODEL_FACTORY[model_name](windows.shape[2], num_classes, args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        progress = tqdm(range(1, args.epochs + 1), desc=f"{model_name} Fold {fold_idx}", leave=False)
        for epoch in progress:
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            if epoch % args.lr_decay_steps == 0:
                scheduler.step()
            progress.set_postfix(loss=epoch_loss / len(train_loader.dataset))

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                preds.append(torch.argmax(model(xb), dim=1).cpu().numpy())
        preds = np.concatenate(preds)
        fold_acc = accuracy_score(y_test, preds)
        fold_f1 = f1_score(y_test, preds, average="macro")
        print(f"[{model_name}] Fold {fold_idx} Dog {dogs[test_idx][0]} - Acc={fold_acc:.3f} F1={fold_f1:.3f}")

        all_true.append(y_test)
        all_pred.append(preds)
        all_dogs.append(dogs[test_idx])

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    all_dogs = np.concatenate(all_dogs)
    overall_acc = accuracy_score(all_true, all_pred)
    overall_f1 = f1_score(all_true, all_pred, average="macro")
    per_dog = {int(d): accuracy_score(all_true[all_dogs == d], all_pred[all_dogs == d]) for d in np.unique(all_dogs)}
    return overall_acc, overall_f1, per_dog, all_true, all_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight sequence models on raw DogMoveData.")
    parser.add_argument("--data", default="DogMoveData.csv", help="Path to DogMoveData.csv")
    parser.add_argument("--sensors", default="Back,Neck", help="Comma-separated list or 'all'")
    parser.add_argument("--scenarios", default="ACC_GYRO,ACC", help="Comma-separated list or 'all'")
    parser.add_argument("--model", choices=list(MODEL_FACTORY.keys()), default="TCN", help="Model to train.")
    parser.add_argument("--run-name", default="light_models", help="Subfolder under results_raw/")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-decay", type=float, default=0.98, help="Exponential LR decay rate.")
    parser.add_argument("--lr-decay-steps", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128, dest="hidden_dim")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def parse_list(value: str, default: List[str]):
    if not value:
        return default
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts or parts == ["all"]:
        return default
    return parts


def main():
    args = parse_args()
    df = pd.read_csv(args.data)
    for col in BEHAVIOR_COLUMNS:
        if col in df:
            df[col] = df[col].apply(sanitize_behavior)

    sensors = parse_list(args.sensors, ["Back", "Neck"])
    scenario_keys_requested = parse_list(args.scenarios, list(SCENARIOS.keys()))
    scenarios = [key for key in SCENARIOS if key in scenario_keys_requested]

    base_dir = Path("results_raw") / args.run_name
    summary_csv = base_dir / "model_summary.csv"
    per_dog_csv = base_dir / "per_dog_metrics.csv"

    summary_rows = []
    dog_rows = []

    for sensor in sensors:
        for scenario_key in scenarios:
            windows, labels, dogs = build_windows(df, sensor, scenario_key)
            if len(windows) == 0:
                print(f"[WARN] No samples for {sensor} {SCENARIOS[scenario_key]['label']}")
                continue
            label_map = {lbl: idx for idx, lbl in enumerate(sorted(np.unique(labels)))}
            y = np.array([label_map[lbl] for lbl in labels], dtype=np.int64)

            print(
                f"\n>>> Sensor={sensor} Scenario={SCENARIOS[scenario_key]['label']} "
                f"Samples={len(windows)} Model={args.model}"
            )
            acc, f1, per_dog, true_all, pred_all = run_lodo_model(
                args.model, windows, y, dogs, len(label_map), args
            )

            classes = list(label_map.keys())
            save_confusion(sensor, scenario_key, args.model, classes, true_all, pred_all, base_dir)
            save_report(sensor, scenario_key, args.model, classes, true_all, pred_all, base_dir)

            scenario_label = SCENARIOS[scenario_key]["label"]
            summary_rows.append(
                {
                    "sensor": sensor,
                    "scenario": scenario_label,
                    "model": args.model,
                    "num_features": windows.shape[2] * WINDOW_LEN,
                    "selected_features": f"raw_{scenario_label}",
                    "accuracy": acc,
                    "f1_macro": f1,
                }
            )
            for dog, dog_acc in per_dog.items():
                dog_rows.append(
                    {
                        "sensor": sensor,
                        "scenario": scenario_label,
                        "model": args.model,
                        "dog_id": dog,
                        "accuracy": dog_acc,
                    }
                )

    if summary_rows:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame(summary_rows)
        if summary_csv.exists():
            summary_df = pd.concat([pd.read_csv(summary_csv), summary_df], ignore_index=True)
        summary_df.to_csv(summary_csv, index=False)
    if dog_rows:
        per_dog_csv.parent.mkdir(parents=True, exist_ok=True)
        dogs_df = pd.DataFrame(dog_rows)
        if per_dog_csv.exists():
            dogs_df = pd.concat([pd.read_csv(per_dog_csv), dogs_df], ignore_index=True)
        dogs_df.to_csv(per_dog_csv, index=False)

    print("\nRun complete. Results stored under", base_dir)


if __name__ == "__main__":
    main()
