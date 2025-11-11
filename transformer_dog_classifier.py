#!/usr/bin/env python3
"""
Transformer-based behaviour classifier that follows the architecture
described in “Automatic behaviour recognition …” (preprint summary):

* Input windows: 200 samples × 6 channels (3-axis accelerometer + 3-axis gyro),
  50 % overlap, per sensor (Back / Neck).
* Encoder-only Transformer with 3 blocks, 6 attention heads, head dim 32
  (d_model = 192), FFN hidden dim 1024, LeakyReLU activations, positional
  encoding, residual connections, layer norm.
* Global Average Pooling → FC → Softmax over the 7 behaviour classes.
* Training via Leave-One-Dog-Out cross-validation, weighted cross-entropy
  (weights ∝ 1 / class frequency). Results are appended to the same summary
  CSV/XLSX files used by the classical models for easy comparison.

Example
-------
    python transformer_dog_classifier.py --data DogMoveData.csv \
        --sensors Back,Neck --scenarios ACC_GYRO,ACC \
        --epochs 200 --model-name Transformer
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

RESULTS_DIR = Path("results_transformer")
DEFAULT_SUMMARY_CSV = RESULTS_DIR / "model_summary.csv"
DEFAULT_SUMMARY_XLSX = RESULTS_DIR / "model_summary.xlsx"
DEFAULT_PER_DOG_CSV = RESULTS_DIR / "per_dog_metrics.csv"

WINDOW_LEN = 200  # samples (2 seconds @ 100 Hz)
HOP = 100  # 50% overlap
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
    label = label.replace(" ", "_")
    label = label.replace("<", "LT").replace(">", "GT")
    return label


def collect_behavior_names(df: pd.DataFrame) -> List[str]:
    behaviors = set()
    for col in BEHAVIOR_COLUMNS:
        if col not in df:
            continue
        behaviors.update(sanitize_behavior(val) for val in df[col].dropna().unique())
    behaviors.discard(None)
    return sorted(behaviors)


def compute_behavior_percentages(segment: pd.DataFrame, behavior_vocab: List[str]) -> Dict[str, float]:
    counts = {beh: 0 for beh in behavior_vocab}
    for col in BEHAVIOR_COLUMNS:
        if col not in segment:
            continue
        vc = segment[col].value_counts()
        for beh, cnt in vc.items():
            if beh in counts:
                counts[beh] += int(cnt)
    return {beh: 100.0 * counts[beh] / float(WINDOW_LEN) for beh in counts}


def build_windows(df: pd.DataFrame, sensor: str, scenario_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        windows: shape (N, WINDOW_LEN, channels)
        labels:  shape (N,)
        dog_ids: shape (N,)
    """
    scenario = SCENARIOS[scenario_key]
    if sensor.lower() == "back":
        acc_cols = ["ABack_x", "ABack_y", "ABack_z"]
        gyro_cols = ["GBack_x", "GBack_y", "GBack_z"]
    else:
        acc_cols = ["ANeck_x", "ANeck_y", "ANeck_z"]
        gyro_cols = ["GNeck_x", "GNeck_y", "GNeck_z"]

    channel_map = {
        "A_x": acc_cols[0],
        "A_y": acc_cols[1],
        "A_z": acc_cols[2],
        "G_x": gyro_cols[0],
        "G_y": gyro_cols[1],
        "G_z": gyro_cols[2],
    }
    selected_cols = [channel_map[ch] for ch in scenario["channels"]]

    behavior_vocab = collect_behavior_names(df)
    behavior_vocab = sorted(set(behavior_vocab).union(TARGET_BEHAVIORS))

    windows = []
    labels = []
    dog_ids = []

    grouped = df.groupby(["DogID", "TestNum"])
    for (dog_id, test_num), group in grouped:
        group = group.sort_values("t_sec").reset_index(drop=True)
        if len(group) < WINDOW_LEN:
            continue
        for start in range(0, len(group) - WINDOW_LEN + 1, HOP):
            end = start + WINDOW_LEN
            segment = group.iloc[start:end]
            behavior_percentages = compute_behavior_percentages(segment, behavior_vocab)

            valid_behaviors = {k: v for k, v in behavior_percentages.items() if v >= 75.0}
            if len(valid_behaviors) != 1:
                continue
            label = list(valid_behaviors.keys())[0]
            if label not in TARGET_BEHAVIORS:
                continue

            data = segment[selected_cols].to_numpy(dtype=np.float32)
            windows.append(data)
            labels.append(label.replace("_", " "))
            dog_ids.append(int(dog_id))

    if not windows:
        return np.empty((0, WINDOW_LEN, len(selected_cols))), np.array([]), np.array([])
    return np.stack(windows), np.array(labels), np.array(dog_ids)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_ff: int, dropout: float, negative_slope: float = 0.01):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x)
        x = self.ln1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_output))
        return x


class DogTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int,
        num_heads: int,
        dim_ff: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=WINDOW_LEN)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dim_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)  # Global Average Pooling
        x = self.dropout(x)
        return self.classifier(x)


class DogTransformerSmall(DogTransformer):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=96,
            num_heads=4,
            dim_ff=512,
            num_layers=2,
            dropout=0.1,
        )


class ConvNet(nn.Module):
    """Lightweight 1D CNN stack with GAP -> FC (serves as fast baseline)."""

    def __init__(self, input_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=5, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, channels) -> transpose to (batch, channels, seq)
        x = x.transpose(1, 2)
        x = self.net(x)
        return self.head(x)


def standardize(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (test - mean) / std


def prepare_datasets(windows: np.ndarray, labels: np.ndarray, dogs: np.ndarray):
    label_encoder = {label: idx for idx, label in enumerate(sorted(np.unique(labels)))}
    y = np.array([label_encoder[label] for label in labels], dtype=np.int64)
    return y, label_encoder


def run_lodo_training(
    windows: np.ndarray,
    labels: np.ndarray,
    dogs: np.ndarray,
    num_classes: int,
    args,
    model_type: str,
) -> Tuple[float, float, Dict[int, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logo = LeaveOneGroupOut()
    all_true, all_pred, all_dog_ids = [], [], []

    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(windows, labels, dogs), start=1):
        X_train, X_test = windows[train_idx], windows[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train, X_test = standardize(X_train, X_test)
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

        if model_type == "TransformerLarge":
            model = DogTransformer(
                input_dim=windows.shape[2],
                num_classes=num_classes,
                d_model=args.d_model,
                num_heads=args.num_heads,
                dim_ff=args.dim_ff,
                num_layers=args.num_layers,
                dropout=args.dropout,
            ).to(device)
        elif model_type == "TransformerSmall":
            model = DogTransformerSmall(
                input_dim=windows.shape[2],
                num_classes=num_classes,
            ).to(device)
        elif model_type == "ConvNet":
            model = ConvNet(input_channels=windows.shape[2], num_classes=num_classes, dropout=args.dropout).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_decay_rate
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        for epoch in range(1, args.epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
            if epoch % args.lr_decay_steps == 0:
                scheduler.step()
            if args.verbose and epoch % args.log_interval == 0:
                print(f"[Fold {fold_idx:02d}] Epoch {epoch:04d}/{args.epochs}")

        model.eval()
        y_true_fold, y_pred_fold = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred_fold.append(preds)
            y_true_fold = y_test
        y_pred_fold = np.concatenate(y_pred_fold)
        dog_id = dogs[test_idx][0]
        acc = accuracy_score(y_true_fold, y_pred_fold)
        f1 = f1_score(y_true_fold, y_pred_fold, average="macro")
        print(f"[Fold {fold_idx:02d}] Dog {dog_id} - Acc={acc:.3f}, F1_macro={f1:.3f}")

        all_true.append(y_true_fold)
        all_pred.append(y_pred_fold)
        all_dog_ids.append(dogs[test_idx])

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    all_dog_ids = np.concatenate(all_dog_ids)
    overall_acc = accuracy_score(all_true, all_pred)
    overall_f1 = f1_score(all_true, all_pred, average="macro")
    per_dog = {int(d): accuracy_score(all_true[all_dog_ids == d], all_pred[all_dog_ids == d]) for d in np.unique(all_dog_ids)}
    print(f"\nOverall: Acc={overall_acc:.3f}, F1_macro={overall_f1:.3f}")
    return overall_acc, overall_f1, per_dog


def parse_list_arg(value: str, default: List[str], transform=lambda x: x) -> List[str]:
    if value is None:
        return default
    parts = [transform(item.strip()) for item in value.split(",") if item.strip()]
    if not parts or (len(parts) == 1 and parts[0].lower() == "all"):
        return default
    return parts


def normalize_sensor_list(value: str) -> List[str]:
    default = ["Back", "Neck"]
    requested = parse_list_arg(value, default, lambda s: s.capitalize())
    return [sensor for sensor in requested if sensor in default]


def normalize_scenario_list(value: str) -> List[str]:
    default = list(SCENARIOS.keys())
    requested = parse_list_arg(
        value,
        default,
        lambda s: s.replace("+", "_").replace(" ", "_").upper(),
    )
    return [scenario for scenario in requested if scenario in SCENARIOS]


def append_results(summary_rows: List[Dict], dog_rows: List[Dict], summary_csv: Path, summary_xlsx: Path, per_dog_csv: Path) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        if summary_csv.exists():
            summary_df = pd.concat([pd.read_csv(summary_csv), summary_df], ignore_index=True)
        summary_df.to_csv(summary_csv, index=False)
        try:
            summary_df.to_excel(summary_xlsx, index=False)
        except Exception as exc:
            print(f"[WARN] Excel export skipped: {exc}")
    if dog_rows:
        dog_df = pd.DataFrame(dog_rows)
        if per_dog_csv.exists():
            dog_df = pd.concat([pd.read_csv(per_dog_csv), dog_df], ignore_index=True)
        dog_df.to_csv(per_dog_csv, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer-based dog behaviour classifier.")
    parser.add_argument("--data", default="DogMoveData.csv", help="Path to raw DogMoveData CSV.")
    parser.add_argument("--sensors", default="Back,Neck", help="Comma-separated sensors (Back,Neck) or 'all'.")
    parser.add_argument("--scenarios", default="ACC_GYRO,ACC", help="Comma-separated scenarios (ACC_GYRO,ACC) or 'all'.")
    parser.add_argument("--model-name", default="Transformer", help="Name stored in the summary CSV.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    parser.add_argument("--lr-decay-rate", type=float, default=0.95, dest="lr_decay_rate")
    parser.add_argument("--lr-decay-steps", type=int, default=200, dest="lr_decay_steps")
    parser.add_argument("--d-model", type=int, default=192, dest="d_model")
    parser.add_argument("--num-heads", type=int, default=6, dest="num_heads")
    parser.add_argument("--dim-ff", type=int, default=1024, dest="dim_ff")
    parser.add_argument("--num-layers", type=int, default=3, dest="num_layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--models", default="TransformerLarge,TransformerSmall,ConvNet", help="Comma-separated model types to run.")
    parser.add_argument("--summary-dir", default="results_transformer", help="Directory to store summary/per-dog CSVs.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--log-interval", type=int, default=50, dest="log_interval")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.data)
    for col in BEHAVIOR_COLUMNS:
        if col in df:
            df[col] = df[col].apply(sanitize_behavior)

    sensors = normalize_sensor_list(args.sensors)
    scenarios = normalize_scenario_list(args.scenarios)

    summary_rows: List[Dict] = []
    dog_rows: List[Dict] = []

    for sensor in sensors:
        for scenario_key in scenarios:
            windows, labels, dog_ids = build_windows(df, sensor, scenario_key)
            if len(windows) == 0:
                print(f"[WARN] No windows for sensor={sensor} scenario={SCENARIOS[scenario_key]['label']}")
                continue

            label_encoder = {label: idx for idx, label in enumerate(sorted(np.unique(labels)))}
            y = np.array([label_encoder[label] for label in labels], dtype=np.int64)
            print(
                f"\nSensor={sensor} Scenario={SCENARIOS[scenario_key]['label']} "
                f"Samples={len(windows)} Classes={len(label_encoder)}"
            )

            model_list = parse_list_arg(args.models, ["TransformerLarge"])
            for model_type in model_list:
                print(f"\n>>> Training model={model_type} on sensor={sensor} scenario={SCENARIOS[scenario_key]['label']}")
                acc, f1, per_dog = run_lodo_training(
                    windows,
                    y,
                    dog_ids,
                    num_classes=len(label_encoder),
                    args=args,
                    model_type=model_type,
                )

                summary_rows.append(
                    {
                        "sensor": sensor,
                        "scenario": SCENARIOS[scenario_key]["label"],
                        "model": model_type,
                        "num_features": windows.shape[2] * WINDOW_LEN,
                        "selected_features": f"raw_{SCENARIOS[scenario_key]['label']}",
                        "accuracy": acc,
                        "f1_macro": f1,
                    }
                )
                for dog, dog_acc in per_dog.items():
                    dog_rows.append(
                        {
                            "sensor": sensor,
                            "scenario": SCENARIOS[scenario_key]["label"],
                            "model": model_type,
                            "dog_id": dog,
                            "accuracy": dog_acc,
                        }
                    )

    summary_dir = Path(args.summary_dir)
    summary_csv = summary_dir / "model_summary.csv"
    summary_xlsx = summary_dir / "model_summary.xlsx"
    per_dog_csv = summary_dir / "per_dog_metrics.csv"
    append_results(summary_rows, dog_rows, summary_csv, summary_xlsx, per_dog_csv)


if __name__ == "__main__":
    main()
