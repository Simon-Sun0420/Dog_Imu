#!/usr/bin/env python3
"""
Port of the Matlab feature/label preparation pipeline described in
“Automatic behaviour recognition in accelerometer data from dogs” (Applied Animal Behaviour Science, 2021).

The script mirrors CalcFeatures_Acc_Gyro.m and CollectFeatures.m:
1. 2 s windows (200 samples) with 50 % overlap are formed per DogID/TestNum.
2. Accelerometer signals are first referenced to the trim-mean posture recorded
   while the dog is standing during the “Task stand” segments.
3. For every window, the same engineered features (total activity, axis means,
   offsets, ECDF samples, and run-length peak counts for both accelerometer
   and gyroscope) are computed separately for the back and neck sensors.
4. Behaviour coverage percentages per window are calculated and the dataset is
   filtered to only keep windows where exactly one of the target behaviours
   occupies ≥ 75 % of the samples (as done in CollectFeatures.m).
5. The final per-sensor feature matrices (Xd in the Matlab code) together with
   labels, DogID, and TestNum are stored in processed_dog_features.pkl.
"""

import warnings

warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration (mirrors MasterScript.m / CalcFeatures_Acc_Gyro.m constants)
# ---------------------------------------------------------------------------
FS = 100  # Hz
WINDOW_LEN = 200  # samples (2 seconds)
OVERLAP_SAMPLES = WINDOW_LEN // 2  # 50 % overlap → hop = 1 s
NECDF = 7  # interpolation points for ECDFs
PEAK_THRESHOLD = 0.2
PEAK_MIN_RUN = 5
WINDOW_SEC = WINDOW_LEN / FS
TARGET_BEHAVIORS = [
    "Walking",
    "Standing",
    "Lying_chest",
    "Trotting",
    "Sitting",
    "Galloping",
    "Sniffing",
]
BEHAVIOR_COLUMNS = ["Behavior_1", "Behavior_2", "Behavior_3"]
OUTPUT_FILE = "processed_dog_features.pkl"


# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SensorWindowFeatures:
    dog_id: int
    test_num: int
    window_id: str
    task: Optional[str]
    behaviors: Dict[str, float] = field(default_factory=dict)
    values: Dict[str, float] = field(default_factory=dict)

    def to_record(self, feature_order: List[str], behavior_order: List[str]) -> Dict[str, float]:
        """Convert to a flat record suitable for DataFrame construction."""
        record = {
            "DogID": self.dog_id,
            "TestNum": self.test_num,
            "Task": self.task if self.task is not None else "<undefined>",
            "window_id": self.window_id,
        }
        for beh in behavior_order:
            record[beh] = self.behaviors.get(beh, 0.0)
        for name in feature_order:
            record[name] = self.values.get(name, 0.0)
        return record


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def sanitize_behavior(label: Optional[str]) -> Optional[str]:
    """Match Matlab’s ValidCats naming (spaces→underscore, < → LT)."""
    if label is None or pd.isna(label):
        return None
    label = label.strip()
    if not label or label == "<undefined>":
        return None
    label = label.replace(" ", "_")
    label = label.replace("<", "LT").replace(">", "GT")
    return label


def collect_behavior_names(df: pd.DataFrame) -> List[str]:
    """Return sorted list of all sanitized behaviour labels."""
    behaviors = set()
    for col in BEHAVIOR_COLUMNS:
        if col not in df:
            continue
        behaviors.update(sanitize_behavior(val) for val in df[col].dropna().unique())
    behaviors.discard(None)
    return sorted(behaviors)


def trimmed_mean_offset(values: np.ndarray) -> np.ndarray:
    """Trimmed mean with 5 % from each tail (Matlab trimmean(...,10))."""
    if values.size == 0:
        return np.zeros(values.shape[1], dtype=np.float32)
    proportion = 0.05
    n = values.shape[0]
    trim = int(np.floor(proportion * n))
    if trim == 0:
        return values.mean(axis=0)
    trimmed = np.sort(values, axis=0)[trim : n - trim]
    return trimmed.mean(axis=0)


def count_peak_rate(signal_axis: np.ndarray, mean_val: float) -> float:
    """Count run-lengths above mean+threshold and normalise per second."""
    threshold = mean_val + PEAK_THRESHOLD
    mask = signal_axis > threshold
    run_lengths = []
    run = 0
    for flag in mask:
        if flag:
            run += 1
        elif run:
            run_lengths.append(run)
            run = 0
    if run:
        run_lengths.append(run)
    peaks = sum(1 for length in run_lengths if length >= PEAK_MIN_RUN)
    return peaks / WINDOW_SEC


def ecdf_quantiles(signal_axis: np.ndarray) -> List[float]:
    """Sample ECDF via evenly spaced quantiles (0→1)."""
    if signal_axis.size == 0:
        return [0.0] * NECDF
    probs = np.linspace(0.0, 1.0, NECDF)
    return list(np.quantile(signal_axis, probs, method="linear"))


def compute_modal_features(signal: np.ndarray, prefix: str) -> Dict[str, float]:
    """Compute engineered features for either accelerometer or gyro."""
    means = signal.mean(axis=0)
    total_activity = float(np.sum(np.std(signal, axis=0, ddof=0)))
    offset = float(np.linalg.norm(means))
    peak_rates = [count_peak_rate(signal[:, i], means[i]) for i in range(3)]
    ecdf_values = []
    axes = ["X", "Y", "Z"]
    for idx, axis in enumerate(axes):
        ecdfs = ecdf_quantiles(signal[:, idx])
        for q_idx, value in enumerate(ecdfs, start=1):
            ecdf_values.append((f"{prefix}{axis}{q_idx}", float(value)))
    feature_dict = {
        f"{prefix}TotAct": total_activity,
        f"{prefix}MeanX": float(means[0]),
        f"{prefix}MeanY": float(means[1]),
        f"{prefix}MeanZ": float(means[2]),
        f"{prefix}Offset": offset,
        f"{prefix}NMeanCros": float(np.mean(peak_rates)),
    }
    feature_dict.update(dict(ecdf_values))
    return feature_dict


def build_feature_template(modality_prefix: str) -> List[str]:
    """Deterministic order for feature columns (matches CollectFeatures)."""
    names = [
        f"{modality_prefix}TotAct",
        f"{modality_prefix}MeanX",
        f"{modality_prefix}MeanY",
        f"{modality_prefix}MeanZ",
        f"{modality_prefix}Offset",
        f"{modality_prefix}NMeanCros",
    ]
    for axis in "XYZ":
        for idx in range(1, NECDF + 1):
            names.append(f"{modality_prefix}{axis}{idx}")
    return names


BACK_FEATURE_ORDER = (
    build_feature_template("A")
    + build_feature_template("G")
)


def compute_behavior_percentages(segment: pd.DataFrame, behavior_vocab: List[str]) -> Dict[str, float]:
    """Replicate CollectFeatures.m behaviour aggregation."""
    counts = {beh: 0 for beh in behavior_vocab}
    for col in BEHAVIOR_COLUMNS:
        if col not in segment:
            continue
        vc = segment[col].value_counts()
        for beh, cnt in vc.items():
            if beh in counts:
                counts[beh] += int(cnt)
    return {beh: 100.0 * counts[beh] / float(WINDOW_LEN) for beh in counts}


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------
def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv("DogMoveData.csv")
    for col in BEHAVIOR_COLUMNS:
        if col in df:
            df[col] = df[col].apply(sanitize_behavior)
    return df


def extract_window_features(
    df: pd.DataFrame, behavior_vocab: List[str]
) -> Dict[str, pd.DataFrame]:
    sensor_records: Dict[str, List[SensorWindowFeatures]] = {"Back": [], "Neck": []}
    groups = df.groupby(["DogID", "TestNum"], sort=True)
    window_counter = 0

    for (dog_id, test_num), group in tqdm(groups, desc="Processing groups"):
        group = group.sort_values("t_sec").reset_index(drop=True)
        if len(group) < WINDOW_LEN:
            continue

        # Baseline removal (trimmed mean over standing segments)
        stand_mask = (group["Task"] == "Task stand") & group.apply(
            lambda row: any(
                sanitize_behavior(row.get(col)) == "Standing" for col in BEHAVIOR_COLUMNS
            ),
            axis=1,
        )
        if stand_mask.any():
            stand_values = group.loc[stand_mask, ["ABack_x", "ABack_y", "ABack_z"]].to_numpy()
            offsets = trimmed_mean_offset(stand_values)
            group[["ABack_x", "ABack_y", "ABack_z"]] = (
                group[["ABack_x", "ABack_y", "ABack_z"]].to_numpy() - offsets
            )
            stand_values_neck = group.loc[stand_mask, ["ANeck_x", "ANeck_y", "ANeck_z"]].to_numpy()
            neck_offsets = trimmed_mean_offset(stand_values_neck)
            group[["ANeck_x", "ANeck_y", "ANeck_z"]] = (
                group[["ANeck_x", "ANeck_y", "ANeck_z"]].to_numpy() - neck_offsets
            )

        starts = range(0, len(group) - WINDOW_LEN + 1, OVERLAP_SAMPLES)
        for start in starts:
            end = start + WINDOW_LEN
            window_counter += 1
            window_id = f"{dog_id}_{test_num}_{start}"
            segment = group.iloc[start:end]

            task_series = segment["Task"].dropna()
            task_label = task_series.iloc[0] if not task_series.empty else None

            behavior_counts: Dict[str, float] = compute_behavior_percentages(segment, behavior_vocab)

            a_back = segment[["ABack_x", "ABack_y", "ABack_z"]].to_numpy(dtype=np.float32)
            g_back = segment[["GBack_x", "GBack_y", "GBack_z"]].to_numpy(dtype=np.float32)
            a_neck = segment[["ANeck_x", "ANeck_y", "ANeck_z"]].to_numpy(dtype=np.float32)
            g_neck = segment[["GNeck_x", "GNeck_y", "GNeck_z"]].to_numpy(dtype=np.float32)

            back_features = {
                **compute_modal_features(a_back, "A"),
                **compute_modal_features(g_back, "G"),
            }
            neck_features = {
                **compute_modal_features(a_neck, "A"),
                **compute_modal_features(g_neck, "G"),
            }

            sensor_records["Back"].append(
                SensorWindowFeatures(
                    dog_id=dog_id,
                    test_num=test_num,
                    window_id=window_id,
                    task=task_label,
                    behaviors=behavior_counts,
                    values=back_features,
                )
            )
            sensor_records["Neck"].append(
                SensorWindowFeatures(
                    dog_id=dog_id,
                    test_num=test_num,
                    window_id=window_id,
                    task=task_label,
                    behaviors=behavior_counts,
                    values=neck_features,
                )
            )

    feature_order = BACK_FEATURE_ORDER
    behavior_order = behavior_vocab
    frames = {}
    for sensor, records in sensor_records.items():
        data = [rec.to_record(feature_order, behavior_order) for rec in records]
        frames[sensor] = pd.DataFrame(data)
    return frames


def assign_labels(back_df: pd.DataFrame, behavior_vocab: List[str]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Index]:
    """Replicate CollectFeatures.m label selection using Back sensor behaviours."""
    available_behaviors = [beh for beh in TARGET_BEHAVIORS if beh in behavior_vocab]
    if not available_behaviors:
        raise ValueError("No target behaviours present in the dataset.")

    class_data = back_df[available_behaviors].fillna(0.0)
    coverage_mask = class_data >= 75.0
    num_matches = coverage_mask.sum(axis=1)
    primary = class_data.idxmax(axis=1)
    labels = primary.where(num_matches == 1, other=None)
    valid_mask = labels.notna()
    friendly_labels = labels[valid_mask].str.replace("_", " ")
    dog_ids = back_df.loc[valid_mask, "DogID"].astype(int)
    test_nums = back_df.loc[valid_mask, "TestNum"].astype(int)
    valid_windows = back_df.loc[valid_mask, "window_id"]
    return friendly_labels.reset_index(drop=True), dog_ids.reset_index(drop=True), test_nums.reset_index(drop=True), valid_windows


def save_processed_features(frames: Dict[str, pd.DataFrame], labels: pd.Series, dog_ids: pd.Series, test_nums: pd.Series, window_ids: pd.Index) -> None:
    """Persist the processed dataset as a pickle file."""
    feature_cols = BACK_FEATURE_ORDER
    behaviour_cols = sorted({col for col in frames["Back"].columns if col not in {"DogID", "TestNum", "Task", "window_id", *feature_cols}})

    payload = {
        "feature_columns": feature_cols,
        "behaviour_columns": behaviour_cols,
        "labels": labels,
        "dog_ids": dog_ids,
        "test_nums": test_nums,
        "sensors": {},
    }
    window_id_list = list(window_ids)
    for sensor_name, df in frames.items():
        df_indexed = df.set_index("window_id")
        sensor_matrix = df_indexed.loc[window_id_list, feature_cols].reset_index(drop=True)
        payload["sensors"][sensor_name] = sensor_matrix

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved processed dataset to {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("Loading DogMoveData.csv ...")
    df = load_and_prepare()
    behavior_vocab = collect_behavior_names(df)
    behavior_vocab = sorted(set(behavior_vocab).union(TARGET_BEHAVIORS))
    if not behavior_vocab:
        raise RuntimeError("No behaviour annotations found in the dataset.")

    print("Extracting windowed features (replicating CalcFeatures_Acc_Gyro.m)...")
    frames = extract_window_features(df, behavior_vocab)
    back_df = frames["Back"]
    print(f"Generated {len(back_df)} windows for the Back sensor.")

    print("Assigning labels (replicating CollectFeatures.m)...")
    labels, dog_ids, test_nums, valid_windows = assign_labels(back_df, behavior_vocab)
    print(f"Retained {len(labels)} pure windows (≥75 % single behaviour).")

    print("Saving processed feature matrices...")
    save_processed_features(frames, labels, dog_ids, test_nums, valid_windows)


if __name__ == "__main__":
    main()
