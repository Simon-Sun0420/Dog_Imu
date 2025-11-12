#!/usr/bin/env python3
"""
Utility script to parse dog_imu_lodo.py console output and append the
completed metrics to results/model_summary.csv and results/per_dog_metrics.csv.

Usage:
    1. Copy the console output (from “=== Sensor: ... ===” down to the end)
       into a text file, e.g. completed_log.txt.
    2. Run:
           python append_results_from_log.py --log completed_log.txt

Only the sections that contain both the “Sensor [...] - MODEL: Acc=...”
line and the following “Per-dog accuracy” block will be recorded. The
script is idempotent; if you run it multiple times with the same log
snippet, you may want to deduplicate rows manually.
"""

import argparse
import ast
import csv
import re
from pathlib import Path


def parse_log(log_text: str):
    entries = []
    current_sensor = None
    current_scenario = None
    current_model = None
    current_features = []
    collecting_dogs = False
    dog_rows = []
    current_entry = None

    sensor_pattern = re.compile(r"=== Sensor: (.+?) \| Scenario: (.+?) ===")
    model_pattern = re.compile(
        r"^(?P<sensor>\w+)\s+\[(?P<scenario>[^\]]+)\]\s+-\s+(?P<model>\w+):\s+Acc=(?P<acc>[0-9.]+),\s+F1_macro=(?P<f1>[0-9.]+)"
    )
    dog_pattern = re.compile(r"Dog\s+(\d+):\s+([0-9.]+)")

    lines = log_text.splitlines()
    for line in lines + [""]:  # extra empty line to flush last block
        line = line.rstrip()
        sensor_match = sensor_pattern.match(line)
        if sensor_match:
            current_sensor, current_scenario = sensor_match.groups()
            continue

        if line.startswith("Selecting features for "):
            current_model = line.split("for ", 1)[1].split("...", 1)[0].strip()
            current_features = []
            continue

        if line.startswith("Selected") and "features:" in line:
            features_str = line.split("features:", 1)[1].strip()
            try:
                current_features = ast.literal_eval(features_str)
            except Exception:
                current_features = []
            continue

        model_match = model_pattern.match(line)
        if model_match:
            current_entry = {
                "sensor": model_match.group("sensor"),
                "scenario": model_match.group("scenario"),
                "model": model_match.group("model"),
                "accuracy": float(model_match.group("acc")),
                "f1_macro": float(model_match.group("f1")),
                "features": current_features[:],
            }
            collecting_dogs = False
            dog_rows = []
            continue

        if line.startswith("Per-dog accuracy"):
            collecting_dogs = True
            dog_rows = []
            continue

        if collecting_dogs and line.startswith("  Dog"):
            dog_match = dog_pattern.search(line)
            if dog_match:
                dog_rows.append(
                    {"dog_id": int(dog_match.group(1)), "accuracy": float(dog_match.group(2))}
                )
            continue

        if collecting_dogs and line.strip() == "":
            if current_entry and dog_rows:
                current_entry["per_dog"] = dog_rows[:]
                entries.append(current_entry)
            collecting_dogs = False
            current_entry = None
            dog_rows = []
            continue

    return entries


def append_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Append parsed log results to CSVs.")
    parser.add_argument("--log", required=True, help="Path to the saved console output.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing model_summary.csv and per_dog_metrics.csv",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"{log_path} does not exist.")

    entries = parse_log(log_path.read_text())
    if not entries:
        print("No completed model sections found in the log.")
        return

    summary_rows = []
    dog_rows = []
    for entry in entries:
        summary_rows.append(
            {
                "sensor": entry["sensor"],
                "scenario": entry["scenario"],
                "model": entry["model"],
                "num_features": len(entry.get("features", [])),
                "selected_features": ";".join(entry.get("features", [])),
                "accuracy": entry["accuracy"],
                "f1_macro": entry["f1_macro"],
            }
        )
        for dog in entry.get("per_dog", []):
            dog_rows.append(
                {
                    "sensor": entry["sensor"],
                    "scenario": entry["scenario"],
                    "model": entry["model"],
                    "dog_id": dog["dog_id"],
                    "accuracy": dog["accuracy"],
                }
            )

    results_dir = Path(args.results_dir)
    summary_csv = results_dir / "model_summary.csv"
    per_dog_csv = results_dir / "per_dog_metrics.csv"

    append_csv(
        summary_csv,
        ["sensor", "scenario", "model", "num_features", "selected_features", "accuracy", "f1_macro"],
        summary_rows,
    )
    append_csv(
        per_dog_csv,
        ["sensor", "scenario", "model", "dog_id", "accuracy"],
        dog_rows,
    )

    print(f"Appended {len(summary_rows)} summary rows and {len(dog_rows)} per-dog rows.")


if __name__ == "__main__":
    main()
