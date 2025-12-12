import os
import glob
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PVSolarForecastDataset(Dataset):
    """
    Time-series dataset for forecasting pv_power_kw.

    - Can load one or multiple simulation CSV files.
    - Each sample:
        x: [input_window, num_features]
        y: [1] -> pv_power_kw at t + forecast_horizon

    Features we use (per timestep):
      - pv_power_kw
      - minute_of_day (normalized to [0, 1])
    """

    def __init__(
        self,
        csv_paths: Sequence[str],
        input_window: int = 60,      # past 60 minutes
        forecast_horizon: int = 15,  # predict 15 minutes ahead
        train: bool = True,
        train_split_ratio: float = 0.8,
    ):
        super().__init__()

        self.input_window = input_window
        self.forecast_horizon = forecast_horizon

        # --- Collect all rows from all CSVs ---
        all_dfs = []
        for path in csv_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                all_dfs.append(df)
            else:
                print(f"[WARN] CSV not found: {path}")

        if not all_dfs:
            raise RuntimeError("No valid CSV files found for PVSolarForecastDataset")

        full_df = pd.concat(all_dfs, ignore_index=True)

        # --- Build feature matrix ---
        pv = full_df["pv_power_kw"].values.astype(np.float32)
        minute = full_df["minute_of_day"].values.astype(np.float32)
        minute_norm = minute / 1440.0  # 24*60

        features = np.stack([pv, minute_norm], axis=1)  # [T, 2]
        targets = pv  # predict future pv

        # --- Sliding window over full sequence ---
        seq_x = []
        seq_y = []

        max_start = len(features) - input_window - forecast_horizon
        for start in range(max_start):
            end = start + input_window
            target_idx = end + forecast_horizon - 1

            x = features[start:end]      # [input_window, 2]
            y = targets[target_idx]      # scalar

            seq_x.append(x)
            seq_y.append(y)

        seq_x = np.stack(seq_x, axis=0)  # [N, input_window, 2]
        seq_y = np.stack(seq_y, axis=0)  # [N]

        # --- Simple train/val split ---
        split_idx = int(train_split_ratio * len(seq_x))
        if train:
            self.x = torch.from_numpy(seq_x[:split_idx])
            self.y = torch.from_numpy(seq_y[:split_idx]).unsqueeze(-1)  # [N,1]
        else:
            self.x = torch.from_numpy(seq_x[split_idx:])
            self.y = torch.from_numpy(seq_y[split_idx:]).unsqueeze(-1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def default_csv_paths() -> List[str]:
    """
    Helper to grab all relevant CSVs for training.
    You can add/remove files here as needed.
    """
    paths = []

    # Core “with battery” scenarios
    candidates = [
        "data/daily_sim_with_battery_baseline_normal.csv",
        "data/daily_sim_with_battery_cloudy_day.csv",
        "data/daily_sim_with_battery_small_batt.csv",
        "data/daily_sim_with_battery_big_batt.csv",
        "data/daily_sim_cloud_shock.csv",
        "data/daily_sim_multi_spike.csv",
        "data/daily_sim_batt_emergency.csv",
        "data/daily_sim_grid_blackout.csv",
        "data/daily_sim_user_misconfig.csv",
    ]

    for p in candidates:
        if os.path.exists(p):
            paths.append(p)

    # If you want, also auto-include any CSVs matching a pattern:
    paths.extend(glob.glob("data/daily_sim_with_battery_*.csv"))

    # Remove duplicates while preserving order
    seen = set()
    final_paths = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            final_paths.append(p)

    return final_paths


if __name__ == "__main__":
    # Small self-test
    csvs = default_csv_paths()
    print("Using CSVs:")
    for c in csvs:
        print("  -", c)

    ds_train = PVSolarForecastDataset(csv_paths=csvs, input_window=60, forecast_horizon=15, train=True)
    ds_val = PVSolarForecastDataset(csv_paths=csvs, input_window=60, forecast_horizon=15, train=False)

    print(f"\nTrain samples: {len(ds_train)}")
    print(f"Val samples  : {len(ds_val)}")

    x0, y0 = ds_train[0]
    print(f"x0 shape: {x0.shape}  (should be [60, 2])")
    print(f"y0 shape: {y0.shape}  (should be [1])")
    print(f"Example target value: {y0.item():.3f} kW")
