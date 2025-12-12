# ml/load_dataset.py
import os
import glob
from typing import Sequence, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiLoadForecastDataset(Dataset):
    """
    Multi-output dataset to forecast per-circuit loads.

    Features per timestep:
      - bedroom_kw, hall_kw, kitchen_kw
      - pv_power_kw (optional, helpful)
      - minute_of_day normalized

    Target:
      - vector [bedroom_kw, hall_kw, kitchen_kw] at t + forecast_horizon

    The dataset concatenates all provided CSVs in temporal order and
    constructs sliding windows.
    """

    def __init__(
        self,
        csv_paths: Sequence[str],
        input_window: int = 60,
        forecast_horizon: int = 1,
        include_pv: bool = True,
        train: bool = True,
        train_split_ratio: float = 0.8,
    ):
        super().__init__()
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.include_pv = include_pv

        # load CSVs
        dfs = []
        for p in csv_paths:
            if os.path.exists(p):
                dfs.append(pd.read_csv(p))
            else:
                print(f"[WARN] CSV not found: {p}")

        if not dfs:
            raise RuntimeError("No CSVs found for MultiLoadForecastDataset")

        df = pd.concat(dfs, ignore_index=True)

        # required columns
        for col in ("bedroom_kw", "hall_kw", "kitchen_kw", "minute_of_day"):
            if col not in df.columns:
                raise RuntimeError(f"Required column missing in CSVs: {col}")

        bedroom = df["bedroom_kw"].values.astype(np.float32)
        hall = df["hall_kw"].values.astype(np.float32)
        kitchen = df["kitchen_kw"].values.astype(np.float32)
        minute = df["minute_of_day"].values.astype(np.float32)
        minute_norm = minute / 1440.0

        features_list = [bedroom, hall, kitchen]
        if include_pv and "pv_power_kw" in df.columns:
            pv = df["pv_power_kw"].values.astype(np.float32)
            features_list.append(pv)

        features_list.append(minute_norm)

        # shape [T, num_features]
        features = np.stack(features_list, axis=1)

        # targets: future loads for 3 circuits
        targets = np.stack([bedroom, hall, kitchen], axis=1)  # [T, 3]

        seq_x = []
        seq_y = []

        max_start = len(features) - input_window - forecast_horizon + 1
        for start in range(max_start):
            end = start + input_window
            target_idx = end + forecast_horizon - 1

            x = features[start:end]         # [input_window, F]
            y = targets[target_idx]         # [3]

            seq_x.append(x)
            seq_y.append(y)

        seq_x = np.stack(seq_x, axis=0)  # [N, input_window, F]
        seq_y = np.stack(seq_y, axis=0)  # [N, 3]

        # train/val split
        split = int(train_split_ratio * len(seq_x))
        if train:
            self.x = torch.from_numpy(seq_x[:split])
            self.y = torch.from_numpy(seq_y[:split])
        else:
            self.x = torch.from_numpy(seq_x[split:])
            self.y = torch.from_numpy(seq_y[split:])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def default_csv_paths() -> List[str]:
    # reuse the same CSV list you have
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
    final = []
    for p in candidates:
        if os.path.exists(p):
            final.append(p)
    # add pattern-based ones too
    final.extend(glob.glob("data/daily_sim_with_battery_*.csv"))
    # dedupe preserving order
    seen = set(); out=[]
    for p in final:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


if __name__ == "__main__":
    paths = default_csv_paths()
    print("CSV paths:", paths)
    ds = MultiLoadForecastDataset(paths, input_window=60, forecast_horizon=1, include_pv=True, train=True)
    ds_val = MultiLoadForecastDataset(paths, input_window=60, forecast_horizon=1, include_pv=True, train=False)
    print("Train size:", len(ds))
    print("Val size:", len(ds_val))
    x,y = ds[0]
    print("x shape:", x.shape, "y shape:", y.shape)
