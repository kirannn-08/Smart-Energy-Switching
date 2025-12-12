# ml/drift_detector.py
"""
Drift detector: evaluates current PV + Load models on recent archive days,
logs daily MAE into ml/metrics_log.csv and triggers a retrain if MAE degrades.

Run from project root (cron-friendly):
    python3 ml/drift_detector.py
"""

import os
import glob
import csv
import numpy as np
import torch
from datetime import datetime, timedelta

from pv_dataset import PVSolarForecastDataset, default_csv_paths as pv_default_csvs
from pv_lstm_train import PVSolarLSTM
from load_dataset import MultiLoadForecastDataset, default_csv_paths as load_default_csvs
from load_lstm_train import MultiLoadLSTM

PV_MODEL = "ml/pv_lstm_latest.pt"
LOAD_MODEL = "ml/load_lstm_latest.pt"
METRICS_LOG = "ml/metrics_log.csv"
RECENT_DAYS = 7
DRIFT_MULTIPLIER = 1.25  # trigger if current MAE > DRIFT_MULTIPLIER * mean(last N MAE)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_pv(csv_paths):
    ds = PVSolarForecastDataset(csv_paths=csv_paths, input_window=60, forecast_horizon=15, train=False)
    if len(ds) == 0:
        return None
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    model = PVSolarLSTM(input_size=2, hidden_size=64, num_layers=2).to(DEVICE)
    model.load_state_dict(torch.load(PV_MODEL, map_location=DEVICE))
    model.eval()
    errors = []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            yp = model(x)
            yp = torch.clamp(yp, min=0.0)
            errors.append(torch.abs(yp - y).cpu().numpy())
    if not errors:
        return None
    errors = np.concatenate(errors).ravel()
    return float(np.mean(errors))  # MAE in kW

def evaluate_load(csv_paths):
    ds = MultiLoadForecastDataset(csv_paths=csv_paths, input_window=60, forecast_horizon=1, include_pv=True, train=False)
    if len(ds) == 0:
        return None
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    model = MultiLoadLSTM(input_size=ds.x.shape[2], hidden_size=128, num_layers=2).to(DEVICE)
    model.load_state_dict(torch.load(LOAD_MODEL, map_location=DEVICE))
    model.eval()
    errors = []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            yp = model(x)
            yp = torch.clamp(yp, min=0.0)
            errors.append(torch.mean(torch.abs(yp - y), dim=1).cpu().numpy())
    if not errors:
        return None
    errors = np.concatenate(errors).ravel()
    return float(np.mean(errors))  # MAE averaged across circuits

def read_recent_metrics(log_file=METRICS_LOG, days=RECENT_DAYS):
    if not os.path.exists(log_file):
        return []
    rows = []
    with open(log_file, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    # filter last N days
    out = []
    cutoff = datetime.now() - timedelta(days=days)
    for r in rows:
        dt = datetime.fromisoformat(r["timestamp"])
        if dt >= cutoff:
            out.append(r)
    return out

def append_metric(pv_mae, load_mae, log_file=METRICS_LOG):
    existed = os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        w = csv.writer(f)
        if not existed:
            w.writerow(["timestamp", "pv_mae", "load_mae"])
        w.writerow([datetime.now().isoformat(), pv_mae if pv_mae is not None else "", load_mae if load_mae is not None else ""])

if __name__ == "__main__":
    # select a validation set: use last X archive files if available
    archive_files = sorted(glob.glob("data/archive/*.csv"), key=os.path.getmtime, reverse=True)
    if not archive_files:
        # fallback: use any recent daily_sim files
        archive_files = sorted(glob.glob("data/daily_sim_*.csv"), key=os.path.getmtime, reverse=True)

    val_files = archive_files[:5]  # evaluate on last 5 files
    if not val_files:
        print("No files found for evaluation.")
        exit(0)

    print("Evaluating on:", val_files)
    pv_mae = evaluate_pv(val_files)  # may be None if model missing
    load_mae = evaluate_load(val_files)

    print(f"PV MAE: {pv_mae}, Load MAE: {load_mae}")
    append_metric(pv_mae, load_mae)

    recent = read_recent_metrics()
    pv_maes = [float(r["pv_mae"]) for r in recent if r["pv_mae"]]
    load_maes = [float(r["load_mae"]) for r in recent if r["load_mae"]]

    do_retrain = False
    if pv_mae is not None and pv_maes:
        mean_pv = sum(pv_maes) / len(pv_maes)
        if pv_mae > mean_pv * DRIFT_MULTIPLIER:
            print("PV drift detected:", pv_mae, ">", mean_pv * DRIFT_MULTIPLIER)
            do_retrain = True

    if load_mae is not None and load_maes:
        mean_load = sum(load_maes) / len(load_maes)
        if load_mae > mean_load * DRIFT_MULTIPLIER:
            print("Load drift detected:", load_mae, ">", mean_load * DRIFT_MULTIPLIER)
            do_retrain = True

    if do_retrain:
        print("Triggering full retrain...")
        # call retrain_full script
        os.system("python3 ml/retrain_full.py")
    else:
        print("No significant drift detected.")