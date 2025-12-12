# ml/retrain_daily.py
"""
Fine-tune PV and Load models on the most-recent simulation CSVs.
Run from project root:
    python3 ml/retrain_daily.py
"""

import os
import glob
import torch
import datetime

from pv_dataset import PVSolarForecastDataset, default_csv_paths as pv_default_csvs
from pv_lstm_train import PVSolarLSTM
from load_dataset import MultiLoadForecastDataset, default_csv_paths as load_default_csvs
from load_lstm_train import MultiLoadLSTM

# ---- config ----
NUM_RECENT = 3           # number of newest CSV files to fine-tune on
PV_MODEL_PATH = "ml/pv_lstm_latest.pt"
LOAD_MODEL_PATH = "ml/load_lstm_latest.pt"
PV_MODEL_SAVE = "ml/pv_lstm_latest.pt"
LOAD_MODEL_SAVE = "ml/load_lstm_latest.pt"
PV_INPUT_WINDOW = 60
PV_FORECAST_HORIZON = 15
LOAD_INPUT_WINDOW = 60
LOAD_FORECAST_HORIZON = 1
EPOCHS = 3
LR = 1e-4
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pick_recent_csvs(patterns=("data/daily_sim_*.csv", "data/daily_sim_with_battery_*.csv"), n=NUM_RECENT):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(files, key=os.path.getmtime, reverse=True)
    return files[:n]


def fine_tune_pv(csv_paths, model_path=PV_MODEL_PATH):
    print("Fine-tuning PV model on:", csv_paths)
    ds = PVSolarForecastDataset(csv_paths=csv_paths, input_window=PV_INPUT_WINDOW, forecast_horizon=PV_FORECAST_HORIZON, train=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = PVSolarLSTM(input_size=2, hidden_size=64, num_layers=2).to(DEVICE)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Loaded existing PV model:", model_path)
    else:
        print("No existing PV model found; training from scratch.")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for e in range(EPOCHS):
        total = 0.0
        for x,y in loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            opt.zero_grad()
            yp = model(x)
            loss = loss_fn(yp, y)
            loss.backward()
            opt.step()
            total += loss.item()*x.size(0)
        print(f"PV fine-tune epoch {e+1}/{EPOCHS} loss={total/len(ds):.6f}")

    os.makedirs(os.path.dirname(PV_MODEL_SAVE), exist_ok=True)
    torch.save(model.state_dict(), PV_MODEL_SAVE)
    print("Saved PV model to", PV_MODEL_SAVE)
    return PV_MODEL_SAVE


def fine_tune_load(csv_paths, model_path=LOAD_MODEL_PATH):
    print("Fine-tuning Load model on:", csv_paths)
    ds = MultiLoadForecastDataset(csv_paths=csv_paths, input_window=LOAD_INPUT_WINDOW, forecast_horizon=LOAD_FORECAST_HORIZON, include_pv=True, train=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = MultiLoadLSTM(input_size=ds.x.shape[2], hidden_size=128, num_layers=2).to(DEVICE)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Loaded existing Load model:", model_path)
    else:
        print("No existing Load model found; training from scratch.")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for e in range(EPOCHS):
        total = 0.0
        for x,y in loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            opt.zero_grad()
            yp = model(x)
            loss = loss_fn(yp, y)
            loss.backward()
            opt.step()
            total += loss.item()*x.size(0)
        print(f"Load fine-tune epoch {e+1}/{EPOCHS} loss={total/len(ds):.6f}")

    os.makedirs(os.path.dirname(LOAD_MODEL_SAVE), exist_ok=True)
    torch.save(model.state_dict(), LOAD_MODEL_SAVE)
    print("Saved Load model to", LOAD_MODEL_SAVE)
    return LOAD_MODEL_SAVE


if __name__ == "__main__":
    recent = pick_recent_csvs()
    if not recent:
        print("No recent CSVs found. Falling back to default CSV list.")
        recent = pv_default_csvs()[:NUM_RECENT]

    pv_saved = fine_tune_pv(recent)
    load_saved = fine_tune_load(recent)

    print("\nFine-tune completed:", datetime := datetime if False else "")
    print("PV model:", pv_saved)
    print("Load model:", load_saved)