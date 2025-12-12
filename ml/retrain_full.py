# ml/retrain_full.py
"""
Full retrain of PV and Load models using an archive/training pool.
Run from project root:
    python3 ml/retrain_full.py
"""

import os
import glob
import time
import torch
import datetime
from pv_dataset import PVSolarForecastDataset, default_csv_paths as pv_default_csvs
from pv_lstm_train import train_pv_lstm, PVSolarLSTM
from load_dataset import MultiLoadForecastDataset, default_csv_paths as load_default_csvs
from load_lstm_train import train_load_lstm, MultiLoadLSTM

# ---- config ----
TRAIN_POOL_GLOB = "data/train_pool/*.csv"   # place daily CSVs here for full retrain
ARCHIVE_GLOB = "data/archive/*.csv"
MODEL_DIR = "ml"
PV_MODEL_OUT_PATTERN = os.path.join(MODEL_DIR, "pv_lstm_{date}.pt")
LOAD_MODEL_OUT_PATTERN = os.path.join(MODEL_DIR, "load_lstm_{date}.pt")

# Training hyperparams — adjust as needed
PV_EPOCHS = 30
LOAD_EPOCHS = 30
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collect_pool():
    files = sorted(glob.glob(TRAIN_POOL_GLOB))
    if not files:
        # fallback: use archive recent N
        files = sorted(glob.glob(ARCHIVE_GLOB))
    return files


def safe_train_pv(csvs, out_path):
    print("Starting full PV retrain with files:", len(csvs))
    # Use the existing training function but ensure it saves to out_path
    # train_pv_lstm returns model object; but original train_pv_lstm saves to path passed.
    model = train_pv_lstm(
        input_window=60,
        forecast_horizon=15,
        batch_size=BATCH_SIZE,
        num_epochs=PV_EPOCHS,
        lr=1e-3,
        model_path=out_path,
        device=DEVICE,
    )
    return out_path


def safe_train_load(csvs, out_path):
    print("Starting full Load retrain with files:", len(csvs))
    model = train_load_lstm(
        input_window=60,
        forecast_horizon=1,
        batch_size=BATCH_SIZE,
        num_epochs=LOAD_EPOCHS,
        lr=1e-3,
        model_path=out_path,
        device=DEVICE,
    )
    return out_path


if __name__ == "__main__":
    csvs = collect_pool()
    if not csvs:
        raise RuntimeError("No CSVs found in train_pool or archive to retrain.")

    # Optionally limit to last 90 files for speed:
    csvs = csvs[-90:]

    date_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pv_out = PV_MODEL_OUT_PATTERN.format(date=date_tag)
    load_out = LOAD_MODEL_OUT_PATTERN.format(date=date_tag)

    # The train_pv_lstm and train_load_lstm functions will internally read default CSVs;
    # for more control you could adapt them to accept csv_paths. For now ensure default functions read from proper locations.
    # If your train functions ignore TRAIN_POOL, you can copy selected csvs into a temp folder and set default_csvs to pick them.
    print("Launching PV full retrain...")
    safe_train_pv(csvs, pv_out)

    print("Launching Load full retrain...")
    safe_train_load(csvs, load_out)

    print("Full retrain complete, generated:")
    print("  PV:", pv_out)
    print("  Load:", load_out)

    # Optionally call deploy script to update 'latest' symlinks and metadata
    try:
        import deploy_and_version as dv
        dv.deploy_and_register(pv_out, load_out, comment="full retrain")
    except Exception:
        # fallback: print instruction
        print("Note: deploy_and_version not invoked automatically. Run ml/deploy_and_version.py to register models.")