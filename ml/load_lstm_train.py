# ml/load_lstm_train.py  (REPLACE your existing file with this)
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# import dataset (assumes ml/load_dataset.py exists and provides default_csv_paths)
from load_dataset import MultiLoadForecastDataset, default_csv_paths


class MultiLoadLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2, out_size: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        y = self.fc(last)
        return y


def get_project_root():
    # Try to find project root (parent of ml/)
    this_dir = os.path.dirname(os.path.abspath(__file__))  # ml/
    project_root = os.path.dirname(this_dir)
    return project_root


def train_load_lstm(
    input_window: int = 60,
    forecast_horizon: int = 1,
    batch_size: int = 64,
    num_epochs: int = 20,
    lr: float = 1e-3,
    model_path: str | None = None,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    csvs = default_csv_paths()
    if not csvs:
        raise RuntimeError("No CSVs found for load training. Check data/ folder and paths in load_dataset.default_csv_paths()")

    print("Using CSVs:")
    for c in csvs: print("  -", c)

    train_ds = MultiLoadForecastDataset(csv_paths=csvs, input_window=input_window, forecast_horizon=forecast_horizon, include_pv=True, train=True)
    val_ds = MultiLoadForecastDataset(csv_paths=csvs, input_window=input_window, forecast_horizon=forecast_horizon, include_pv=True, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_size = train_ds.x.shape[2]
    model = MultiLoadLSTM(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2, out_size=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTrain samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print("Device:", device)
    print("Starting training...\n")

    # If no explicit model_path given, save under project_root/ml/load_lstm.pt
    if model_path is None:
        project_root = get_project_root()
        model_path = os.path.join(project_root, "ml", "load_lstm.pt")

    best_val = float("inf")
    try:
        for epoch in range(1, num_epochs+1):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device); yb = yb.to(device)
                optimizer.zero_grad()
                yp = model(xb)
                loss = criterion(yp, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_ds)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    yp = model(xb)
                    val_loss += criterion(yp, yb).item() * xb.size(0)
            val_loss /= len(val_ds) if len(val_ds) > 0 else 1.0

            print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f"  -> New best model saved to: {model_path}")

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    # Forced final save (always)
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"[FINAL SAVE] Model saved to: {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save model to {model_path}: {e}")

    return model


def quick_eval(model_path="ml/load_lstm.pt", input_window=60, forecast_horizon=1, num_samples=10, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    csvs = default_csv_paths()
    val_ds = MultiLoadForecastDataset(csv_paths=csvs, input_window=input_window, forecast_horizon=forecast_horizon, include_pv=True, train=False)
    loader = DataLoader(val_ds, batch_size=1, shuffle=True)
    model = MultiLoadLSTM(input_size=val_ds.x.shape[2], hidden_size=128, num_layers=2, dropout=0.2, out_size=3).to(device)

    if not os.path.exists(model_path):
        print(f"[WARN] model file not found: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("\nQuick eval (bedroom,hall,kitchen) true vs pred:")
    with torch.no_grad():
        for i, (x,y) in enumerate(loader):
            if i>=num_samples: break
            x = x.to(device); y = y.to(device)
            yp = model(x)
            yp = torch.clamp(yp, min=0.0)
            print(f"{i+1:02d}: true=[{y[0,0]:.3f},{y[0,1]:.3f},{y[0,2]:.3f}]  pred=[{yp[0,0]:.3f},{yp[0,1]:.3f},{yp[0,2]:.3f}]")


if __name__ == "__main__":
    # Running as script: use sensible defaults
    model = train_load_lstm(
        input_window=60,
        forecast_horizon=1,
        batch_size=64,
        num_epochs=15,
        lr=1e-3,
        model_path=None,   # will be resolved to project_root/ml/load_lstm.pt
        device=None,
    )

    # Quick eval (only if model exists)
    project_root = get_project_root()
    model_path = os.path.join(project_root, "ml", "load_lstm.pt")
    quick_eval(model_path=model_path, input_window=60, forecast_horizon=1, num_samples=10)
