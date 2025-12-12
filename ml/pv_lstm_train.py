import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pv_dataset import PVSolarForecastDataset, default_csv_paths


class PVSolarLSTM(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)          # out: [batch, seq_len, hidden]
        last_hidden = out[:, -1, :]    # [batch, hidden]
        y_hat = self.fc(last_hidden)   # [batch, 1]
        return y_hat


def train_pv_lstm(
    input_window: int = 60,
    forecast_horizon: int = 15,
    batch_size: int = 64,
    num_epochs: int = 20,
    lr: float = 1e-3,
    model_path: str = "ml/pv_lstm.pt",
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    csvs = default_csv_paths()
    if not csvs:
        raise RuntimeError("No CSV files found for training. Check data/ folder.")

    print("Using CSVs for training:")
    for c in csvs:
        print("  -", c)

    train_ds = PVSolarForecastDataset(
        csv_paths=csvs,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        train=True,
    )
    val_ds = PVSolarForecastDataset(
        csv_paths=csvs,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PVSolarLSTM(input_size=2, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTrain samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Training on device: {device}\n")

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)  # [B, T, 2]
            y_batch = y_batch.to(device)  # [B, 1]

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x_batch.size(0)

        train_loss = train_loss_sum / len(train_ds)

        # ---- Validate ----
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss_sum += loss.item() * x_batch.size(0)

        val_loss = val_loss_sum / len(val_ds) if len(val_ds) > 0 else 0.0

        print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss and len(val_ds) > 0:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"  -> New best model saved to {model_path}")

    print("\nTraining complete.")
    return model


def quick_eval(
    model_path: str = "ml/pv_lstm.pt",
    input_window: int = 60,
    forecast_horizon: int = 15,
    num_samples: int = 10,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    csvs = default_csv_paths()
    val_ds = PVSolarForecastDataset(
        csv_paths=csvs,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        train=False,
    )

    if len(val_ds) == 0:
        print("No validation samples available.")
        return

    loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    model = PVSolarLSTM(input_size=2, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("\nQuick evaluation on random validation samples:")
    import itertools

    with torch.no_grad():
        for i, (x, y_true) in zip(range(num_samples), loader):
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = model(x)
            print(f"Sample {i+1:02d}: true={y_true.item():.3f} kW, pred={y_pred.item():.3f} kW")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = train_pv_lstm(
        input_window=60,
        forecast_horizon=15,
        batch_size=64,
        num_epochs=20,
        lr=1e-3,
        model_path="ml/pv_lstm.pt",
        device=device,
    )

    quick_eval(
        model_path="ml/pv_lstm.pt",
        input_window=60,
        forecast_horizon=15,
        num_samples=10,
        device=device,
    )
