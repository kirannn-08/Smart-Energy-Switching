import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from pv_dataset import PVSolarForecastDataset, default_csv_paths
from pv_lstm_train import PVSolarLSTM


def plot_val_sequence(
    model_path: str = "ml/pv_lstm.pt",
    input_window: int = 60,
    forecast_horizon: int = 15,
    num_points: int = 200,
):
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

    loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = PVSolarLSTM(input_size=2, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    true_vals = []
    pred_vals = []

    with torch.no_grad():
        for i, (x, y_true) in enumerate(loader):
            if i >= num_points:
                break
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = model(x)
            # clamp to non-negative for nicer plotting
            y_pred = torch.clamp(y_pred, min=0.0)

            true_vals.append(y_true.item())
            pred_vals.append(y_pred.item())

    plt.figure(figsize=(12, 4))
    plt.plot(true_vals, label="True PV (kW)")
    plt.plot(pred_vals, label="Predicted PV (kW)", linestyle="--")
    plt.xlabel("Validation sample index")
    plt.ylabel("PV power (kW)")
    plt.title("PV Forecast: True vs Predicted on Validation Samples")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_val_sequence()
