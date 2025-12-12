# ml/load_lstm_plot.py
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from load_dataset import MultiLoadForecastDataset, default_csv_paths
from load_lstm_train import MultiLoadLSTM

def plot_predictions(model_path="ml/load_lstm.pt", input_window=60, forecast_horizon=1, n=300):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    csvs = default_csv_paths()
    val_ds = MultiLoadForecastDataset(csv_paths=csvs, input_window=input_window, forecast_horizon=forecast_horizon, include_pv=True, train=False)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    model = MultiLoadLSTM(input_size=val_ds.x.shape[2], hidden_size=128, num_layers=2, dropout=0.2, out_size=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tr_b, tr_h, tr_k = [], [], []
    pr_b, pr_h, pr_k = [], [], []

    with torch.no_grad():
        for i, (x,y) in enumerate(loader):
            if i>=n: break
            x = x.to(device)
            y = y.to(device)
            yp = model(x)
            yp = torch.clamp(yp, min=0.0)
            tr_b.append(y[0,0].item()); tr_h.append(y[0,1].item()); tr_k.append(y[0,2].item())
            pr_b.append(yp[0,0].item()); pr_h.append(yp[0,1].item()); pr_k.append(yp[0,2].item())

    plt.figure(figsize=(14,6))
    plt.subplot(3,1,1); plt.plot(tr_b, label="true"); plt.plot(pr_b, '--', label="pred"); plt.title("Bedroom"); plt.legend(); plt.grid(True)
    plt.subplot(3,1,2); plt.plot(tr_h, label="true"); plt.plot(pr_h, '--', label="pred"); plt.title("Hall"); plt.legend(); plt.grid(True)
    plt.subplot(3,1,3); plt.plot(tr_k, label="true"); plt.plot(pr_k, '--', label="pred"); plt.title("Kitchen"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    plot_predictions()
