from pv_dataset import PVSolarForecastDataset, default_csv_paths

if __name__ == "__main__":
    csvs = default_csv_paths()
    print("Found CSVs:")
    for c in csvs:
        print("  ", c)

    ds_train = PVSolarForecastDataset(csv_paths=csvs, input_window=60, forecast_horizon=15, train=True)
    ds_val = PVSolarForecastDataset(csv_paths=csvs, input_window=60, forecast_horizon=15, train=False)

    print("\nTrain size:", len(ds_train))
    print("Val size  :", len(ds_val))

    x, y = ds_train[0]
    print("Sample x shape:", x.shape)
    print("Sample y shape:", y.shape)
    print("Sample target (kW):", y.item())
