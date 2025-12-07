import os
import pandas as pd
import matplotlib.pyplot as plt

from compare_scenarios import analyse_one
from metrics_scenarios import analyse_switches_and_time


def load_energy_comparison():
    """
    Uses analyse_one() from compare_scenarios.py to build
    an energy-flow comparison DataFrame for key scenarios.
    """
    scenarios = [
        ("baseline_normal", "data/daily_sim_with_battery_baseline_normal.csv"),
        ("cloudy_day", "data/daily_sim_with_battery_cloudy_day.csv"),
        ("small_batt", "data/daily_sim_with_battery_small_batt.csv"),
        ("big_batt", "data/daily_sim_with_battery_big_batt.csv"),
        # Add more if present:
        ("cloud_shock", "data/daily_sim_cloud_shock.csv"),
        ("multi_spike", "data/daily_sim_multi_spike.csv"),
        ("batt_emergency", "data/daily_sim_batt_emergency.csv"),
        ("grid_blackout", "data/daily_sim_grid_blackout.csv"),
        ("user_misconfig", "data/daily_sim_user_misconfig.csv"),
    ]

    rows = []
    for name, path in scenarios:
        if not os.path.exists(path):
            continue
        stats = analyse_one(path, step_minutes=1)
        row = {"scenario": name, **stats}
        row["pv_used_kwh"] = row["pv_to_load_kwh"] + row["pv_to_batt_kwh"]
        rows.append(row)

    if not rows:
        raise RuntimeError("No scenario CSVs found for energy comparison.")

    df = pd.DataFrame(rows)
    return df


def plot_energy_bars(df: pd.DataFrame):
    """
    Make bar charts for:
      - PV used vs PV wasted per scenario
      - Grid energy per scenario
    """
    scenarios = df["scenario"].tolist()
    x = range(len(scenarios))

    pv_used = df["pv_used_kwh"].values
    pv_wasted = df["pv_wasted_kwh"].values
    grid_used = df["grid_to_load_kwh"].values

    plt.figure(figsize=(12, 5))
    # Subplot 1: PV used vs wasted
    plt.subplot(1, 2, 1)
    width = 0.35
    plt.bar([i - width / 2 for i in x], pv_used, width, label="PV used (kWh)")
    plt.bar([i + width / 2 for i in x], pv_wasted, width, label="PV wasted (kWh)")
    plt.xticks(list(x), scenarios, rotation=45, ha="right")
    plt.ylabel("Energy (kWh)")
    plt.title("PV Used vs PV Wasted per Scenario")
    plt.legend()
    plt.grid(True, axis="y")

    # Subplot 2: Grid energy
    plt.subplot(1, 2, 2)
    plt.bar(x, grid_used)
    plt.xticks(list(x), scenarios, rotation=45, ha="right")
    plt.ylabel("Energy (kWh)")
    plt.title("Grid Energy Usage per Scenario")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.show()


def load_solar_time_metrics():
    """
    Uses analyse_switches_and_time() from metrics_scenarios.py
    to build a DataFrame of solar_pct_day for key scenarios and circuits.
    """
    scenarios = {
        "baseline_normal": "data/daily_sim_with_battery_baseline_normal.csv",
        "cloudy_day": "data/daily_sim_with_battery_cloudy_day.csv",
        "small_batt": "data/daily_sim_with_battery_small_batt.csv",
        "big_batt": "data/daily_sim_with_battery_big_batt.csv",
        "user_misconfig": "data/daily_sim_user_misconfig.csv",
    }

    frames = []
    for sc_name, path in scenarios.items():
        if not os.path.exists(path):
            continue
        df_m = analyse_switches_and_time(path, step_minutes=1)
        df_m.insert(0, "scenario", sc_name)
        frames.append(df_m)

    if not frames:
        raise RuntimeError("No scenario files found for solar time metrics.")

    full = pd.concat(frames, ignore_index=True)
    return full


def plot_solar_pct_heatmap_style(df_metrics: pd.DataFrame):
    """
    Not a real heatmap, but grouped bar chart of solar_pct_day per circuit
    across scenarios, for the main three circuits.
    """
    # Focus on bedroom, hall, kitchen
    df_metrics = df_metrics[df_metrics["circuit"].isin(["bedroom", "hall", "kitchen"])]

    scenarios = sorted(df_metrics["scenario"].unique())
    circuits = ["bedroom", "hall", "kitchen"]

    x = range(len(scenarios))
    width = 0.25

    plt.figure(figsize=(12, 5))

    for idx, circuit in enumerate(circuits):
        subset = df_metrics[df_metrics["circuit"] == circuit]
        # Align subset with scenarios order
        values = []
        for sc in scenarios:
            row = subset[subset["scenario"] == sc]
            if not row.empty:
                values.append(float(row["solar_pct_day"].iloc[0]))
            else:
                values.append(0.0)
        offsets = [i + (idx - 1) * width for i in x]  # -width, 0, +width
        plt.bar(offsets, values, width, label=circuit)

    plt.xticks(list(x), scenarios, rotation=45, ha="right")
    plt.ylabel("Time on Solar (% of day)")
    plt.title("Solar Usage per Circuit across Scenarios")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1) Energy flow comparison
    energy_df = load_energy_comparison()
    print("\nEnergy comparison:\n", energy_df)
    plot_energy_bars(energy_df)

    # 2) Solar time metrics comparison (per room)
    metrics_df = load_solar_time_metrics()
    print("\nSolar-time metrics:\n", metrics_df)
    plot_solar_pct_heatmap_style(metrics_df)
