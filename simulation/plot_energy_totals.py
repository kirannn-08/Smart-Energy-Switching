import os
import pandas as pd
import matplotlib.pyplot as plt
from summary_totals import compute_energy_totals


def load_totals():
    scenarios = {
        "baseline": "data/daily_sim_with_battery_baseline_normal.csv",
        "cloudy_day": "data/daily_sim_with_battery_cloudy_day.csv",
        "small_batt": "data/daily_sim_with_battery_small_batt.csv",
        "big_batt": "data/daily_sim_with_battery_big_batt.csv",
        "cloud_shock": "data/daily_sim_cloud_shock.csv",
        "multi_spike": "data/daily_sim_multi_spike.csv",
        "batt_emergency": "data/daily_sim_batt_emergency.csv",
        "grid_blackout": "data/daily_sim_grid_blackout.csv",
        "user_misconfig": "data/daily_sim_user_misconfig.csv",
    }

    rows = []
    for name, path in scenarios.items():
        if not os.path.exists(path):
            continue
        d = compute_energy_totals(path)
        d["scenario"] = name
        rows.append(d)

    df = pd.DataFrame(rows)
    return df


def plot_totals(df: pd.DataFrame):
    # Extract values
    scenarios = df["scenario"].tolist()
    x = range(len(scenarios))

    pv_used = df["total_pv_used_kwh"].tolist()
    grid_used = df["total_grid_used_kwh"].tolist()
    pv_wasted = df["total_pv_wasted_kwh"].tolist()

    # -----------------------------
    # Plot A: PV Used, Grid Used, PV Wasted
    # -----------------------------
    plt.figure(figsize=(14, 6))

    width = 0.25
    plt.bar([i - width for i in x], pv_used, width, label="PV Used (kWh)")
    plt.bar(x, grid_used, width, label="Grid Used (kWh)")
    plt.bar([i + width for i in x], pv_wasted, width, label="PV Wasted (kWh)")

    plt.xticks(list(x), scenarios, rotation=45, ha="right")
    plt.ylabel("Energy (kWh)")
    plt.title("Energy Comparison Across Scenarios")
    plt.legend()
    plt.grid(axis="y")

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot B: Solar Used vs Total Load
    # -----------------------------
    plt.figure(figsize=(14, 6))

    solar_used = df["total_solar_used_kwh"].tolist()
    total_load = df["total_load_kwh"].tolist()

    width = 0.35
    plt.bar([i - width/2 for i in x], solar_used, width, label="Solar Used (kWh)")
    plt.bar([i + width/2 for i in x], total_load, width, label="Total Load (kWh)")

    plt.xticks(list(x), scenarios, rotation=45, ha="right")
    plt.ylabel("Energy (kWh)")
    plt.title("Solar Contribution vs Total Load per Scenario")
    plt.legend()
    plt.grid(axis="y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_totals()
    print(df)
    plot_totals(df)
