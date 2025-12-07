import os
import pandas as pd
from summary_totals import compute_energy_totals

def print_totals_for_scenarios():
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
        summary = compute_energy_totals(path)
        summary["scenario"] = name
        rows.append(summary)

    if not rows:
        print("No scenario CSVs found.")
        return

    df = pd.DataFrame(rows)
    df = df[
        [
            "scenario",
            "total_load_kwh",
            "total_pv_gen_kwh",
            "total_pv_used_kwh",
            "total_solar_used_kwh",
            "total_grid_used_kwh",
            "total_pv_wasted_kwh",
        ]
    ]

    print("\n=== Total Energy Summary Across Scenarios ===\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    print_totals_for_scenarios()
