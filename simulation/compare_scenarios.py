import os
import pandas as pd


def analyse_one(csv_path: str, step_minutes: int = 1):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    step_hours = step_minutes / 60.0

    total_pv_gen_kwh = (df["pv_power_kw"].sum()) * step_hours
    total_pv_to_load_kwh = (df["pv_to_load_kw"].sum()) * step_hours
    total_pv_to_batt_kwh = (df["pv_to_battery_kw"].sum()) * step_hours
    total_pv_wasted_kwh = (df["pv_wasted_kw"].sum()) * step_hours
    total_grid_to_load_kwh = (df["grid_to_load_kw"].sum()) * step_hours

    return {
        "pv_gen_kwh": total_pv_gen_kwh,
        "pv_to_load_kwh": total_pv_to_load_kwh,
        "pv_to_batt_kwh": total_pv_to_batt_kwh,
        "pv_wasted_kwh": total_pv_wasted_kwh,
        "grid_to_load_kwh": total_grid_to_load_kwh,
    }


def compare_scenarios():
    scenarios = [
        "baseline_normal",
        "cloudy_day",
        "small_batt",
        "big_batt",
    ]

    rows = []
    for name in scenarios:
        csv_path = f"data/daily_sim_with_battery_{name}.csv"
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping.")
            continue

        stats = analyse_one(csv_path, step_minutes=1)
        row = {
            "scenario": name,
            **stats,
        }
        rows.append(row)

    if not rows:
        print("No scenarios found.")
        return

    df = pd.DataFrame(rows)
    # Also compute effective solar usage (PV to load + PV to battery)
    df["pv_used_kwh"] = df["pv_to_load_kwh"] + df["pv_to_batt_kwh"]

    print("\n=== Scenario Comparison (Energy Flows) ===\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))
    print("\nNotes:")
    print("  - pv_gen_kwh     : Total PV generated")
    print("  - pv_used_kwh    : PV used either directly or via battery")
    print("  - pv_wasted_kwh  : PV not used (lost)")
    print("  - grid_to_load_kwh : Energy drawn from grid\n")


if __name__ == "__main__":
    compare_scenarios()
