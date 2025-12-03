import os
import pandas as pd


def analyse_daily_results(csv_path: str = "data/daily_sim_result.csv", step_minutes: int = 1):
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}. Run run_daily_simulation.py first.")
        return

    df = pd.read_csv(csv_path)

    # Each row = one time step with duration = step_minutes
    step_hours = step_minutes / 60.0

    # Total PV energy generated [kWh]
    total_pv_kwh = (df["pv_power_kw"].sum()) * step_hours

    # Compute per-room solar/grid energy
    rooms = ["bedroom", "hall", "kitchen"]
    heavy = ["washing_machine", "motor_pump", "geyser"]

    results = {}

    for name in rooms + heavy:
        power_col = f"{name}_kw"
        source_col = f"{name}_source"

        if power_col not in df.columns or source_col not in df.columns:
            continue

        room_df = df[[power_col, source_col]].copy()

        # Solar energy = sum(power * dt) where source == "SOLAR"
        solar_mask = room_df[source_col] == "SOLAR"
        grid_mask = room_df[source_col] == "GRID"

        solar_kwh = (room_df.loc[solar_mask, power_col].sum()) * step_hours
        grid_kwh = (room_df.loc[grid_mask, power_col].sum()) * step_hours

        results[name] = {
            "solar_kwh": solar_kwh,
            "grid_kwh": grid_kwh,
            "total_kwh": solar_kwh + grid_kwh,
        }

    # Total house load energy (all rooms + heavy)
    total_load_kwh = sum(info["total_kwh"] for info in results.values())
    total_solar_used_kwh = sum(info["solar_kwh"] for info in results.values())
    total_grid_used_kwh = sum(info["grid_kwh"] for info in results.values())

    # PV wasted = generated - used_on_solar_side (if positive)
    pv_wasted_kwh = max(total_pv_kwh - total_solar_used_kwh, 0.0)

    # ---------- Print summary ----------
    print("\n=== Daily Simulation Energy Summary ===\n")
    print(f"Step duration        : {step_minutes} minute(s)")
    print(f"Total PV generated   : {total_pv_kwh:.2f} kWh")
    print(f"Total load consumption: {total_load_kwh:.2f} kWh")
    print(f"  - from SOLAR       : {total_solar_used_kwh:.2f} kWh")
    print(f"  - from GRID        : {total_grid_used_kwh:.2f} kWh")
    print(f"PV energy unused (wasted): {pv_wasted_kwh:.2f} kWh\n")

    print("Per-circuit breakdown:")
    for name, info in results.items():
        print(f"  {name:16s} | "
              f"total: {info['total_kwh']:.2f} kWh | "
              f"solar: {info['solar_kwh']:.2f} kWh | "
              f"grid: {info['grid_kwh']:.2f} kWh")
    print()


if __name__ == "__main__":
    analyse_daily_results()
