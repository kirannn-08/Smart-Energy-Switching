import os
import pandas as pd


def analyse_daily_with_battery(
    csv_path: str = "data/daily_sim_with_battery.csv",
    step_minutes: int = 1,
):
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}. Run run_daily_sim_with_battery.py first.")
        return

    df = pd.read_csv(csv_path)

    step_hours = step_minutes / 60.0

    # --- PV & battery energy flows ---
    total_pv_gen_kwh = (df["pv_power_kw"].sum()) * step_hours
    total_pv_to_load_kwh = (df["pv_to_load_kw"].sum()) * step_hours
    total_pv_to_batt_kwh = (df["pv_to_battery_kw"].sum()) * step_hours
    total_pv_wasted_kwh = (df["pv_wasted_kw"].sum()) * step_hours

    total_grid_to_load_kwh = (df["grid_to_load_kw"].sum()) * step_hours

    total_batt_charge_kwh = (df["battery_charge_kw"].sum()) * step_hours
    total_batt_discharge_kwh = (df["battery_discharge_kw"].sum()) * step_hours

    # --- Per-circuit totals (regardless of source) ---
    circuits = ["bedroom", "hall", "kitchen", "washing_machine", "motor_pump", "geyser"]
    circuit_energy = {}
    for name in circuits:
        col = f"{name}_kw"
        if col not in df.columns:
            continue
        circuit_energy[name] = (df[col].sum()) * step_hours

    # --- Print summary ---
    print("\n=== Daily Simulation WITH Battery — Energy Summary ===\n")
    print(f"Step duration           : {step_minutes} minute(s)")
    print(f"Total PV generated      : {total_pv_gen_kwh:.2f} kWh")
    print(f"  - PV -> direct loads  : {total_pv_to_load_kwh:.2f} kWh")
    print(f"  - PV -> battery       : {total_pv_to_batt_kwh:.2f} kWh")
    print(f"  - PV wasted           : {total_pv_wasted_kwh:.2f} kWh\n")

    print(f"Total grid -> loads     : {total_grid_to_load_kwh:.2f} kWh")
    print(f"Battery charge energy   : {total_batt_charge_kwh:.2f} kWh")
    print(f"Battery discharge energy: {total_batt_discharge_kwh:.2f} kWh\n")

    print("Per-circuit total consumption (all sources):")
    for name, kwh in circuit_energy.items():
        print(f"  {name:16s} : {kwh:.2f} kWh")
    print()

    # Rough check for consistency
    total_load_kwh = sum(circuit_energy.values())
    print(f"Approx total load (sum of circuits): {total_load_kwh:.2f} kWh")
    print("Note: small differences may appear due to rounding/efficiencies.\n")


if __name__ == "__main__":
    analyse_daily_with_battery()
