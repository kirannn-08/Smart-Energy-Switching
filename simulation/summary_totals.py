import pandas as pd

def compute_energy_totals(csv_path: str, step_minutes: int = 1):
    """
    Computes total energy usage and PV flows from the simulation CSV.

    Returns:
        A dict containing:
            - total_load_kwh
            - total_pv_gen_kwh
            - total_pv_used_kwh
            - total_solar_used_kwh     (pv_to_load + battery_discharge)
            - total_grid_used_kwh
            - total_pv_wasted_kwh
    """
    df = pd.read_csv(csv_path)

    dt_hours = step_minutes / 60.0

    # Extract columns
    pv = df["pv_power_kw"].sum() * dt_hours
    pv_to_load = df["pv_to_load_kw"].sum() * dt_hours
    pv_to_batt = df["pv_to_battery_kw"].sum() * dt_hours
    battery_discharge = df["battery_discharge_kw"].sum() * dt_hours
    grid_to_load = df["grid_to_load_kw"].sum() * dt_hours
    wasted = df["pv_wasted_kw"].sum() * dt_hours

    # Total load = solar-side + grid-side load
    total_load = pv_to_load + battery_discharge + grid_to_load

    # PV used directly or indirectly through battery
    pv_used = pv_to_load + pv_to_batt

    # Total solar used = PV used directly + battery discharge
    solar_used = pv_to_load + battery_discharge

    return {
        "csv_path": csv_path,
        "total_load_kwh": round(total_load, 3),
        "total_pv_gen_kwh": round(pv, 3),
        "total_pv_used_kwh": round(pv_used, 3),
        "total_solar_used_kwh": round(solar_used, 3),
        "total_grid_used_kwh": round(grid_to_load, 3),
        "total_pv_wasted_kwh": round(wasted, 3),
    }
