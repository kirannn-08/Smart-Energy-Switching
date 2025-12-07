import math
from datetime import datetime
import pandas as pd

from run_daily_sim_with_battery import run_daily_sim_with_battery


def simulate_pv_with_cloud_shock(minute_of_day, peak_kw=1.0):
    """
    Normal PV curve except between 12:00–13:00
    PV is dropped to 10% (big cloud event).
    """

    sunrise = 6 * 60
    sunset = 18 * 60

    # Night time
    if minute_of_day < sunrise or minute_of_day > sunset:
        return 0.0

    # Normal clear-sky PV baseline
    x = (minute_of_day - sunrise) / (sunset - sunrise) * math.pi
    pv = peak_kw * math.sin(x)

    # Shock window: 12:00–13:00
    shock_start = 12 * 60
    shock_end = shock_start + 60  # 1 hour window

    if shock_start <= minute_of_day < shock_end:
        pv *= 0.10  # 90% drop

    return max(pv, 0.0)


def run_cloud_shock_scenario():
    print("\n=== RUNNING CLOUD SHOCK SCENARIO ===\n")

    # Run the daily simulation but override ONLY the PV function
    run_daily_sim_with_battery(
        date=datetime.now(),
        step_minutes=1,
        pv_peak_kw=1.0,
        battery_capacity_kwh=2.0,
        battery_initial_soc_ratio=0.5,
        battery_max_charge_kw=0.8,
        battery_max_discharge_kw=0.8,
        battery_reserve_ratio=0.4,
        output_csv_path="data/daily_sim_cloud_shock.csv",
        custom_pv_function=simulate_pv_with_cloud_shock,  # <--- IMPORTANT
    )

    print("Saved: data/daily_sim_cloud_shock.csv")


if __name__ == "__main__":
    run_cloud_shock_scenario()
