import os
import sys
from datetime import datetime

# Make sure we can import the shared simulation function
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation.run_daily_sim_with_battery import run_daily_sim_with_battery  # adjust if needed


def run_battery_emergency_scenario():
    """
    Scenario A3: Battery Emergency / Low SOC

    - Battery capacity: 2.0 kWh
    - Initial SOC: 10% (0.2 kWh)
    - Reserve: 40% (0.8 kWh)

    Expected behavior:
    - At night / when PV=0, battery is BELOW reserve -> controller should keep
      solar-eligible loads on GRID (no discharge).
    - As PV comes in during the day, battery charges.
    - Once SOC rises above reserve, normal behavior resumes.
    """

    output_csv = "data/daily_sim_batt_emergency.csv"
    print("\n=== RUNNING BATTERY EMERGENCY / LOW SOC SCENARIO ===\n")
    print(f"Output CSV: {output_csv}\n")

    run_daily_sim_with_battery(
        date=datetime.now(),
        step_minutes=1,
        pv_peak_kw=1.0,               # normal 1 kW PV
        battery_capacity_kwh=2.0,
        battery_initial_soc_ratio=0.10,  # 10% SOC start (emergency)
        battery_max_charge_kw=0.8,
        battery_max_discharge_kw=0.8,
        battery_reserve_ratio=0.40,      # 40% must be preserved
        output_csv_path=output_csv,
        # no custom PV function here -> normal clear day
    )

    print("Battery emergency simulation complete.")


if __name__ == "__main__":
    run_battery_emergency_scenario()
