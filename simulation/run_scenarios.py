import os
from datetime import datetime

from run_daily_sim_with_battery import run_daily_sim_with_battery
from analyse_daily_with_battery import analyse_daily_with_battery


def run_scenarios():
    # You can adjust this date or keep 'today'
    base_date = datetime.now()

    scenarios = [
        {
            "name": "baseline_normal",
            "description": "1 kW PV, 2 kWh battery, 40% reserve (current default)",
            "pv_peak_kw": 1.0,
            "battery_capacity_kwh": 2.0,
            "battery_initial_soc_ratio": 0.5,
            "battery_reserve_ratio": 0.4,
            "battery_max_charge_kw": 0.8,
            "battery_max_discharge_kw": 0.8,
        },
        {
            "name": "cloudy_day",
            "description": "Cloudy day: PV peak limited to 0.6 kW, same battery as baseline",
            "pv_peak_kw": 0.6,
            "battery_capacity_kwh": 2.0,
            "battery_initial_soc_ratio": 0.5,
            "battery_reserve_ratio": 0.4,
            "battery_max_charge_kw": 0.6,
            "battery_max_discharge_kw": 0.8,
        },
        {
            "name": "small_batt",
            "description": "Small battery (1 kWh, 30% reserve) vs same loads",
            "pv_peak_kw": 1.0,
            "battery_capacity_kwh": 1.0,
            "battery_initial_soc_ratio": 0.5,
            "battery_reserve_ratio": 0.3,
            "battery_max_charge_kw": 0.6,
            "battery_max_discharge_kw": 0.6,
        },
        {
            "name": "big_batt",
            "description": "Big battery (4 kWh, 40% reserve) to absorb more PV",
            "pv_peak_kw": 1.0,
            "battery_capacity_kwh": 4.0,
            "battery_initial_soc_ratio": 0.5,
            "battery_reserve_ratio": 0.4,
            "battery_max_charge_kw": 1.0,
            "battery_max_discharge_kw": 1.0,
        },
    ]

    os.makedirs("data", exist_ok=True)

    for sc in scenarios:
        name = sc["name"]
        csv_path = f"data/daily_sim_with_battery_{name}.csv"

        print("\n" + "=" * 80)
        print(f"Running scenario: {name}")
        print(f"Description     : {sc['description']}")
        print(f"Output CSV      : {csv_path}")
        print("=" * 80 + "\n")

        run_daily_sim_with_battery(
            date=base_date,
            step_minutes=1,
            pv_peak_kw=sc["pv_peak_kw"],
            battery_capacity_kwh=sc["battery_capacity_kwh"],
            battery_initial_soc_ratio=sc["battery_initial_soc_ratio"],
            battery_max_charge_kw=sc["battery_max_charge_kw"],
            battery_max_discharge_kw=sc["battery_max_discharge_kw"],
            battery_reserve_ratio=sc["battery_reserve_ratio"],
            output_csv_path=csv_path,
        )

        # Analyse this scenario
        print(f"\nEnergy summary for scenario: {name}")
        analyse_daily_with_battery(csv_path=csv_path, step_minutes=1)


if __name__ == "__main__":
    run_scenarios()
