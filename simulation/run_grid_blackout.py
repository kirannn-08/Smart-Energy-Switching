import os
import sys
import math
from datetime import datetime, timedelta
import csv

# --- Add project root to Python path ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from logic.switching import Load, SwitchingController


# ---------- Solar & Load Models ----------

def simulate_pv_power_kw(minute_of_day: int, peak_kw: float = 1.0) -> float:
    """
    Standard 1 kW solar curve:

    - Sunrise ~06:00 (360 min)
    - Sunset  ~18:00 (1080 min)
    """
    sunrise = 6 * 60
    sunset = 18 * 60

    if minute_of_day < sunrise or minute_of_day > sunset:
        return 0.0

    x = (minute_of_day - sunrise) / (sunset - sunrise) * math.pi
    pv = peak_kw * math.sin(x)
    return max(pv, 0.0)


def simulate_room_base_loads(minute_of_day: int):
    """
    Base (non-spike) loads [kW] for bedroom, hall, kitchen.
    Same pattern as baseline.
    """
    hour = minute_of_day // 60

    bedroom = 0.05
    hall = 0.05
    kitchen = 0.05

    if 6 <= hour < 9:  # morning
        bedroom = 0.08
        hall = 0.05
        kitchen = 0.15
    elif 9 <= hour < 18:  # daytime
        bedroom = 0.05
        hall = 0.08
        kitchen = 0.10
    elif 18 <= hour < 23:  # evening peak
        bedroom = 0.12
        hall = 0.15
        kitchen = 0.18
    else:  # night
        bedroom = 0.03
        hall = 0.03
        kitchen = 0.03

    return bedroom, hall, kitchen


# ---------- Battery Model (same as other sims) ----------

class Battery:
    def __init__(
        self,
        capacity_kwh: float,
        soc_initial_kwh: float,
        max_charge_kw: float,
        max_discharge_kw: float,
        eff_charge: float = 0.95,
        eff_discharge: float = 0.95,
    ) -> None:
        self.capacity_kwh = capacity_kwh
        self.soc_kwh = min(max(soc_initial_kwh, 0.0), capacity_kwh)
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.eff_charge = eff_charge
        self.eff_discharge = eff_discharge

    @property
    def soc_ratio(self) -> float:
        return self.soc_kwh / self.capacity_kwh if self.capacity_kwh > 0 else 0.0

    def charge(self, available_kw: float, dt_hours: float) -> float:
        if available_kw <= 0 or self.capacity_kwh <= 0:
            return 0.0
        power = min(available_kw, self.max_charge_kw)
        energy_in_kwh = power * dt_hours * self.eff_charge
        room_kwh = self.capacity_kwh - self.soc_kwh
        if energy_in_kwh > room_kwh:
            energy_in_kwh = room_kwh
            power = energy_in_kwh / (dt_hours * self.eff_charge)
        self.soc_kwh += energy_in_kwh
        return power

    def discharge(self, required_kw: float, dt_hours: float) -> float:
        if required_kw <= 0 or self.capacity_kwh <= 0:
            return 0.0
        power = min(required_kw, self.max_discharge_kw)
        energy_out_kwh = power * dt_hours / self.eff_discharge
        if energy_out_kwh > self.soc_kwh:
            energy_out_kwh = self.soc_kwh
            power = energy_out_kwh * self.eff_discharge / dt_hours
        self.soc_kwh -= energy_out_kwh
        return power


# ---------- Grid Blackout Scenario ----------

def run_grid_blackout_day(
    date: datetime,
    step_minutes: int = 1,
    pv_peak_kw: float = 1.0,
    battery_capacity_kwh: float = 2.0,
    battery_initial_soc_ratio: float = 0.5,
    battery_max_charge_kw: float = 0.8,
    battery_max_discharge_kw: float = 0.8,
    battery_reserve_ratio: float = 0.4,
    blackout_start_hour: int = 14,
    blackout_end_hour: int = 16,
    output_csv_path: str = "data/daily_sim_grid_blackout.csv",
):
    """
    Full-day simulation with a 2-hour grid blackout (default 14:00–16:00).

    - During blackout:
        * Grid is unavailable (grid_to_load_kw = 0).
        * Solar-eligible circuits are forced to SOLAR side (PV+Battery only).
        * Grid-only heavy loads (washing machine, pump, geyser) are disabled (0 kW).
    """

    step_hours = step_minutes / 60.0
    step_seconds = step_minutes * 60

    controller = SwitchingController(
        min_hold_steps=2,
        step_seconds=step_seconds,
        cooldown_seconds=30,
        solar_margin=0.15,
    )

    battery = Battery(
        capacity_kwh=battery_capacity_kwh,
        soc_initial_kwh=battery_capacity_kwh * battery_initial_soc_ratio,
        max_charge_kw=battery_max_charge_kw,
        max_discharge_kw=battery_max_discharge_kw,
    )

    total_minutes = 24 * 60
    total_steps = total_minutes // step_minutes

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    current_time = datetime(date.year, date.month, date.day, 0, 0, 0)

    with open(output_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "minute_of_day",
            "pv_power_kw",
            "battery_soc_kwh",
            "battery_soc_pct",
            "battery_charge_kw",
            "battery_discharge_kw",
            "pv_to_load_kw",
            "pv_to_battery_kw",
            "pv_wasted_kw",
            "grid_to_load_kw",
            "grid_available",
            "bedroom_kw",
            "hall_kw",
            "kitchen_kw",
            "washing_machine_kw",
            "motor_pump_kw",
            "geyser_kw",
            "bedroom_source",
            "hall_source",
            "kitchen_source",
            "washing_machine_source",
            "motor_pump_source",
            "geyser_source",
        ])

        print("\n=== RUNNING GRID BLACKOUT SCENARIO ===")
        print(f"Grid blackout from {blackout_start_hour:02d}:00 to {blackout_end_hour:02d}:00")
        print(f"Output CSV: {output_csv_path}\n")

        for step in range(total_steps):
            minute_of_day = step * step_minutes
            hour = minute_of_day // 60

            # Is grid available in this minute?
            blackout_active = blackout_start_hour <= hour < blackout_end_hour
            grid_available = 0 if blackout_active else 1

            # 1) PV
            pv_power_kw = simulate_pv_power_kw(minute_of_day, peak_kw=pv_peak_kw)

            # 2) Base loads
            bedroom_kw, hall_kw, kitchen_kw = simulate_room_base_loads(minute_of_day)

            # 3) Heavy loads (use same pattern as baseline, BUT turn them off during blackout)
            wm_kw = 0.0
            pump_kw = 0.0
            geyser_kw = 0.0

            if not blackout_active:
                # Only allow heavy loads when grid is present (simple assumption)
                if (hour == 10 and 0 <= minute_of_day % 60 <= 29) or \
                   (hour == 16 and 0 <= minute_of_day % 60 <= 29):
                    wm_kw = 1.0

                if (hour == 6 and 0 <= minute_of_day % 60 <= 14) or \
                   (hour == 19 and 0 <= minute_of_day % 60 <= 14):
                    pump_kw = 0.8

                if (hour == 7 and 0 <= minute_of_day % 60 <= 29) or \
                   (hour == 21 and 0 <= minute_of_day % 60 <= 29):
                    geyser_kw = 1.5

            # 4) Build loads
            loads = [
                Load(
                    "bedroom_circuit",
                    power_kw=bedroom_kw,
                    can_run_on_solar=True,
                    high_event_threshold_kw=0.8,
                ),
                Load(
                    "hall_circuit",
                    power_kw=hall_kw,
                    can_run_on_solar=True,
                    high_event_threshold_kw=0.8,
                ),
                Load(
                    "kitchen_circuit",
                    power_kw=kitchen_kw,
                    can_run_on_solar=True,
                    high_event_threshold_kw=0.8,
                ),
                Load("washing_machine", power_kw=wm_kw, can_run_on_solar=False),
                Load("motor_pump", power_kw=pump_kw, can_run_on_solar=False),
                Load("geyser", power_kw=geyser_kw, can_run_on_solar=False),
            ]

            allocation = controller.decide(pv_power_kw, loads)

            # 5) Nighttime reserve rule (same logic as earlier sims)
            if pv_power_kw == 0.0:
                if battery.soc_ratio > battery_reserve_ratio:
                    for load in loads:
                        if load.can_run_on_solar:
                            allocation[load.name] = "SOLAR"
                else:
                    for load in loads:
                        if load.can_run_on_solar:
                            allocation[load.name] = "GRID"

            # 6) Grid blackout override:
            if blackout_active:
                # Grid is down: all solar-eligible circuits must be on inverter side.
                for load in loads:
                    if load.can_run_on_solar:
                        allocation[load.name] = "SOLAR"
                # Heavy loads are already 0 kW, so leaving them on GRID is fine.

            # 7) Energy flows: PV + Battery + (maybe) Grid
            inverter_side_load_kw = 0.0
            grid_side_load_kw = 0.0
            for load in loads:
                src = allocation[load.name]
                if src == "SOLAR":
                    inverter_side_load_kw += load.power_kw
                else:
                    grid_side_load_kw += load.power_kw

            # PV to loads on inverter side
            pv_to_load_kw = min(pv_power_kw, inverter_side_load_kw)
            remaining_pv_kw = max(pv_power_kw - pv_to_load_kw, 0.0)

            # Deficit on inverter side handled by battery, NO grid support
            deficit_kw = max(inverter_side_load_kw - pv_to_load_kw, 0.0)
            battery_discharge_kw = 0.0
            grid_support_kw = 0.0

            if deficit_kw > 0:
                battery_discharge_kw = battery.discharge(deficit_kw, step_hours)
                # If still deficit after battery, we simply cannot serve that extra load.
                # We do NOT use grid during blackout.

            # Surplus PV charges battery
            battery_charge_kw = 0.0
            pv_to_battery_kw = 0.0
            pv_wasted_kw = 0.0

            if remaining_pv_kw > 0:
                battery_charge_kw = battery.charge(remaining_pv_kw, step_hours)
                pv_to_battery_kw = battery_charge_kw
                pv_wasted_kw = max(remaining_pv_kw - battery_charge_kw, 0.0)

            # Grid supplies only grid-side loads when available (no support to inverter side here)
            grid_to_load_kw = 0.0
            if grid_available:
                grid_to_load_kw = grid_side_load_kw  # no grid_support_kw in this model

            ts_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([
                ts_str,
                minute_of_day,
                round(pv_power_kw, 3),
                round(battery.soc_kwh, 3),
                round(battery.soc_ratio * 100.0, 1),
                round(battery_charge_kw, 3),
                round(battery_discharge_kw, 3),
                round(pv_to_load_kw, 3),
                round(pv_to_battery_kw, 3),
                round(pv_wasted_kw, 3),
                round(grid_to_load_kw, 3),
                grid_available,
                round(bedroom_kw, 3),
                round(hall_kw, 3),
                round(kitchen_kw, 3),
                round(wm_kw, 3),
                round(pump_kw, 3),
                round(geyser_kw, 3),
                allocation["bedroom_circuit"],
                allocation["hall_circuit"],
                allocation["kitchen_circuit"],
                allocation["washing_machine"],
                allocation["motor_pump"],
                allocation["geyser"],
            ])

            current_time += timedelta(minutes=step_minutes)

        print("Grid blackout simulation complete.")


if __name__ == "__main__":
    run_grid_blackout_day(datetime.now())
