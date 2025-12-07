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


# ---------- Solar & Load Models (reuse_same_as_before) ----------

def simulate_pv_power_kw(minute_of_day: int, peak_kw: float = 1.0) -> float:
    """
    Simple 1 kW solar curve for a day.

    - Sunrise ~06:00 (360 min)
    - Sunset  ~18:00 (1080 min)
    - Peak at ~12:00 (720 min)
    """
    sunrise = 6 * 60
    sunset = 18 * 60

    if minute_of_day < sunrise or minute_of_day > sunset:
        return 0.0

    # Map [sunrise, sunset] -> [0, pi]
    x = (minute_of_day - sunrise) / (sunset - sunrise) * math.pi
    pv = peak_kw * math.sin(x)
    return max(pv, 0.0)


def simulate_room_base_loads(minute_of_day: int):
    """
    Base (non-spike) loads [kW] for bedroom, hall, kitchen.
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


def add_spikes(minute_index: int, bedroom_kw: float, hall_kw: float, kitchen_kw: float):
    """
    Inject spikes for:
    - Kitchen mixer: 07:10–07:14 and 19:10–19:14
    - Bedroom iron: 08:30–08:34 and 21:00–21:04
    - Hall hairdryer: 09:00–09:04 and 20:30–20:34
    """
    hour = (minute_index // 60) % 24
    minute = minute_index % 60

    def in_window(h, m_start, m_end):
        return hour == h and m_start <= minute <= m_end

    if in_window(7, 10, 14) or in_window(19, 10, 14):
        kitchen_kw = 1.2

    if in_window(8, 30, 34) or in_window(21, 0, 4):
        bedroom_kw = 1.2

    if in_window(9, 0, 4) or in_window(20, 30, 34):
        hall_kw = 1.4

    return bedroom_kw, hall_kw, kitchen_kw


# ---------- Battery Model ----------

class Battery:
    """
    Simple battery model.

    - capacity_kwh: total usable capacity [kWh]
    - soc_kwh     : current energy stored [kWh]
    - max_charge_kw / max_discharge_kw: power limits
    - eff_charge, eff_discharge: efficiencies (0-1)
    """

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
        """
        Charge the battery using up to available_kw for duration dt_hours.

        Returns actual power used for charging [kW].
        """
        if available_kw <= 0 or self.capacity_kwh <= 0:
            return 0.0

        # Limit by power and remaining capacity
        power = min(available_kw, self.max_charge_kw)
        energy_in_kwh = power * dt_hours * self.eff_charge
        room_kwh = self.capacity_kwh - self.soc_kwh
        if energy_in_kwh > room_kwh:
            # Can't take all the power for whole dt
            energy_in_kwh = room_kwh
            # Equivalent average power
            power = energy_in_kwh / (dt_hours * self.eff_charge)

        self.soc_kwh += energy_in_kwh
        return power

    def discharge(self, required_kw: float, dt_hours: float) -> float:
        """
        Discharge the battery up to required_kw for dt_hours.

        Returns actual power supplied by battery [kW].
        """
        if required_kw <= 0 or self.capacity_kwh <= 0:
            return 0.0

        power = min(required_kw, self.max_discharge_kw)
        # Energy that must be drawn from the battery when considering efficiency
        energy_out_kwh = power * dt_hours / self.eff_discharge
        if energy_out_kwh > self.soc_kwh:
            # Limited by available energy
            energy_out_kwh = self.soc_kwh
            power = energy_out_kwh * self.eff_discharge / dt_hours

        self.soc_kwh -= energy_out_kwh
        return power


# ---------- Simulation Runner with Battery ----------

def run_daily_sim_with_battery(
    date,
    step_minutes=1,
    pv_peak_kw=1.0,
    battery_capacity_kwh=2.0,
    battery_initial_soc_ratio=0.5,
    battery_max_charge_kw=0.8,
    battery_max_discharge_kw=0.8,
    battery_reserve_ratio=0.4,
    custom_pv_function=None,
    output_csv_path="data/daily_sim_with_battery.csv",
):
    
    """
    24-hour simulation with battery-aware energy flows.

    - Controller decides which loads are on inverter side (PV+Battery) vs GRID-only.
    - Battery decides how surplus/deficit on inverter side is handled.
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

    from datetime import datetime, timedelta
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

        print(f"Running daily simulation WITH battery for {date.date()}...")
        print(f"Battery capacity       : {battery_capacity_kwh} kWh")
        print(f"Initial SOC            : {battery_initial_soc_ratio*100:.1f} %")
        print(f"Max charge/discharge   : {battery_max_charge_kw} / {battery_max_discharge_kw} kW")
        print(f"Step size              : {step_minutes} minute(s)")
        print(f"Output CSV             : {output_csv_path}\n")

        for step in range(total_steps):
            minute_of_day = step * step_minutes

            # 1) PV and base loads
            pv_power_kw = simulate_pv_power_kw(minute_of_day, peak_kw=pv_peak_kw)
            bedroom_kw, hall_kw, kitchen_kw = simulate_room_base_loads(minute_of_day)
            bedroom_kw, hall_kw, kitchen_kw = add_spikes(
                minute_of_day, bedroom_kw, hall_kw, kitchen_kw
            )

            # Heavy loads (same pattern as before)
            hour = minute_of_day // 60
            wm_kw = 0.0
            pump_kw = 0.0
            geyser_kw = 0.0

            if (hour == 10 and 0 <= minute_of_day % 60 <= 29) or \
               (hour == 16 and 0 <= minute_of_day % 60 <= 29):
                wm_kw = 1.0

            if (hour == 6 and 0 <= minute_of_day % 60 <= 14) or \
               (hour == 19 and 0 <= minute_of_day % 60 <= 14):
                pump_kw = 0.8

            if (hour == 7 and 0 <= minute_of_day % 60 <= 29) or \
               (hour == 21 and 0 <= minute_of_day % 60 <= 29):
                geyser_kw = 1.5

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

                        # --- Override behavior at night: use battery until reserve, then grid ---

            # If no solar and battery is above reserve → force inverter-side loads on SOLAR
            if pv_power_kw == 0.0:
                if battery.soc_ratio > battery_reserve_ratio:
                    # Use battery for all solar-eligible circuits
                    for load in loads:
                        if load.can_run_on_solar:
                            allocation[load.name] = "SOLAR"
                else:
                    # Battery at/below reserve -> preserve it, keep circuits on GRID
                    for load in loads:
                        if load.can_run_on_solar:
                            allocation[load.name] = "GRID"


            # --- Energy flow on inverter side (PV + battery) ---

            # Total load that the controller has put on inverter side (SOLAR)
            inverter_side_load_kw = 0.0
            grid_side_load_kw = 0.0

            for load in loads:
                src = allocation[load.name]
                if src == "SOLAR":
                    inverter_side_load_kw += load.power_kw
                else:
                    grid_side_load_kw += load.power_kw

            # PV first supplies inverter-side load
            pv_to_load_kw = min(pv_power_kw, inverter_side_load_kw)
            remaining_pv_kw = max(pv_power_kw - pv_to_load_kw, 0.0)

            # If PV < inverter load: try to discharge battery to support
            deficit_kw = max(inverter_side_load_kw - pv_to_load_kw, 0.0)
            battery_discharge_kw = 0.0
            grid_support_kw = 0.0

            if deficit_kw > 0:
                battery_discharge_kw = battery.discharge(deficit_kw, step_hours)
                # If still deficit after battery, grid must support inverter side
                grid_support_kw = max(deficit_kw - battery_discharge_kw, 0.0)

            # If PV > inverter load: try to charge battery with surplus
            battery_charge_kw = 0.0
            pv_to_battery_kw = 0.0
            pv_wasted_kw = 0.0

            if remaining_pv_kw > 0:
                battery_charge_kw = battery.charge(remaining_pv_kw, step_hours)
                pv_to_battery_kw = battery_charge_kw
                pv_wasted_kw = max(remaining_pv_kw - battery_charge_kw, 0.0)

            # Total grid power going directly to grid-side loads + inverter support
            grid_to_load_kw = grid_side_load_kw + grid_support_kw

            # Log row
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

        print("Simulation with battery complete.")


if __name__ == "__main__":
    from datetime import datetime
    run_daily_sim_with_battery(datetime.now())
