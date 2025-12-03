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


# ---------- Solar & Load Models (Simple but realistic) ----------

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
    # Smooth bell curve between 0 and peak_kw
    pv = peak_kw * math.sin(x)
    return max(pv, 0.0)


def simulate_room_base_loads(minute_of_day: int):
    """
    Base (non-spike) loads [kW] for bedroom, hall, kitchen.
    Rough occupancy pattern:
    - Morning (6–9): some usage
    - Daytime (9–18): moderate
    - Evening (18–23): higher
    - Night (23–6): very low
    """
    hour = minute_of_day // 60

    # default very low
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
    # Convert minute-of-day index to hour, minute
    hour = (minute_index // 60) % 24
    minute = minute_index % 60

    # Helper for time window
    def in_window(h, m_start, m_end):
        return hour == h and m_start <= minute <= m_end

    # Kitchen mixer windows
    if in_window(7, 10, 14) or in_window(19, 10, 14):
        kitchen_kw = 1.2

    # Bedroom iron windows
    if in_window(8, 30, 34) or in_window(21, 0, 4):
        bedroom_kw = 1.2

    # Hall hairdryer windows
    if in_window(9, 0, 4) or in_window(20, 30, 34):
        hall_kw = 1.4

    return bedroom_kw, hall_kw, kitchen_kw


# ---------- Simulation Runner ----------

def run_daily_simulation(
    date: datetime,
    step_minutes: int = 1,
    pv_peak_kw: float = 1.0,
    output_csv_path: str = "data/daily_sim_result.csv",
):
    """
    Run a 24-hour simulation at given resolution (step_minutes).

    Logs per-step:
    - timestamp
    - pv_power_kw
    - per-room loads (bedroom, hall, kitchen)
    - per-room source (SOLAR/GRID)
    - grid-only heavy loads (wm, pump, geyser)
    """

    # Each step is step_minutes; for controller we convert to seconds
    step_seconds = step_minutes * 60

    controller = SwitchingController(
        min_hold_steps=2,        # 2-minute hysteresis
        step_seconds=step_seconds,
        cooldown_seconds=30,     # 30s cooldown after spikes
        solar_margin=0.15,       # 15% headroom for a 1 kW system
    )

    total_minutes = 24 * 60
    total_steps = total_minutes // step_minutes

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with open(output_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "minute_of_day",
            "pv_power_kw",
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

        print(f"Running daily simulation for {date.date()}...")
        print(f"Step size: {step_minutes} minute(s)")
        print(f"Output CSV: {output_csv_path}\n")

        current_time = datetime(date.year, date.month, date.day, 0, 0, 0)

        for step in range(total_steps):
            minute_of_day = step * step_minutes

            # 1) PV power for this minute
            pv_power_kw = simulate_pv_power_kw(minute_of_day, peak_kw=pv_peak_kw)

            # 2) Base room loads
            bedroom_kw, hall_kw, kitchen_kw = simulate_room_base_loads(minute_of_day)

            # 3) Inject spikes (mixer, iron, hairdryer)
            bedroom_kw, hall_kw, kitchen_kw = add_spikes(
                minute_of_day, bedroom_kw, hall_kw, kitchen_kw
            )

            # 4) Heavy loads (always grid-side, you can customize)
            # Here we just simulate some occasional usage pattern.
            hour = minute_of_day // 60
            wm_kw = 0.0
            pump_kw = 0.0
            geyser_kw = 0.0

            # Example: washing machine runs 2 cycles a day ~1 kW
            if (hour == 10 and 0 <= minute_of_day % 60 <= 29) or \
               (hour == 16 and 0 <= minute_of_day % 60 <= 29):
                wm_kw = 1.0

            # Example: motor pump runs 15 min in morning & evening
            if (hour == 6 and 0 <= minute_of_day % 60 <= 14) or \
               (hour == 19 and 0 <= minute_of_day % 60 <= 14):
                pump_kw = 0.8

            # Example: geyser runs 30 min in morning & night
            if (hour == 7 and 0 <= minute_of_day % 60 <= 29) or \
               (hour == 21 and 0 <= minute_of_day % 60 <= 29):
                geyser_kw = 1.5

            # 5) Build loads list for controller
            loads = [
                Load(
                    "bedroom_circuit",
                    power_kw=bedroom_kw,
                    can_run_on_solar=True,
                    high_event_threshold_kw=0.8,  # iron-like spike
                ),
                Load(
                    "hall_circuit",
                    power_kw=hall_kw,
                    can_run_on_solar=True,
                    high_event_threshold_kw=0.8,  # hairdryer-like spike
                ),
                Load(
                    "kitchen_circuit",
                    power_kw=kitchen_kw,
                    can_run_on_solar=True,
                    high_event_threshold_kw=0.8,  # mixer-like spike
                ),

                # Always-grid heavy circuits:
                Load("washing_machine", power_kw=wm_kw, can_run_on_solar=False),
                Load("motor_pump", power_kw=pump_kw, can_run_on_solar=False),
                Load("geyser", power_kw=geyser_kw, can_run_on_solar=False),
            ]

            allocation = controller.decide(pv_power_kw, loads)

            # 6) Log row
            ts_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            row = [
                ts_str,
                minute_of_day,
                round(pv_power_kw, 3),
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
            ]
            writer.writerow(row)

            # 7) Step time forward
            current_time += timedelta(minutes=step_minutes)

        print("Simulation complete.")


if __name__ == "__main__":
    # Run for 'today' (date part only)
    run_daily_simulation(datetime.now())
