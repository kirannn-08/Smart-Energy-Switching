import os
import sys

# --- Add project root to Python path ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from logic.switching import Load, SwitchingController


def simulate_kitchen_power(step: int) -> float:
    """
    Simulate kitchen circuit power [kW].

    - Most of the time: ~0.2 kW (lights, small loads)
    - For a short window (e.g., steps 5–8): ~1.2 kW (mixer ON)
    """
    if 5 <= step <= 8:
        return 1.2  # mixer ON
    else:
        return 0.2  # normal usage


def main():
    # 1 kW solar system, step = 5 seconds, cooldown = 30 seconds
    controller = SwitchingController(
        min_hold_steps=1,      # allow switching fairly quickly (we rely on cooldown for spikes)
        step_seconds=5,        # 1 decision step = 5 seconds
        cooldown_seconds=30,   # 30 second cooldown after spike
        solar_margin=0.15,     # keep 15% safety margin on PV
    )

    total_steps = 20  # 20 * 5s = 100 seconds (~1.7 minutes) just for demo
    pv_power_kw = 1.0  # assume around 1 kW PV available during test

    print("\nMixer spike simulation on 1 kW solar system:\n")
    print("Legend: KITCHEN column shows where the kitchen circuit is powered from.\n")

    for step in range(total_steps):
        # Simulate real-time measurements for each step
        kitchen_kw = simulate_kitchen_power(step)

        # Example fixed loads for other circuits (you can tweak these)
        bedroom_kw = 0.15
        hall_kw = 0.25

        # Heavy loads always on grid-only side
        wm_kw = 0.0     # washing machine OFF in this scenario
        pump_kw = 0.0   # motor pump OFF

        loads = [
            Load("bedroom_circuit", power_kw=bedroom_kw, can_run_on_solar=True),
            Load("hall_circuit", power_kw=hall_kw, can_run_on_solar=True),

            # Kitchen circuit: can run on solar, but mixer spike should force GRID
            Load(
                "kitchen_circuit",
                power_kw=kitchen_kw,
                can_run_on_solar=True,
                high_event_threshold_kw=0.8,  # >0.8 kW treated as mixer/iron/hairdryer event
            ),

            # Always-grid heavy circuits:
            Load("washing_machine", power_kw=wm_kw, can_run_on_solar=False),
            Load("motor_pump", power_kw=pump_kw, can_run_on_solar=False),
        ]

        allocation = controller.decide(pv_power_kw, loads)

        # Pretty print for this step
        print(f"Step {step:02d} | PV: {pv_power_kw:.2f} kW | "
              f"kitchen power: {kitchen_kw:.2f} kW")

        for load in loads:
            src = allocation[load.name]
            marker = ""
            if load.name == "kitchen_circuit":
                # highlight kitchen
                marker = "  <-- KITCHEN"
            print(f"  {load.name:16s} -> {src}{marker}")
        print("-" * 50)


if __name__ == "__main__":
    main()
