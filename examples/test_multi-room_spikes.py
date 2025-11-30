import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from logic.switching import Load , SwitchingController

def simulate_room_powers(step: int):
    """
    assuming the following spikes for data 

    - bedroom spike (iron box ): in steps 10-13 -> ~1.2 kW
    - Hall spike (hairdryer): in steps 15-17 -> ~1.4 kW
    - kitchen spike (mixer): in steps 5-8 -> ~1.2kW
    - otherwise the base loads will be in 
    """

    # base loads in kW
    bedroom_kw = 0.15
    hall_kw = 0.20
    kitchen_kw = 0.20

    #adding kitchen mixer spike 
    if 5 <= step <= 8:
        kitchen_kw =1.2 


    #adding bedroom mixer spike 
    if 10 <= step <= 12:
        kitchen_kw =1.2 


    #adding hall mixer spike 
    if 15 <= step <= 17:
        kitchen_kw =1.2 

    return bedroom_kw, hall_kw, kitchen_kw


def main():
     # 1 kW solar system, step = 5 seconds, cooldown = 30 seconds
    controller = SwitchingController(
        min_hold_steps=1,      # rely mainly on cooldown for spikes
        step_seconds=5,        # 1 step = 5 seconds
        cooldown_seconds=30,   # stay on GRID for 30 seconds after a spike
        solar_margin=0.15,     # keep 15% margin on solar
    )

    
    total_steps = 25          # 25 * 5s = 125 seconds (~2 minutes)
    pv_power_kw = 1.0         # assume ~1 kW PV available

    print("\nMulti-room spike simulation on 1 kW solar system:\n")
    print("Spikes:")
    print("  Kitchen  : steps  5-8  (mixer)")
    print("  Bedroom  : steps 10-13 (iron)")
    print("  Hall     : steps 15-17 (hairdryer)")
    print("\nEach step = 5 seconds. Cooldown = 30 seconds.\n")

    for step in range(total_steps):
        bedroom_kw, hall_kw, kitchen_kw = simulate_room_powers(step)

        loads = [
            Load(
                "bedroom_circuit",
                power_kw=bedroom_kw,
                can_run_on_solar=True,
                high_event_threshold_kw=0.8,  # iron spike
            ),
            Load(
                "hall_circuit",
                power_kw=hall_kw,
                can_run_on_solar=True,
                high_event_threshold_kw=0.8,  # hairdryer spike
            ),
            Load(
                "kitchen_circuit",
                power_kw=kitchen_kw,
                can_run_on_solar=True,
                high_event_threshold_kw=0.8,  # mixer spike
            ),

            # Always-grid heavy circuits (off for this test, but included)
            Load("washing_machine", power_kw=0.0, can_run_on_solar=False),
            Load("motor_pump", power_kw=0.0, can_run_on_solar=False),
        ]

        allocation = controller.decide(pv_power_kw, loads)

        print(
            f"Step {step:02d} | "
            f"PV: {pv_power_kw:.2f} kW | "
            f"Bedroom: {bedroom_kw:.2f} kW | "
            f"Hall: {hall_kw:.2f} kW | "
            f"Kitchen: {kitchen_kw:.2f} kW"
        )

        for load in loads:
            src = allocation[load.name]
            marker = ""
            if load.name == "bedroom_circuit":
                marker = "  <-- BEDROOM"
            elif load.name == "hall_circuit":
                marker = "  <-- HALL"
            elif load.name == "kitchen_circuit":
                marker = "  <-- KITCHEN"

            print(f"  {load.name:16s} -> {src}{marker}")
        print("-" * 70)


if __name__ == "__main__":
    main()