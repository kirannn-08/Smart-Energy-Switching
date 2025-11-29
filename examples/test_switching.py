"""
Quick test script for the switching logic.

Run with:
    python3 examples/test_switching.py
from the project root.
"""

import os
import sys

# --- Add project root to Python path ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now this import will work
from logic.switching import Load, allocate_sources_for_step


def main():
    pv_power_kw = 2.5  # available solar power now

    loads = [
        Load("room1_lights_fan", power_kw=0.2, can_run_on_solar=True, is_high_power=False),
        Load("room2_pc_fan", power_kw=0.4, can_run_on_solar=True, is_high_power=False),
        Load("washing_machine", power_kw=1.2, can_run_on_solar=True, is_high_power=True),
        Load("motor_pump", power_kw=1.5, can_run_on_solar=True, is_high_power=True),
        Load("router_backup", power_kw=0.05, can_run_on_solar=False, is_high_power=False),  # must be on grid
    ]

    allocation = allocate_sources_for_step(pv_power_kw, loads)

    print(f"Available PV power: {pv_power_kw:.2f} kW\n")
    print("Loads:")
    for load in loads:
        print(f"  {load}")

    print("\nAllocation:")
    for name, src in allocation.items():
        print(f"  {name:20s} -> {src}")


if __name__ == "__main__":
    main()
