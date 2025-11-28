"""
This file briefs about the switching logic for the 
Smart energy system 

skeletal code for test / updates will be further made

This current logic decides priority automatically based on:
- Load side (power_kw): how loads are more 'solar-friendly'
- Whether the load can run on solar or not
- Whether the load contains any high power appliances

"""

from typing import List, Dict, Literal

Source = Literal["SOLAR", "GRID"]


class Load:
    """
    Represents a single load (room/appliance) at a given time step.

    Attributes:
        name: string
        power_kw: float
        can_run_on_solar: bool
        is_high_power: bool
    """

    def __init__(
        self,
        name: str,
        power_kw: float,
        can_run_on_solar: bool = True,
        is_high_power: bool = False,
    ) -> None:
        self.name = name
        self.power_kw = float(power_kw)
        self.can_run_on_solar = bool(can_run_on_solar)
        self.is_high_power = bool(is_high_power)

    def __repr__(self) -> str:
        return (
            f"Load(name={self.name!r}, power_kw={self.power_kw:.2f}, "
            f"can_run_on_solar={self.can_run_on_solar}, "
            f"is_high_power={self.is_high_power})"
        )


def allocate_sources_for_step(
    pv_power_kw: float,
    loads: List[Load],
    solar_margin: float = 0.10,
    high_power_solar_factor: float = 0.6,
    high_power_min_kw: float = 0.8,
) -> Dict[str, Source]:
    """
    Decide which loads run on SOLAR vs GRID for a single time step.

    Priority is not given manually. The algorithm:
    - Puts all grid-only loads on GRID.
    - Among solar-eligible loads:
        * lower power loads are considered first
        * high-power loads are only put on solar if there is plenty of capacity

   
    """
    allocation: Dict[str, Source] = {}

    # Step 1: if no solar, everything goes to grid
    if pv_power_kw <= 0:
        for load in loads:
            allocation[load.name] = "GRID"
        return allocation

    # Usable solar after keeping some margin
    pv_remaining = max(pv_power_kw * (1.0 - solar_margin), 0.0)

    # Step 2: handle grid-only loads immediately
    solar_eligible: List[Load] = []
    for load in loads:
        if not load.can_run_on_solar:
            allocation[load.name] = "GRID"
        else:
            solar_eligible.append(load)

    # Step 3: sort solar-eligible loads by algorithmic "priority"
    # We want:
    #   - smaller loads first (so we can pack more onto solar)
    #   - high-power loads are considered later (less preferred)
    def priority_key(l: Load):
        effective_high_power = l.is_high_power or (l.power_kw >= high_power_min_kw)
        return (
            effective_high_power,  # False (0) before True (1)
            l.power_kw,            # lower power first
        )

    solar_eligible_sorted = sorted(solar_eligible, key=priority_key)

    # Step 4: greedy allocation to solar
    for load in solar_eligible_sorted:
        # No solar left → must be grid
        if pv_remaining <= 0:
            allocation[load.name] = "GRID"
            continue

        effective_high_power = load.is_high_power or (load.power_kw >= high_power_min_kw)

        if effective_high_power:
            # Allow only if clearly enough solar capacity left
            if load.power_kw <= pv_remaining * high_power_solar_factor:
                allocation[load.name] = "SOLAR"
                pv_remaining -= load.power_kw
            else:
                allocation[load.name] = "GRID"
        else:
            # Non-high-power → use solar if it fits
            if load.power_kw <= pv_remaining:
                allocation[load.name] = "SOLAR"
                pv_remaining -= load.power_kw
            else:
                allocation[load.name] = "GRID"

    return allocation
