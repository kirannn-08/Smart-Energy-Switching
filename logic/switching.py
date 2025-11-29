"""
Switching logic for Smart Energy System.

Key ideas:
- No manual priorities: priority is derived from load size & type.
- Base rule: small, solar-eligible loads are preferred on SOLAR.
- High-power or spiky loads can be forced to GRID for safety & stability.
- Hysteresis: avoid rapid flipping between SOLAR and GRID.
"""

from typing import List, Dict, Literal, Optional

Source = Literal["SOLAR", "GRID"]


class Load:
    """
    Represents a single load (circuit / room) at a given time step.

    Attributes:
        name: identifier of the load (e.g., "kitchen_circuit")
        power_kw: current power consumption of this load [kW]
        can_run_on_solar: False if this load must stay on grid only (e.g., geyser)
        is_high_power: True if typical high-power device (motor, etc.)
        high_event_threshold_kw:
            If not None:
                - when power_kw >= high_event_threshold_kw,
                  this is treated as a spike event (e.g., mixer / iron / hairdryer)
                  and the circuit will be forced to GRID for some cooldown period.
    """

    def __init__(
        self,
        name: str,
        power_kw: float,
        can_run_on_solar: bool = True,
        is_high_power: bool = False,
        high_event_threshold_kw: Optional[float] = None,
    ) -> None:
        self.name = name
        self.power_kw = float(power_kw)
        self.can_run_on_solar = bool(can_run_on_solar)
        self.is_high_power = bool(is_high_power)
        self.high_event_threshold_kw = high_event_threshold_kw

    def __repr__(self) -> str:
        return (
            f"Load(name={self.name!r}, power_kw={self.power_kw:.2f}, "
            f"can_run_on_solar={self.can_run_on_solar}, "
            f"is_high_power={self.is_high_power}, "
            f"high_event_threshold_kw={self.high_event_threshold_kw})"
        )


def allocate_sources_for_step(
    pv_power_kw: float,
    loads: List[Load],
    solar_margin: float = 0.15,
    high_power_solar_factor: float = 0.6,
    high_power_min_kw: float = 0.8,
) -> Dict[str, Source]:
    """
    Stateless allocation: decide which loads run on SOLAR vs GRID for a single time step.

    Algorithm:
    - All grid-only loads -> GRID.
    - Among solar-eligible loads:
        * smaller loads (kW) and non-high-power loads are preferred for SOLAR.
        * high-power loads are used on SOLAR only if enough capacity remains.

    Note: spike handling (mixer / iron etc.) is done in SwitchingController,
          this function just does basic solar allocation.
    """
    allocation: Dict[str, Source] = {}

    # Step 1: if no solar, everything goes to grid
    if pv_power_kw <= 0:
        for load in loads:
            allocation[load.name] = "GRID"
        return allocation

    # Usable solar after keeping some margin (for a 1 kW system, be conservative)
    pv_remaining = max(pv_power_kw * (1.0 - solar_margin), 0.0)

    # Step 2: handle grid-only loads immediately
    solar_eligible: List[Load] = []
    for load in loads:
        if not load.can_run_on_solar:
            allocation[load.name] = "GRID"
        else:
            solar_eligible.append(load)

    # Step 3: sort solar-eligible loads by algorithmic "priority"
    def priority_key(l: Load):
        effective_high_power = l.is_high_power or (l.power_kw >= high_power_min_kw)
        return (
            effective_high_power,  # False (0) before True (1)
            l.power_kw,            # then lower power first
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


class SwitchingController:
    """
    Stateful controller that wraps allocate_sources_for_step and
    adds:
    - hysteresis (anti-chattering)
    - spike detection + cooldown (for mixer / iron / hairdryer behavior)

    For spike behavior:
    - If a load has high_event_threshold_kw set and power >= threshold:
        * force that load to GRID immediately
        * keep it on GRID for cooldown_seconds (even after it drops)
    """

    def __init__(
        self,
        min_hold_steps: int = 3,       # steps before switching allowed (for hysteresis)
        step_seconds: int = 5,         # one decision step = 5 seconds (example)
        cooldown_seconds: int = 30,    # after spike, stay on GRID this long
        solar_margin: float = 0.15,
        high_power_solar_factor: float = 0.6,
        high_power_min_kw: float = 0.8,
    ) -> None:
        self.min_hold_steps = int(min_hold_steps)
        self.step_seconds = int(step_seconds)
        self.cooldown_seconds = int(cooldown_seconds)
        self.cooldown_steps = max(self.cooldown_seconds // self.step_seconds, 1)

        self.solar_margin = solar_margin
        self.high_power_solar_factor = high_power_solar_factor
        self.high_power_min_kw = high_power_min_kw

        # internal state:
        # name -> {"source": Source, "steps": int}
        self._state: Dict[str, Dict[str, Optional[Source]]] = {}
        # spike/cooldown tracking: name -> step index until which GRID is forced
        self._force_grid_until_step: Dict[str, int] = {}
        self._step_index: int = 0

    def _update_state(self, name: str, new_source: Source) -> None:
        entry = self._state.get(name)
        if entry is None:
            self._state[name] = {"source": new_source, "steps": 1}
        else:
            if entry["source"] == new_source:
                entry["steps"] += 1
            else:
                entry["source"] = new_source
                entry["steps"] = 1

    def _check_and_update_spikes(self, loads: List[Load]) -> None:
        """
        For each load with a high_event_threshold_kw:
        - If power >= threshold:
            * mark this load as forced to GRID for cooldown_steps.
        """
        for load in loads:
            if load.high_event_threshold_kw is None:
                continue
            if load.power_kw >= load.high_event_threshold_kw:
                # Spike detected: force GRID from now until cooldown expires
                self._force_grid_until_step[load.name] = self._step_index + self.cooldown_steps

    def decide(
        self,
        pv_power_kw: float,
        loads: List[Load],
    ) -> Dict[str, Source]:
        """
        Decide sources with hysteresis and spike handling, for one time step.

        Args:
            pv_power_kw: available solar power at this time step.
            loads: current loads with their power values.

        Returns:
            allocation dict name -> "SOLAR" | "GRID"
        """
        # Step index advances each call
        self._step_index += 1

        # First handle spike detection
        self._check_and_update_spikes(loads)

        # Base stateless suggestion (without spike logic)
        suggested = allocate_sources_for_step(
            pv_power_kw=pv_power_kw,
            loads=loads,
            solar_margin=self.solar_margin,
            high_power_solar_factor=self.high_power_solar_factor,
            high_power_min_kw=self.high_power_min_kw,
        )

        final_allocation: Dict[str, Source] = {}

        # Apply spike rules + hysteresis
        for load in loads:
            name = load.name
            suggested_src = suggested[name]

            # Check if this load is currently under forced-grid window
            force_until = self._force_grid_until_step.get(name)
            if force_until is not None and self._step_index <= force_until:
                forced_src: Source = "GRID"
                final_src = forced_src
            else:
                # No active spike cooldown → normal hysteresis
                prev_entry = self._state.get(name)
                if prev_entry is None:
                    final_src = suggested_src
                else:
                    prev_src: Source = prev_entry["source"]  # type: ignore
                    steps = prev_entry["steps"]

                    if prev_src == suggested_src:
                        final_src = suggested_src
                    else:
                        # suggestion wants to switch
                        if steps >= self.min_hold_steps:
                            final_src = suggested_src
                        else:
                            final_src = prev_src

            final_allocation[name] = final_src
            self._update_state(name, final_src)

        return final_allocation
