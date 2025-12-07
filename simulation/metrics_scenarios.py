import os
import pandas as pd


def analyse_switches_and_time(
    csv_path: str,
    step_minutes: int = 1,
    circuits=("bedroom", "hall", "kitchen"),
    spike_threshold_kw: float = 1.0,
):
    """
    Compute, for each circuit:
      - total minutes on SOLAR / GRID
      - % of day on SOLAR
      - number of source switches (SOLAR<->GRID)
      - number of high-power 'spike' events (power >= spike_threshold_kw)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    total_steps = len(df)
    total_minutes = total_steps * step_minutes

    results = []

    for name in circuits:
        power_col = f"{name}_kw"
        source_col = f"{name}_source"

        if power_col not in df.columns or source_col not in df.columns:
            # Scenario might not have this circuit; skip
            continue

        src_series = df[source_col].astype(str)
        power_series = df[power_col].astype(float)

        # Time on solar/grid
        solar_steps = (src_series == "SOLAR").sum()
        grid_steps = (src_series == "GRID").sum()

        solar_minutes = solar_steps * step_minutes
        grid_minutes = grid_steps * step_minutes

        solar_pct = 100.0 * solar_minutes / total_minutes if total_minutes > 0 else 0.0

        # Active-only solar time (only when load > 0)
        active_mask = power_series > 0.0
        active_steps = active_mask.sum()
        active_solar_steps = ((src_series == "SOLAR") & active_mask).sum()
        active_solar_pct = (
            100.0 * active_solar_steps / active_steps if active_steps > 0 else 0.0
        )

        # Count switches (SOLAR<->GRID)
        switches = 0
        prev = None
        for s in src_series:
            if prev is not None and s != prev:
                switches += 1
            prev = s

        # Approx spike events: power >= spike_threshold_kw,
        # count only when crossing from below threshold to above
        spikes = 0
        prev_high = False
        for p in power_series:
            high = p >= spike_threshold_kw
            if high and not prev_high:
                spikes += 1
            prev_high = high

        results.append({
            "circuit": name,
            "solar_minutes": solar_minutes,
            "grid_minutes": grid_minutes,
            "solar_pct_day": solar_pct,
            "solar_pct_when_active": active_solar_pct,
            "switches": switches,
            "spike_events": spikes,
        })

    return pd.DataFrame(results)


def compare_scenarios_with_metrics():
    # Add any scenarios you’ve generated here:
    scenarios = {
        "baseline_normal": "data/daily_sim_with_battery_baseline_normal.csv",
        "cloudy_day": "data/daily_sim_with_battery_cloudy_day.csv",
        "small_batt": "data/daily_sim_with_battery_small_batt.csv",
        "big_batt": "data/daily_sim_with_battery_big_batt.csv",
        # Optional extra scenarios if present:
        "cloud_shock": "data/daily_sim_cloud_shock.csv",
        "multi_spike": "data/daily_sim_multi_spike.csv",
        "batt_emergency": "data/daily_sim_batt_emergency.csv",
        "grid_blackout": "data/daily_sim_grid_blackout.csv",
        "user_misconfig": "data/daily_sim_user_misconfig.csv",
    }

    all_rows = []

    for sc_name, path in scenarios.items():
        if not os.path.exists(path):
            continue

        df_metrics = analyse_switches_and_time(path, step_minutes=1)
        df_metrics.insert(0, "scenario", sc_name)
        all_rows.append(df_metrics)

    if not all_rows:
        print("No scenario files found. Check paths.")
        return

    full = pd.concat(all_rows, ignore_index=True)

    # Nice formatting
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)

    print("\n=== Scenario Metrics: Time on Solar, Switches, Spikes ===\n")
    print(
        full.to_string(
            index=False,
            float_format=lambda x: f"{x:0.1f}",
        )
    )

    # Optional: you could also pivot for a more compact summary later.
    print("\nColumns:")
    print("  scenario                : which scenario")
    print("  circuit                 : which room circuit")
    print("  solar_minutes           : total minutes on inverter/solar side")
    print("  grid_minutes            : total minutes on grid side")
    print("  solar_pct_day           : % of the full day on solar")
    print("  solar_pct_when_active   : % of time on solar when load > 0")
    print("  switches                : total SOLAR<->GRID transitions")
    print("  spike_events            : times power >= spike_threshold_kw (default 1.0 kW)\n")


if __name__ == "__main__":
    compare_scenarios_with_metrics()
