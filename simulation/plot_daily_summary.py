import os
import pandas as pd
import matplotlib.pyplot as plt


def minutes_to_hours(series):
    return series / 60.0


def plot_daily_summary(
    no_battery_csv="data/daily_sim_result.csv",
    battery_csv="data/daily_sim_with_battery.csv",
):
    # --- Load without battery ---
    if not os.path.exists(no_battery_csv):
        print("Missing daily_sim_result.csv")
        return

    df = pd.read_csv(no_battery_csv)
    df["hour"] = minutes_to_hours(df["minute_of_day"])
    df["total_load_kw"] = (
        df["bedroom_kw"]
        + df["hall_kw"]
        + df["kitchen_kw"]
        + df["washing_machine_kw"]
        + df["motor_pump_kw"]
        + df["geyser_kw"]
    )

    # --- Load battery file if exists ---
    df_batt = None
    if os.path.exists(battery_csv):
        df_batt = pd.read_csv(battery_csv)
        df_batt["hour"] = minutes_to_hours(df_batt["minute_of_day"])

    # ------------ Create Figure with Subplots -----------
    fig, axs = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    # --- Plot 1: PV vs Total Load ---
    axs[0].plot(df["hour"], df["pv_power_kw"], label="PV Power (kW)")
    axs[0].plot(df["hour"], df["total_load_kw"], label="Total Load (kW)")
    axs[0].set_title("PV vs Total Load")
    axs[0].set_ylabel("Power (kW)")
    axs[0].legend()
    axs[0].grid(True)

    # --- Plot 2: Per-Room Loads ---
    axs[1].plot(df["hour"], df["bedroom_kw"], label="Bedroom")
    axs[1].plot(df["hour"], df["hall_kw"], label="Hall")
    axs[1].plot(df["hour"], df["kitchen_kw"], label="Kitchen")
    axs[1].set_title("Per-Room Load Timeline")
    axs[1].set_ylabel("Power (kW)")
    axs[1].legend()
    axs[1].grid(True)

    # --- Plot 3: Switching (Solar=1, Grid=0) ---
    def encode(src):
        return 1 if src == "SOLAR" else 0

    df["bed_flag"] = df["bedroom_source"].apply(encode)
    df["hall_flag"] = df["hall_source"].apply(encode)
    df["kit_flag"] = df["kitchen_source"].apply(encode)

    axs[2].step(df["hour"], df["bed_flag"], where="post", label="Bedroom")
    axs[2].step(df["hour"], df["hall_flag"], where="post", label="Hall")
    axs[2].step(df["hour"], df["kit_flag"], where="post", label="Kitchen")
    axs[2].set_title("Solar/Grid Switching (1=Solar, 0=Grid)")
    axs[2].set_ylabel("Source")
    axs[2].set_yticks([0, 1])
    axs[2].legend()
    axs[2].grid(True)

    # --- Plot 4: Battery SOC ---
    if df_batt is not None:
        axs[3].plot(df_batt["hour"], df_batt["battery_soc_pct"], label="Battery SOC (%)")
        axs[3].set_ylabel("SOC (%)")
        axs[3].set_title("Battery State of Charge")
        axs[3].set_ylim(0, 100)
        axs[3].grid(True)
    else:
        axs[3].text(0.3, 0.5, "Battery CSV not found", fontsize=14)
        axs[3].set_axis_off()

    # Shared X label
    axs[3].set_xlabel("Hour of Day")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_daily_summary()
