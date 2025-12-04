import os
import pandas as pd
import matplotlib.pyplot as plt


def minutes_to_hours(series):
    return series / 60.0


def plot_pv_and_total_load(csv_path="data/daily_sim_result.csv"):
    df = pd.read_csv(csv_path)
    df["total_load_kw"] = (
        df["bedroom_kw"]
        + df["hall_kw"]
        + df["kitchen_kw"]
        + df["washing_machine_kw"]
        + df["motor_pump_kw"]
        + df["geyser_kw"]
    )

    hours = minutes_to_hours(df["minute_of_day"])

    plt.figure(figsize=(12, 4))
    plt.plot(hours, df["pv_power_kw"], label="PV Power (kW)")
    plt.plot(hours, df["total_load_kw"], label="Total Load (kW)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Power (kW)")
    plt.title("PV vs Total Load (Daily)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_room_loads(csv_path="data/daily_sim_result.csv"):
    df = pd.read_csv(csv_path)
    hours = minutes_to_hours(df["minute_of_day"])

    plt.figure(figsize=(12, 4))
    plt.plot(hours, df["bedroom_kw"], label="Bedroom")
    plt.plot(hours, df["hall_kw"], label="Hall")
    plt.plot(hours, df["kitchen_kw"], label="Kitchen")
    plt.xlabel("Hour of Day")
    plt.ylabel("Power (kW)")
    plt.title("Per-Room Load Timeline")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_switching(csv_path="data/daily_sim_result.csv"):
    df = pd.read_csv(csv_path)
    hours = minutes_to_hours(df["minute_of_day"])

    def encode(src):
        return 1 if src == "SOLAR" else 0

    df["bedroom_flag"] = df["bedroom_source"].apply(encode)
    df["hall_flag"] = df["hall_source"].apply(encode)
    df["kitchen_flag"] = df["kitchen_source"].apply(encode)

    plt.figure(figsize=(12, 4))
    plt.step(hours, df["bedroom_flag"], where="post", label="Bedroom (1=Solar, 0=Grid)")
    plt.step(hours, df["hall_flag"],    where="post", label="Hall (1=Solar, 0=Grid)")
    plt.step(hours, df["kitchen_flag"], where="post", label="Kitchen (1=Solar, 0=Grid)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Source")
    plt.yticks([0, 1], ["Grid", "Solar"])
    plt.title("Solar/Grid Switching Timeline")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_battery_soc(csv_path="data/daily_sim_with_battery.csv"):
    if not os.path.exists(csv_path):
        print("Battery CSV not found, skipping battery SOC plot.")
        return

    df = pd.read_csv(csv_path)
    hours = minutes_to_hours(df["minute_of_day"])

    plt.figure(figsize=(12, 4))
    plt.plot(hours, df["battery_soc_pct"])
    plt.xlabel("Hour of Day")
    plt.ylabel("Battery SOC (%)")
    plt.title("Battery State of Charge Over the Day")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if os.path.exists("data/daily_sim_result.csv"):
        print("Plotting from data/daily_sim_result.csv ...")
        plot_pv_and_total_load()
        plot_room_loads()
        plot_switching()
    else:
        print("data/daily_sim_result.csv not found – run the daily simulation first.")

    if os.path.exists("data/daily_sim_with_battery.csv"):
        print("Plotting battery SOC from data/daily_sim_with_battery.csv ...")
        plot_battery_soc()
