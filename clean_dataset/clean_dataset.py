import pandas as pd

RAW_FILE = "household_power_consumption.txt"
CLEAN_FILE = "household_power_consumption_clean.txt"
MAX_GAP_MINUTES = 60
MIN_VALID_MINUTES = 30

raw = pd.read_csv(RAW_FILE, sep=";", na_values="?", low_memory=False)
print(f"Raw minute readings          : {len(raw):,}")

raw["datetime"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
df = raw.drop(columns=["Date", "Time"]).set_index("datetime").astype(float).sort_index()

duplicates = int(df.index.duplicated().sum())
df = df[~df.index.duplicated()]
print(f"Duplicate timestamps removed : {duplicates}")

negatives = int((df["Global_active_power"] < 0).sum())
print(f"Negative power values        : {negatives}")

df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="min"))
missing_before = int(df["Global_active_power"].isna().sum())
print(f"Missing readings             : {missing_before:,} ({missing_before / len(df) * 100:.2f}%)")

df = df.interpolate(limit=MAX_GAP_MINUTES)
missing_after = int(df["Global_active_power"].isna().sum())
print(f"Still missing after filling  : {missing_after:,} (gaps longer than {MAX_GAP_MINUTES} minutes)")

hourly = pd.DataFrame({
    "consumption_kwh": df["Global_active_power"].resample("h").sum(min_count=MIN_VALID_MINUTES) / 60,
    "kitchen_kwh": df["Sub_metering_1"].resample("h").sum(min_count=MIN_VALID_MINUTES) / 1000,
    "laundry_kwh": df["Sub_metering_2"].resample("h").sum(min_count=MIN_VALID_MINUTES) / 1000,
    "water_heater_ac_kwh": df["Sub_metering_3"].resample("h").sum(min_count=MIN_VALID_MINUTES) / 1000,
    "average_voltage": df["Voltage"].resample("h").mean(),
})
total_hours = len(hourly)

clean = hourly.dropna().round(4)
clean.index.name = "datetime"
print(f"Hours before removal         : {total_hours:,}")
print(f"Incomplete hours removed     : {total_hours - len(clean):,}")
print(f"Complete hourly records      : {len(clean):,}")

clean.to_csv(CLEAN_FILE, sep=";")
print(f"\nSaved: {CLEAN_FILE}")
print(clean.describe().round(3))
