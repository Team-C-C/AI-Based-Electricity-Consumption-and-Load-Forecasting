import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from common import load_hourly

series = load_hourly()["consumption_kwh"].asfreq("h")

last_weeks = series.iloc[-24 * 28:].interpolate()
decomp = seasonal_decompose(last_weeks, model="additive", period=24)
decomp.plot()
plt.tight_layout()
plt.savefig("time_series_decomposition.png", dpi=150)
plt.close()
print("Saved time_series_decomposition.png")

kwh = series.dropna()
profile = kwh.groupby([kwh.index.dayofweek >= 5, kwh.index.hour]).mean()
monthly = kwh.resample("D").sum(min_count=20).resample("MS").mean()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(profile.loc[False].index, profile.loc[False].values, label="Weekday")
ax[0].plot(profile.loc[True].index, profile.loc[True].values, label="Weekend")
ax[0].set_xlabel("Hour of day")
ax[0].set_ylabel("Average consumption (kWh per hour)")
ax[0].set_title("Average daily load profile")
ax[0].legend()
ax[1].plot(monthly.index, monthly.values)
ax[1].set_ylabel("Average kWh per day")
ax[1].set_title("Monthly consumption trend")
plt.tight_layout()
plt.savefig("load_profile.png", dpi=150)
plt.close()
print("Saved load_profile.png")
for label, grp in (("Weekday", False), ("Weekend", True)):
    top = profile.loc[grp].sort_values(ascending=False).head(2)
    print(f"{label} busiest hours: " + ", ".join(f"{h:02d}:00 ({v:.2f} kWh)" for h, v in top.items()))

for lag in (1, 24, 168):
    print(f"Autocorrelation at lag {lag:>3} h = {series.autocorr(lag):.2f}")
