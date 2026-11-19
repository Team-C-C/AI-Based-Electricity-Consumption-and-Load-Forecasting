import argparse
import os
import joblib
import numpy as np
import pandas as pd
from common import FEATURES, MODEL_DIR, WINDOW, load_hourly

parser = argparse.ArgumentParser()
parser.add_argument("--hours", type=int, default=24, help="number of hours to forecast")
parser.add_argument("--model", choices=["rf", "lstm"], default="rf", help="rf = Random Forest")
parser.add_argument("--start", default=None, help='first hour to forecast, e.g. "2010-11-20 00:00"')
args = parser.parse_args()

series = load_hourly()["consumption_kwh"]
start = pd.Timestamp(args.start).floor("h") if args.start else series.index[-1] + pd.Timedelta(hours=1)
history = series[series.index < start]
if len(history) < 168:
    raise SystemExit("Need at least 168 hours (one week) of history before the start time.")
print(f"Model: {'Random Forest' if args.model == 'rf' else 'Long Short-Term Memory'} | "
      f"history up to {history.index[-1]} | forecasting {args.hours} hours from {start}\n")

values = list(history.values)
times = [start + pd.Timedelta(hours=k) for k in range(args.hours)]
preds = []

if args.model == "rf":
    rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
    for ts in times:
        row = pd.DataFrame([{
            "hour": ts.hour, "dayofweek": ts.dayofweek, "month": ts.month,
            "is_weekend": int(ts.dayofweek >= 5),
            "lag_1": values[-1], "lag_24": values[-24], "lag_168": values[-168],
            "roll_24": float(np.mean(values[-24:])),
        }])[FEATURES]
        p = float(rf.predict(row)[0])
        preds.append(p)
        values.append(p)
else:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import load_model
    lstm = load_model(os.path.join(MODEL_DIR, "lstm_model.keras"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    window = list(scaler.transform(pd.DataFrame({"kwh": values[-WINDOW:]})).ravel())
    for _ in times:
        p_scaled = float(lstm.predict(np.array(window[-WINDOW:])[None, :, None], verbose=0)[0, 0])
        preds.append(float(scaler.inverse_transform([[p_scaled]])[0, 0]))
        window.append(p_scaled)

result = pd.DataFrame({"predicted_kwh": np.round(preds, 3)}, index=pd.Index(times, name="datetime"))
actual = series.reindex(result.index)
if actual.notna().any():
    result["actual_kwh"] = actual.round(3)
    result["error_kwh"] = (result["predicted_kwh"] - result["actual_kwh"]).round(3)

print(result.to_string())
peak_time = result["predicted_kwh"].idxmax()
print(f"\nTotal forecast consumption : {result['predicted_kwh'].sum():.1f} kWh")
print(f"Peak demand                : {result['predicted_kwh'].max():.2f} kWh at {peak_time:%Y-%m-%d %H:%M}")
print(f"Lowest demand              : {result['predicted_kwh'].min():.2f} kWh at {result['predicted_kwh'].idxmin():%Y-%m-%d %H:%M}")
if "error_kwh" in result and result["actual_kwh"].notna().any():
    print(f"Mean absolute error on these hours: {result['error_kwh'].abs().mean():.3f} kWh")

result.to_csv("predictions.txt", sep=";")
print("\nSaved predictions.txt")
