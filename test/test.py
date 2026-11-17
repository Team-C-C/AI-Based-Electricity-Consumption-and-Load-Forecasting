import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (FEATURES, MODEL_DIR, WINDOW, evaluate, load_hourly,
                    make_features, make_windows, split_data)

data = make_features(load_hourly())
train, test, split = split_data(data)
print(f"Testing on {len(test):,} unseen hours "
      f"({test.index.min():%Y-%m-%d} to {test.index.max():%Y-%m-%d})\n")

results = [evaluate("Baseline (lag_24)", test["kwh"], test["lag_24"])]

rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
rf_pred = rf.predict(test[FEATURES])
results.append(evaluate("Random Forest", test["kwh"], rf_pred))

lstm_pred = None
lstm_path = os.path.join(MODEL_DIR, "lstm_model.keras")
if os.path.exists(lstm_path):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import load_model
    lstm = load_model(lstm_path)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    scaled = scaler.transform(data[["kwh"]])
    Xs, ys = make_windows(scaled[:, 0], WINDOW)
    Xs_test = Xs[split - WINDOW:]
    lstm_pred = scaler.inverse_transform(lstm.predict(Xs_test, verbose=0)).ravel()
    results.append(evaluate("Long Short-Term Memory", test["kwh"], lstm_pred))
else:
    print("(No Long Short-Term Memory model found, skipped.)")

base = results[0]
print("\nError reduction compared with the baseline:")
for r in results[1:]:
    print(f"  {r['name']:<24} Mean Absolute Error {100 * (1 - r['mae'] / base['mae']):.0f}% lower | "
          f"Root Mean Squared Error {100 * (1 - r['rmse'] / base['rmse']):.0f}% lower")

n = 24 * 7
plt.figure(figsize=(12, 4))
plt.plot(test.index[-n:], test["kwh"].values[-n:], label="Actual")
plt.plot(test.index[-n:], rf_pred[-n:], label="Random Forest")
if lstm_pred is not None:
    plt.plot(test.index[-n:], lstm_pred[-n:], label="Long Short-Term Memory")
plt.ylabel("Electricity consumption (kWh per hour)")
plt.title("Forecast versus actual consumption (last 7 days of the test period)")
plt.legend()
plt.tight_layout()
plt.savefig("forecast_vs_actual.png", dpi=150)
print("\nSaved forecast_vs_actual.png")
