import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from common import (FEATURES, MODEL_DIR, WINDOW, load_hourly, make_features,
                    make_windows, split_data)

os.makedirs(MODEL_DIR, exist_ok=True)
data = make_features(load_hourly())
train, test, split = split_data(data)
print(f"Hourly records: {len(data):,} | training: {len(train):,} | testing: {len(test):,}")

print("\nTraining Random Forest ...")
rf = RandomForestRegressor(n_estimators=200, max_depth=18, n_jobs=-1, random_state=42)
rf.fit(train[FEATURES], train["kwh"])
joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.joblib"), compress=3)
print("Saved models/rf_model.joblib")

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras import callbacks, layers, models
except ImportError:
    print("\nTensorFlow is not installed, so the Long Short-Term Memory network was skipped.")
    raise SystemExit

np.random.seed(42)
tf.random.set_seed(42)

scaler = MinMaxScaler()
scaler.fit(train[["kwh"]])
scaled = scaler.transform(data[["kwh"]])

Xs, ys = make_windows(scaled[:, 0], WINDOW)
cut = split - WINDOW
Xs_train, ys_train = Xs[:cut], ys[:cut]

print("\nTraining Long Short-Term Memory network ...")
lstm = models.Sequential([
    layers.Input(shape=(WINDOW, 1)),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32),
    layers.Dense(1),
])
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(Xs_train, ys_train, epochs=30, batch_size=64, validation_split=0.1,
         callbacks=[callbacks.EarlyStopping(patience=30, restore_best_weights=True)], verbose=2)

lstm.save(os.path.join(MODEL_DIR, "lstm_model.keras"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
print("Saved models/lstm_model.keras and models/scaler.joblib")
