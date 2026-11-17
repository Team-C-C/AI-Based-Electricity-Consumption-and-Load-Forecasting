import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_FILE = "household_power_consumption_clean.txt"
MODEL_DIR = "models"
FEATURES = ["hour", "dayofweek", "month", "is_weekend",
            "lag_1", "lag_24", "lag_168", "roll_24"]
WINDOW = 24
TRAIN_FRACTION = 0.8
FEATURES_SHORT = FEATURES
PEAK_HOURS = set(range(7, 10)) | set(range(18, 21))
TARIFF_PEAK, TARIFF_OFFPEAK = 3.50, 1.50


def load_hourly(path=DATA_FILE):
    return pd.read_csv(path, sep=";", parse_dates=["datetime"], index_col="datetime")


def make_features(hourly):
    data = pd.DataFrame({"kwh": hourly["consumption_kwh"]})
    data["hour"] = data.index.hour
    data["dayofweek"] = data.index.dayofweek
    data["month"] = data.index.month
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)
    data["lag_1"] = data["kwh"].shift(1)
    data["lag_24"] = data["kwh"].shift(24)
    data["lag_168"] = data["kwh"].shift(168)
    data["roll_24"] = data["kwh"].shift(1).rolling(24).mean()
    return data.dropna()


def split_data(data):
    split = int(len(data) * TRAIN_FRACTION)
    return data.iloc[:split], data.iloc[split:], split


def make_windows(values, window=WINDOW):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])
    return np.array(X)[..., None], np.array(y)


def evaluate(name, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 0.05, None))) * 100)
    print(f"{name:<24} Mean Absolute Error={mae:.3f} kWh | Root Mean Squared Error={rmse:.3f} kWh "
          f"| Mean Absolute Percentage Error={mape:.1f}%")
    return {"name": name, "mae": mae, "rmse": rmse, "mape": mape}


def clean_data(path=DATA_FILE):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run clean_dataset.py first.")
    hourly = load_hourly(path)
    return hourly.rename(columns={"consumption_kwh": "kwh", "kitchen_kwh": "kitchen",
                                  "laundry_kwh": "laundry", "water_heater_ac_kwh": "water_heater_ac"})


def prepare_history(series, start):
    history = series[series.index < start]
    if len(history) < 168:
        raise SystemExit("Need at least 168 hours (one week) of history before the start time.")
    return history


def forecast_ml(model, features, history, start, hours):
    values = list(history.values)
    times = [start + pd.Timedelta(hours=k) for k in range(hours)]
    preds = []
    for ts in times:
        row = pd.DataFrame([{
            "hour": ts.hour, "dayofweek": ts.dayofweek, "month": ts.month,
            "is_weekend": int(ts.dayofweek >= 5),
            "lag_1": values[-1], "lag_24": values[-24], "lag_168": values[-168],
            "roll_24": float(np.mean(values[-24:])),
        }])[features]
        p = float(model.predict(row)[0])
        preds.append(p)
        values.append(p)
    return pd.Series(preds, index=pd.Index(times, name="datetime"))


def forecast_next_day(hourly, M):
    series = hourly["kwh"]
    start = series.index[-1] + pd.Timedelta(hours=1)
    history = prepare_history(series, start)
    return forecast_ml(M["hour_final"], FEATURES_SHORT, history, start, 24)


def require_models():
    path = os.path.join(MODEL_DIR, "rf_model.joblib")
    if not os.path.exists(path):
        raise SystemExit("Trained model not found. Run train.py first.")
    rf = joblib.load(path)
    data = make_features(load_hourly())
    train, test, split = split_data(data)
    pred = rf.predict(test[FEATURES])
    err = np.abs(test["kwh"].values - pred)
    mae = float(err.mean())
    rmse = float(np.sqrt(np.mean((test["kwh"].values - pred) ** 2)))
    mape = float(np.mean(err / np.clip(test["kwh"].values, 0.05, None)) * 100)
    wape = float(err.sum() / test["kwh"].sum() * 100)
    return {"hour_final": rf, "best_name": "Random Forest",
            "results": {"Random Forest": (mae, rmse, mape, wape)}}
