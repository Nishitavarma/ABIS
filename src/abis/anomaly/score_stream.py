import joblib
import pandas as pd

from abis.utils.config import SETTINGS
from abis.streaming.stream_simulator import stream_csv


def main():
    print("📥 Loading model artifacts...")

    scaler = joblib.load("models/scaler.joblib")
    model = joblib.load("models/isolation_forest.joblib")
    feature_names = joblib.load("models/feature_names.joblib")

    print("✅ Loaded scaler + model + feature list.")
    print("📡 Streaming and scoring events...")

    for i, event in enumerate(
        stream_csv(SETTINGS.data_path, SETTINGS.stream_delay_seconds, SETTINGS.max_rows),
        start=1
    ):
        row = pd.DataFrame([event])
        X_row = row.reindex(columns=feature_names, fill_value=0)

        X_scaled = scaler.transform(X_row)

        score = float(model.decision_function(X_scaled)[0])
        pred = int(model.predict(X_scaled)[0])  # -1 anomaly, 1 normal
        is_anomaly = (pred == -1)

        print(f"Event {i}: anomaly_score={score:.4f}  anomaly={is_anomaly}")


if __name__ == "__main__":
    main()
