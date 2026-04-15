import joblib
import pandas as pd
import numpy as np
from collections import deque

from abis.utils.config import SETTINGS
from abis.streaming.stream_simulator import stream_csv
from abis.drift.psi_drift import psi_drift_score
from abis.adapt.model_registry import next_version_dir
from abis.adapt.retrain import train_isolation_forest_from_array, save_model_artifacts


def load_current_artifacts():
    scaler = joblib.load("models/scaler.joblib")
    model = joblib.load("models/isolation_forest.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    return scaler, model, feature_names


def main():
    print("🤖 ABIS Adaptive Run starting...")

    # Load current model
    scaler, model, feature_names = load_current_artifacts()
    print("✅ Loaded current anomaly model.")

    ref_size = SETTINGS.reference_window_size
    cur_size = SETTINGS.current_window_size

    reference_window = deque(maxlen=ref_size)
    current_window = deque(maxlen=cur_size)

    drift_alert_count = 0

    print("📡 Streaming + anomaly scoring + drift monitoring...")

    for i, event in enumerate(
        stream_csv(SETTINGS.data_path, SETTINGS.stream_delay_seconds, SETTINGS.max_rows),
        start=1
    ):
        row = pd.DataFrame([event]).reindex(columns=feature_names, fill_value=0)
        x = row.values.astype(float)[0]  # raw numeric vector

        # ---- ANOMALY SCORE (per event) ----
        x_scaled_for_score = scaler.transform(row)
        anomaly_score = float(model.decision_function(x_scaled_for_score)[0])
        is_anomaly = (int(model.predict(x_scaled_for_score)[0]) == -1)

        # Print anomaly result (you can later send to dashboard/API)
        if is_anomaly:
            print(f"Event {i}: 🚨 ANOMALY score={anomaly_score:.4f}")
        else:
            # keep it light so terminal isn't spammy
            if i % 50 == 0:
                print(f"Event {i}: normal score={anomaly_score:.4f}")

        # ---- DRIFT WINDOWS (need many events) ----
        # Build reference window first
        if len(reference_window) < ref_size:
            reference_window.append(x)
            if len(reference_window) == ref_size:
                print(f"✅ Reference window filled ({ref_size}). Drift monitoring ON.")
            continue

        # Fill current window
        current_window.append(x)

        # When current window full → compute drift
        if len(current_window) == cur_size:
            ref = np.array(reference_window)
            cur = np.array(current_window)

            # IMPORTANT: use same scaler logic for drift comparisons
            # We'll fit scaler on reference for drift comparison stability
            drift_scaler = joblib.load("models/scaler.joblib")
            ref_s = drift_scaler.transform(ref)
            cur_s = drift_scaler.transform(cur)

            drift_score = psi_drift_score(ref_s, cur_s, bins=10)
            drift_alert = drift_score > SETTINGS.drift_threshold

            print(f"Event {i}: 📈 drift_score={drift_score:.4f} drift_alert={drift_alert}")

            # ---- ADAPTIVE LEARNING TRIGGER ----
            if SETTINGS.retrain_on_drift and drift_alert:
                drift_alert_count += 1
                print(f"⚠️ Drift alert count = {drift_alert_count}")

                if drift_alert_count >= SETTINGS.drift_alerts_to_retrain:
                    print("🧠 Retraining triggered! Creating a new model version...")

                    # Train using RECENT behavior (current window)
                    # (You can also combine ref+cur, but this is simple and clear)
                    new_scaler, new_model = train_isolation_forest_from_array(cur)

                    version_dir = next_version_dir()
                    save_model_artifacts(version_dir, new_scaler, new_model, feature_names)

                    print(f"✅ New model saved to: {version_dir}")
                    print("🔁 ABIS switched to the new model.")

                    # Switch current model in memory too
                    scaler, model = new_scaler, new_model

                    # Reset drift counter and update baseline reference to the new normal
                    drift_alert_count = 0
                    reference_window.clear()
                    for v in cur:
                        reference_window.append(v)

                    print("✅ Reference window updated to new behavior baseline.")

            # clear current window to start collecting next chunk
            current_window.clear()


if __name__ == "__main__":
    main()
