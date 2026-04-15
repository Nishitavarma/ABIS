import joblib
import pandas as pd
import numpy as np
from collections import deque

from abis.utils.config import SETTINGS
from abis.streaming.stream_simulator import stream_csv
from abis.drift.psi_drift import psi_drift_score


def main():
    print("📥 Loading scaler + feature names for drift monitoring...")
    scaler = joblib.load("models/scaler.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    print("✅ Loaded drift artifacts.")

    ref_size = SETTINGS.reference_window_size
    cur_size = SETTINGS.current_window_size

    reference_window = deque(maxlen=ref_size)
    current_window = deque(maxlen=cur_size)

    print("📡 Streaming events and monitoring drift...")
    for i, event in enumerate(
        stream_csv(SETTINGS.data_path, SETTINGS.stream_delay_seconds, SETTINGS.max_rows),
        start=1
    ):
        row = pd.DataFrame([event]).reindex(columns=feature_names, fill_value=0)
        x_scaled = scaler.transform(row)[0]  # 1D vector

        # First fill reference window, then fill current window
        if len(reference_window) < ref_size:
            reference_window.append(x_scaled)
            if len(reference_window) == ref_size:
                print(f"✅ Reference window filled ({ref_size} events). Now monitoring drift...")
            continue

        current_window.append(x_scaled)

        # Only compute drift once current window is full
        if len(current_window) == cur_size:
            ref = np.array(reference_window)
            cur = np.array(current_window)
            drift_score = psi_drift_score(ref, cur, bins=10)

            drift_alert = drift_score > SETTINGS.drift_threshold
            print(f"Event {i}: drift_score={drift_score:.4f}  drift_alert={drift_alert}")

            # Slide the window: clear current window after checking
            current_window.clear()


if __name__ == "__main__":
    main()
