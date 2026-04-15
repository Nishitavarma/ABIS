from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from abis.utils.config import SETTINGS
from abis.preprocess.make_features import load_and_prepare_features


def main():
    print("🧠 Training Isolation Forest...")

    # 1) Load numeric sensor features
    X, feature_names = load_and_prepare_features(str(SETTINGS.data_path))
    print(f"✅ Loaded data with {X.shape[0]} rows and {X.shape[1]} numeric features.")

    # 2) Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3) Train Isolation Forest
    # contamination = expected fraction of anomalies (start small; tune later)
    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42
    )
    model.fit(X_scaled)
    print("✅ Model trained.")

    # 4) Save artifacts
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(model, models_dir / "isolation_forest.joblib")
    joblib.dump(feature_names, models_dir / "feature_names.joblib")

    print("💾 Saved:")
    print(" - models/scaler.joblib")
    print(" - models/isolation_forest.joblib")
    print(" - models/feature_names.joblib")


if __name__ == "__main__":
    main()
