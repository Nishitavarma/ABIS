from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


def train_isolation_forest_from_array(X: np.ndarray):
    """
    Trains scaler + IsolationForest from a numeric array (n_samples, n_features).
    Returns (scaler, model).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42
    )
    model.fit(X_scaled)
    return scaler, model


def save_model_artifacts(version_dir: Path, scaler, model, feature_names):
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, version_dir / "scaler.joblib")
    joblib.dump(model, version_dir / "isolation_forest.joblib")
    joblib.dump(feature_names, version_dir / "feature_names.joblib")

    # Also copy to top-level models/ as the “current” model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(model, models_dir / "isolation_forest.joblib")
    joblib.dump(feature_names, models_dir / "feature_names.joblib")
