import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from abis.utils.config import SETTINGS
from abis.drift.psi_drift import psi_drift_score

app = FastAPI(title="ABIS API", version="1.1")

# ----------------------------
# Paths / Registry helpers
# ----------------------------
MODELS_DIR = Path("models")
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
VERSIONS_DIR = MODELS_DIR / "versions"


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"current_version": "unknown"}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"current_version": "unknown"}


def save_registry(current_version: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"current_version": current_version}
    REGISTRY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def list_versions() -> List[str]:
    if not VERSIONS_DIR.exists():
        return []
    versions = sorted([p.name for p in VERSIONS_DIR.iterdir() if p.is_dir()])
    return versions


# ----------------------------
# Drift settings
# ----------------------------
REF_SIZE = 200         # baseline size
WINDOW_SIZE = 50       # rolling window size
DRIFT_THRESHOLD = 0.2  # PSI threshold


# ----------------------------
# Global state (loaded model + drift buffers)
# ----------------------------
class ModelState:
    def __init__(self):
        self.model_dir: Path = Path(SETTINGS.model_dir)
        self.model_version: str = self.model_dir.name

        self.scaler = None
        self.model = None
        self.feature_names: List[str] = []

        # DATA drift buffers (feature vectors)
        self.reference_buffer: List[np.ndarray] = []
        self.current_buffer = deque(maxlen=WINDOW_SIZE)

        # SCORE drift buffers (model score scalar)
        self.score_reference_buffer: List[np.ndarray] = []
        self.score_current_buffer = deque(maxlen=WINDOW_SIZE)

    def reset_drift_buffers(self):
        self.reference_buffer = []
        self.current_buffer = deque(maxlen=WINDOW_SIZE)
        self.score_reference_buffer = []
        self.score_current_buffer = deque(maxlen=WINDOW_SIZE)

    def load_artifacts_from_dir(self, model_dir: Path):
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        self.model_dir = model_dir
        self.model_version = model_dir.name

        self.scaler = joblib.load(model_dir / "scaler.joblib")
        self.model = joblib.load(model_dir / "isolation_forest.joblib")
        self.feature_names = joblib.load(model_dir / "feature_names.joblib")

        self.reset_drift_buffers()


STATE = ModelState()


def resolve_current_model_dir() -> Path:
    """
    Priority:
    1) models/model_registry.json => models/versions/<current_version>
    2) fallback to SETTINGS.model_dir
    """
    reg = load_registry()
    current_version = reg.get("current_version", "unknown")
    candidate = VERSIONS_DIR / current_version

    if candidate.exists():
        return candidate

    # fallback to SETTINGS.model_dir
    return Path(SETTINGS.model_dir)


# Load once on startup
try:
    STATE.load_artifacts_from_dir(resolve_current_model_dir())
except Exception:
    # Keep server up, but /health will show unknown model if load fails
    pass


# ----------------------------
# Schemas
# ----------------------------
class Event(BaseModel):
    data: Dict[str, Any]


class SwitchModelRequest(BaseModel):
    version: str


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": STATE.model_version,
        "ref_size": REF_SIZE,
        "window_size": WINDOW_SIZE,
        "drift_threshold": DRIFT_THRESHOLD,
        "model_dir": str(STATE.model_dir),
        "registry_current_version": load_registry().get("current_version", "unknown"),
        "available_versions": list_versions(),
    }


@app.get("/models")
def models():
    return {
        "available_versions": list_versions(),
        "current_version": load_registry().get("current_version", "unknown"),
        "active_loaded_version": STATE.model_version,
    }


@app.post("/switch_model")
def switch_model(req: SwitchModelRequest):
    versions = list_versions()
    if req.version not in versions:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown version '{req.version}'. Available: {versions}",
        )

    # Update registry + reload artifacts + reset drift buffers
    save_registry(req.version)
    new_dir = VERSIONS_DIR / req.version
    try:
        STATE.load_artifacts_from_dir(new_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    return {
        "status": "switched",
        "model_version": STATE.model_version,
        "model_dir": str(STATE.model_dir),
        "message": f"Switched to {STATE.model_version}. Drift baseline reset.",
    }


@app.post("/reset_drift")
def reset_drift():
    STATE.reset_drift_buffers()
    return {"status": "ok", "message": "Drift buffers reset."}


@app.post("/score")
def score(event: Event):
    if STATE.model is None or STATE.scaler is None or not STATE.feature_names:
        raise HTTPException(status_code=500, detail="Model not loaded. Check /health.")

    # 1) Build row with correct feature order
    row = {f: float(event.data.get(f, 0.0)) for f in STATE.feature_names}
    df = pd.DataFrame([row], columns=STATE.feature_names)

    # 2) Scale and score anomaly
    # Avoid sklearn warning by passing numpy array (scaler was fit without feature names)
    x_scaled = STATE.scaler.transform(df.to_numpy())
    score_val = float(STATE.model.decision_function(x_scaled)[0])
    is_anomaly = bool(score_val < SETTINGS.anomaly_threshold)

    # ----------------------------
    # DATA drift (feature drift)
    # ----------------------------
    x_vec = x_scaled[0].astype(float)

    data_drift_score: Optional[float] = None
    data_drift_alert = False
    data_drift_status = "warming_up"

    if len(STATE.reference_buffer) < REF_SIZE:
        STATE.reference_buffer.append(x_vec)
    else:
        STATE.current_buffer.append(x_vec)

        if len(STATE.current_buffer) >= WINDOW_SIZE:
            ref_arr = np.array(STATE.reference_buffer)
            cur_arr = np.array(STATE.current_buffer)
            data_drift_score = float(psi_drift_score(ref_arr, cur_arr, bins=10))
            data_drift_alert = bool(data_drift_score > DRIFT_THRESHOLD)
            data_drift_status = "ready"

    # ----------------------------
    # SCORE drift (model-output drift) ✅ model-dependent
    # ----------------------------
    score_vec = np.array([score_val], dtype=float)

    score_drift_score: Optional[float] = None
    score_drift_alert = False
    score_drift_status = "warming_up"

    if len(STATE.score_reference_buffer) < REF_SIZE:
        STATE.score_reference_buffer.append(score_vec)
    else:
        STATE.score_current_buffer.append(score_vec)

        if len(STATE.score_current_buffer) >= WINDOW_SIZE:
            ref_s = np.array(STATE.score_reference_buffer)
            cur_s = np.array(STATE.score_current_buffer)
            score_drift_score = float(psi_drift_score(ref_s, cur_s, bins=10))
            score_drift_alert = bool(score_drift_score > DRIFT_THRESHOLD)
            score_drift_status = "ready"

    return {
        "anomaly_score": score_val,
        "anomaly": is_anomaly,

        # Data drift (input drift)
        "data_drift_score": data_drift_score,
        "data_drift_alert": data_drift_alert,
        "data_drift_status": data_drift_status,

        # Score drift (output drift) ✅ changes across models
        "score_drift_score": score_drift_score,
        "score_drift_alert": score_drift_alert,
        "score_drift_status": score_drift_status,

        "model_version": STATE.model_version,
        "features_used": STATE.feature_names,
    }