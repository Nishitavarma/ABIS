from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 📁 Project root folder (abis/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

@dataclass
class Settings:
    # =========================
    # 📊 DATA SETTINGS
    # =========================
    data_path: Path = PROJECT_ROOT / "data" / "predictive_maintenance.csv"
    stream_delay_seconds: float = 0.01
    max_rows: Optional[int] = 1000

    # =========================
    # 🧠 MODEL STORAGE
    # =========================
    model_dir: Path = PROJECT_ROOT / "models"

    # =========================
    # 🚨 ANOMALY SETTINGS
    # =========================
    anomaly_threshold: float = 0.0

    # =========================
    # 📈 DRIFT DETECTION
    # =========================
    reference_window_size: int = 200
    current_window_size: int = 100
    drift_threshold: float = 0.2

    # =========================
    # 🔁 ADAPTIVE LEARNING
    # =========================
    retrain_on_drift: bool = True
    drift_alerts_to_retrain: int = 1

# 🌍 Global settings object
SETTINGS = Settings()
