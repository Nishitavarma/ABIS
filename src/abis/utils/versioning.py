from pathlib import Path
from datetime import datetime

def next_model_version(models_dir: str = "models") -> str:
    """
    Creates a timestamp-based version string like: v20260111_183045
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"v{ts}"

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
