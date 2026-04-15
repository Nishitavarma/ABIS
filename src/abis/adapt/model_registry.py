from pathlib import Path

MODELS_DIR = Path("models")
VERSIONS_DIR = MODELS_DIR / "versions"


def ensure_dirs():
    MODELS_DIR.mkdir(exist_ok=True)
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)


def next_version_dir() -> Path:
    """
    Returns a new version folder path like:
    models/versions/model_v001, model_v002, ...
    """
    ensure_dirs()
    existing = sorted([p for p in VERSIONS_DIR.glob("model_v*") if p.is_dir()])

    if not existing:
        return VERSIONS_DIR / "model_v001"

    last = existing[-1].name  # e.g., model_v007
    n = int(last.split("_v")[-1])
    return VERSIONS_DIR / ("model_v%03d" % (n + 1))


def set_current_symlink_or_copy(version_dir: Path):
    """
    For Windows, symlinks can be annoying, so we will copy artifacts
    to the top-level models/ files as the 'current' model.
    """
    # The caller will copy scaler/model/feature_names to models/
    pass
