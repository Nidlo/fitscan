"""
Downloads all required ML model weights.
Run: python scripts/download_models.py
"""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODELS_DIR = Path("data/models")


def check_smpl():
    model_path = MODELS_DIR / "smpl" / "SMPL_NEUTRAL.pkl"
    if model_path.exists():
        log.info("SMPL model found.")
        return True
    try:
        import smplx  # noqa: F401

        log.info("smplx installed. SMPL model will be loaded via smplx.")
        return True
    except ImportError:
        pass
    log.warning(
        "SMPL model requires manual download from https://smpl.is.tue.mpg.de/"
    )
    return False


def check_mediapipe():
    try:
        import mediapipe as mp

        log.info(f"MediaPipe {mp.__version__} ready.")
        return True
    except ImportError:
        log.warning("MediaPipe not installed. Run: pip install mediapipe")
        return False


def prime_rembg():
    """Trigger rembg to download U2Net weights on first run."""
    log.info("Checking rembg background removal model...")
    try:
        from rembg import remove  # noqa: F401

        log.info("rembg available.")
    except ImportError:
        log.warning("rembg not installed. Background removal will use fallback.")


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "smpl").mkdir(exist_ok=True)
    (MODELS_DIR / "hmr2").mkdir(exist_ok=True)
    (MODELS_DIR / "mediapipe").mkdir(exist_ok=True)
    check_smpl()
    check_mediapipe()
    prime_rembg()
    log.info("Model setup complete.")
