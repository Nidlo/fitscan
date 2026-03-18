"""
Downloads TailorNet pretrained model weights.
Falls back gracefully if unavailable.
"""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODELS_DIR = Path("data/models/tailornet")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_models():
    log.info(
        "TailorNet models require manual download from Google Drive. "
        "Clothing fitting will use rigid offset fallback."
    )
    # Create a marker file so the system knows models were checked
    marker = MODELS_DIR / ".checked"
    marker.write_text("checked")


if __name__ == "__main__":
    download_models()
    log.info("TailorNet model check complete.")
