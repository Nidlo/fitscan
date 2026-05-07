"""FastAPI application entry point."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Pillow's default DOS guard is conservative; tighten it to twice the
# largest sensible camera capture so a malicious decompression bomb fails
# fast instead of allocating gigabytes (audit FS-C4). Override via env in
# the unlikely case a real workflow needs more.
try:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = int(
        os.getenv("FITSCAN_MAX_IMAGE_PIXELS", str(60 * 1024 * 1024))
    )
except ImportError:  # Pillow is a dep but defend against import-time issues
    pass

from fitscan.config import get_settings

from .admin_routes import router as admin_router
from .routes import router

app = FastAPI(
    title="FitScan Body Measurement API",
    description="AI-powered body measurement from photographs",
    version="0.1.0",
)
# FS-NIDLO-MEAS-03 — CORS allowlist driven from `Settings.cors_origins`
# (defaults to the Laravel backend host only). Set
# `FITSCAN_CORS_ORIGINS=*` in dev if you need to hit the API from
# arbitrary tools; production must keep a finite list.
_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
# FS-NIDLO-OPS-01 — Admin-only ops endpoints (Celery health, future:
# usage metrics for RPT-006). Mounted separately from `router` because
# the auth gate is a different shared secret (X-Admin-API-Key, not the
# per-tenant X-API-Key).
app.include_router(admin_router)
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Serve API documentation at /api-docs
_docs_dir = Path(__file__).parent.parent.parent / "docs"
if _docs_dir.exists():
    app.mount(
        "/api-docs", StaticFiles(directory=str(_docs_dir), html=True), name="api-docs"
    )
