"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

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
from .style import router as style_router


# FS-NIDLO-OPS-01 — Sentry init. No-op when SENTRY_DSN is unset (dev /
# test / fresh installs that haven't provisioned a Sentry project yet),
# so adding the SDK doesn't change behaviour for anyone today. Set
# SENTRY_DSN in production to start receiving error + perf telemetry.
#
# Uses sentry-sdk's first-party LoguruIntegration so `logger.exception`
# / `logger.error` calls across the app reach Sentry without us having
# to build a manual logging-stdlib bridge.
def _init_sentry() -> None:
    settings = get_settings()
    if not settings.sentry_dsn:
        return

    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.loguru import LoguruIntegration
    from sentry_sdk.integrations.starlette import StarletteIntegration

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.app_env,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        integrations=[
            StarletteIntegration(),
            FastApiIntegration(),
            LoguruIntegration(),
        ],
        # Body photos are sensitive PII — never let request bodies or
        # cookies leave the box even if the SDK would normally include
        # them in the breadcrumb stream.
        send_default_pii=False,
    )


_init_sentry()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """FS-NIDLO-OPS-03 — sweep stale upload tempfiles on startup so a
    SIGKILLed worker doesn't leave a user-uploaded body photo on disk
    indefinitely. The sweep is best-effort and never blocks startup.
    """
    from .routes import sweep_upload_tempdir

    try:
        removed = sweep_upload_tempdir()
        if removed:
            logger.info(
                "upload_tempdir_sweep removed={removed} stale tempfiles",
                removed=removed,
            )
    except Exception as e:  # noqa: BLE001 — best-effort, never block startup
        logger.warning("upload_tempdir_sweep failed: {err}", err=str(e))
    yield


app = FastAPI(
    title="FitScan Body Measurement API",
    description="AI-powered body measurement from photographs",
    version="0.1.0",
    lifespan=_lifespan,
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
app.include_router(style_router)
# FS-NIDLO-OPS-01 — Admin-only ops endpoints (Celery health, future:
# usage metrics for RPT-006). Mounted separately from `router` because
# the auth gate is a different shared secret (X-Admin-API-Key, not the
# per-tenant X-API-Key).
app.include_router(admin_router)

# Serve API documentation at /api-docs
_docs_dir = Path(__file__).parent.parent.parent / "docs"
if _docs_dir.exists():
    app.mount(
        "/api-docs", StaticFiles(directory=str(_docs_dir), html=True), name="api-docs"
    )
