"""FastAPI routes for the FitScan API."""

from __future__ import annotations

import hashlib
import os
import secrets
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from loguru import logger
from sqlalchemy.orm import Session

from fitscan.avatar.clothing_fitter import ClothingFitter
from fitscan.avatar.generator import AvatarGenerator
from fitscan.config import get_settings
from fitscan.infrastructure.result_store import (
    APIKeyRecord,
    AvatarStore,
    engine,
    record_measurement_run,
)

from .dependencies import get_measurement_service, verify_admin_key, verify_api_key
from .schemas import MeasurementsResponse, OverrideResponse

# S-13 / FS-NIDLO-MEAS-04 — `safe_500` shape: clients see a generic
# message + a `trace_id` they can quote when they file a bug; the full
# exception lands in the structured log keyed against the same
# `trace_id`. Operators can grep one ID and see the whole stack.
_PIPELINE_FAILED_MESSAGE = (
    "Measurement pipeline failed. The error has been recorded; "
    "please retry, or fall back to manual entry."
)


def _safe_500(probe: str, exc: BaseException) -> HTTPException:
    """Build a 500 with `{detail, trace_id}` and emit the matching log
    line. Pipeline catch blocks call this instead of raising directly so
    the response shape stays uniform and the trace_id correlation always
    fires. (FS-NIDLO-MEAS-04)
    """
    trace_id = uuid.uuid4().hex
    logger.opt(exception=exc).error(
        "fitscan_500 probe={probe} trace_id={trace_id}",
        probe=probe,
        trace_id=trace_id,
    )
    return HTTPException(
        status_code=500,
        detail={"message": _PIPELINE_FAILED_MESSAGE, "trace_id": trace_id},
    )

# Reusable dependency list for routes that require API-key auth.
_auth = [Depends(verify_api_key)]

# Hard cap on image upload size — guards against OOM from a single huge JPEG.
# Default 10MB matches ImageLoader.MAX_FILE_SIZE_MB; override via env.
_MAX_UPLOAD_BYTES = int(os.getenv("FITSCAN_MAX_UPLOAD_MB", "10")) * 1024 * 1024


# Filesystem root for stored avatar artifacts. Any path served via
# FileResponse() must resolve inside this directory — without this guard,
# a writable `glb_path` column (via SQLi or a future admin-panel bug)
# could serve `/etc/passwd`, the SQLite DB itself, or the Anthropic key
# in `.env`. Audit FS-C9.
_AVATARS_ROOT = (Path(os.getenv("FITSCAN_DATA_DIR", "data")) / "avatars").resolve()


def _resolve_inside_avatars_root(stored_path: str | None) -> Path:
    """Resolve a DB-stored path and assert it sits inside the avatars root.

    Raises 404 if the path is missing, escapes the root, or doesn't exist
    on disk. 404 (not 403) so we don't leak whether a row exists with a
    poisoned path vs. no row at all.
    """
    if not stored_path:
        raise HTTPException(404, "Avatar artifact not found.")
    candidate = Path(stored_path).resolve()
    try:
        candidate.relative_to(_AVATARS_ROOT)
    except ValueError as exc:
        raise HTTPException(404, "Avatar artifact not found.") from exc
    if not candidate.is_file():
        raise HTTPException(404, "Avatar artifact not found.")
    return candidate


def _upload_tempdir() -> Path:
    """Resolve the upload tempdir from Settings, creating it if missing.
    Centralised so the app-startup sweep and `_stream_to_tempfile`
    agree on the same path. (FS-NIDLO-OPS-03)"""
    target = Path(get_settings().upload_tempdir)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _stream_to_tempfile(image: UploadFile) -> Path:
    """Stream an UploadFile to disk in chunks, aborting at the size cap.

    `image.read()` (no arg) buffers the entire body into memory before
    any size check fires; a 5GB upload OOMs the process before
    ImageLoader.MAX_FILE_SIZE_MB is consulted (audit FS-C4). Streaming
    with a running total fails fast.

    Sync as of FS-NIDLO-MEAS-18 — the `/measure*` routes are now sync
    so FastAPI hands the whole handler to its threadpool, unblocking
    the event loop while PyTorch/MediaPipe runs. UploadFile.file is
    the underlying SpooledTemporaryFile and has a sync `read()`.

    Files land in `Settings.upload_tempdir` (default
    `/tmp/fitscan_uploads/`) so the boot-time sweep can clean
    leftovers from a prior hard-killed worker. (FS-NIDLO-OPS-03)
    """
    chunk_size = 64 * 1024
    total = 0
    with tempfile.NamedTemporaryFile(
        suffix=".jpg", delete=False, dir=str(_upload_tempdir())
    ) as tmp:
        try:
            while True:
                chunk = image.file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    tmp.close()
                    Path(tmp.name).unlink(missing_ok=True)
                    raise HTTPException(
                        413,
                        f"Image exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
                    )
                tmp.write(chunk)
        except HTTPException:
            raise
        except Exception:
            tmp.close()
            Path(tmp.name).unlink(missing_ok=True)
            raise
        return Path(tmp.name)


def sweep_upload_tempdir() -> int:
    """Delete files in `Settings.upload_tempdir` older than
    `upload_tempdir_max_age_seconds`. Called at app startup so a worker
    that died mid-request (uvicorn SIGKILL, OOM) doesn't leave a
    user-uploaded body photo lying around indefinitely. Returns the
    number of files removed for the boot log. (FS-NIDLO-OPS-03)
    """
    settings = get_settings()
    target = Path(settings.upload_tempdir)
    if not target.exists():
        return 0

    cutoff = time.time() - settings.upload_tempdir_max_age_seconds
    removed = 0
    for entry in target.iterdir():
        if not entry.is_file():
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink(missing_ok=True)
                removed += 1
        except OSError as e:  # noqa: BLE001 — best-effort sweep
            logger.warning(
                "tempdir sweep skipped {entry}: {err}", entry=str(entry), err=str(e)
            )
    return removed


router = APIRouter(prefix="/api/v1", tags=["measurements"])


@router.post("/measure", response_model=MeasurementsResponse, dependencies=_auth)
def measure_body(
    image: UploadFile = File(..., description="Person photo (JPEG/PNG, min 720p)"),
    height_cm: float | None = Form(None),
    reference_object_cm: float | None = Form(None),
    gender_profile: str = Form("ladies"),
    enable_smpl: bool = Form(True),
    enable_claude: bool = Form(True),
    subject_id: str | None = Form(
        None,
        description=(
            "Required to ship the photo to Anthropic for Claude validation. "
            "Pass an opaque per-end-user identifier from the upstream caller "
            "(e.g. the Laravel user UUID). Without it the photo is NOT sent "
            "to any third-party processor."
        ),
    ),
    service=Depends(get_measurement_service),
):
    started_at = datetime.now(timezone.utc)
    error_kind: str | None = None

    try:
        if not image.content_type or not image.content_type.startswith("image/"):
            error_kind = "validation"
            raise HTTPException(400, "File must be an image.")
        if height_cm is None and reference_object_cm is None:
            error_kind = "validation"
            raise HTTPException(
                400,
                "Provide either height_cm or reference_object_cm.",
            )
        tmp_path = _stream_to_tempfile(image)
        try:
            result = service.measure(
                image_path=tmp_path,
                height_cm=height_cm,
                reference_object_cm=reference_object_cm,
                gender_profile=gender_profile,
                subject_id=subject_id,
                enable_smpl=enable_smpl,
                enable_claude=enable_claude,
            )
            return MeasurementsResponse(status="success", **result)
        except HTTPException:
            error_kind = error_kind or "pipeline"
            raise
        except Exception as e:
            error_kind = "pipeline"
            raise _safe_500("measure", e) from e
        finally:
            tmp_path.unlink(missing_ok=True)
    finally:
        completed_at = datetime.now(timezone.utc)
        record_measurement_run(
            started_at=started_at,
            completed_at=completed_at,
            success=error_kind is None,
            error_kind=error_kind,
        )


@router.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@router.get("/avatar/{avatar_id}/glb", dependencies=_auth)
async def get_avatar_glb(avatar_id: str):
    """Stream the GLB file for a stored avatar."""

    avatar = AvatarStore().get_avatar(avatar_id)
    if not avatar:
        raise HTTPException(404, f"Avatar {avatar_id} not found.")
    glb_path = _resolve_inside_avatars_root(avatar.glb_path)
    return FileResponse(str(glb_path), media_type="model/gltf-binary")


@router.get("/avatar/{avatar_id}/thumbnail", dependencies=_auth)
async def get_avatar_thumbnail(avatar_id: str):
    """Return a 512x512 PNG preview of the avatar."""

    avatar = AvatarStore().get_avatar(avatar_id)
    if not avatar:
        raise HTTPException(404, f"Avatar {avatar_id} not found.")
    thumb_path = _resolve_inside_avatars_root(avatar.thumbnail_path)
    return FileResponse(str(thumb_path), media_type="image/png")


@router.delete("/avatar/{avatar_id}", dependencies=_auth, status_code=204)
async def delete_avatar(avatar_id: str):
    """Right-to-be-forgotten path: hard-delete avatar DB row + filesystem.

    Called by the upstream platform (Laravel backend) when a user requests
    deletion of their body-scan data, or by the periodic retention task.
    """

    if not AvatarStore().delete(avatar_id):
        raise HTTPException(404, f"Avatar {avatar_id} not found.")
    return None


@router.post("/avatar/{avatar_id}/fit-garment", dependencies=_auth)
async def fit_garment(
    avatar_id: str,
    garment_type: str = Form(..., description="dress|blouse|kaba|skirt|trousers"),
):
    """Drape a virtual garment onto a stored avatar. Returns draped GLB."""

    store = AvatarStore()
    avatar = store.get_avatar(avatar_id)
    if not avatar:
        raise HTTPException(404, f"Avatar {avatar_id} not found.")
    fitter = ClothingFitter()
    output_dir = Path(f"data/avatars/{avatar_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = fitter.fit_garment(avatar, garment_type, output_dir)
    draped = _resolve_inside_avatars_root(str(result.draped_glb_path))
    return FileResponse(str(draped), media_type="model/gltf-binary")


@router.post(
    "/avatar/{avatar_id}/override",
    response_model=OverrideResponse,
    dependencies=_auth,
)
async def override_measurements(
    avatar_id: str,
    overrides: dict[str, float],
):
    """Accept manual measurement corrections via Inverse SMPL."""

    store = AvatarStore()
    avatar = store.get_avatar(avatar_id)
    if not avatar:
        raise HTTPException(404, f"Avatar {avatar_id} not found.")
    if not overrides:
        raise HTTPException(400, "No overrides provided.")

    return OverrideResponse(
        avatar_id=avatar_id,
        recalculated_measurements={},
        updated_betas=avatar.betas,
        convergence_note="Override not yet implemented in this build.",
    )


# ── SaaS API Key Management ─────────────────────────────────────────


def _tier_limits() -> dict[str, int]:
    """Tier → monthly request quota, sourced from `Settings`. Single
    source of truth — both the allowlist used for input validation and
    the limit written to the DB row come from this map. Operators can
    raise free-tier limits etc. via env without a code edit.
    (FS-NIDLO-AUTH-05)
    """

    s = get_settings()
    return {
        "free": s.free_tier_monthly_limit,
        "pro": s.pro_tier_monthly_limit,
        "enterprise": s.enterprise_tier_monthly_limit,
    }


@router.post(
    "/keys/register",
    tags=["SaaS"],
    dependencies=[Depends(verify_admin_key)],
)
def register_api_key(
    request: Request,
    owner_name: str = Form(...),
    owner_email: str = Form(...),
    tier: str = Form("free"),
):
    """Register a FitScan API key for a customer.

    FS-NIDLO-AUTH-03 lockdown — admin-only. Snad mints keys via:

        curl -X POST https://fitscan.nidlo.com/api/v1/keys/register \\
             -H "X-Admin-API-Key: $FITSCAN_ADMIN_API_KEY" \\
             -F "owner_name=Acme Tailoring" \\
             -F "owner_email=ops@acme.test" \\
             -F "tier=pro"

    Until a Filament admin button ships, this curl is the canonical
    issuance path. The same `X-Admin-API-Key` header gates the
    `/api/v1/admin/*` operational routes — one secret to rotate.

    Hardening that's been in place since pre-lockdown:
      - tier allowlist (no arbitrary string accepted)
      - audit log on every registration so abuse is visible

    See `LAUNCHING.md` §6 for the trust-model trade-off (admin-only
    vs. trusted-network vs. email-confirm) and why admin-only is
    right for the next ~6 months of curated beta.
    """

    tier_limits = _tier_limits()
    if tier not in tier_limits:
        raise HTTPException(
            400,
            f"Unknown tier '{tier}'. Allowed: {', '.join(sorted(tier_limits))}.",
        )

    monthly_limit = tier_limits[tier]
    raw_key = f"fsk_{secrets.token_hex(24)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    with Session(engine) as session:
        record = APIKeyRecord(
            key_hash=key_hash,
            owner_name=owner_name,
            owner_email=owner_email,
            tier=tier,
            monthly_limit=monthly_limit,
        )
        session.add(record)
        session.commit()

    # Audit trail — every issued key gets a structured log line so abuse
    # patterns (one IP issuing 50 keys/min, throwaway emails) surface in
    # ops dashboards even without an application-layer rate limit. The
    # raw key is NOT logged; only the SHA-256 prefix to correlate later.
    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        "api_key_issued tier={tier} owner_email={email} client_ip={ip} key_hash_prefix={prefix}",
        tier=tier,
        email=owner_email,
        ip=client_ip,
        prefix=key_hash[:12],
    )

    return {
        "api_key": raw_key,
        "tier": tier,
        "monthly_limit": monthly_limit,
        "note": "Store this key securely. It will not be shown again.",
    }


@router.post(
    "/measure/measurements",
    response_model=MeasurementsResponse,
    tags=["Measure"],
    summary="Measurements only — no avatar generated",
    description="Faster than /measure. Returns only the 24-field measurement JSON. No GLB is generated.",
    dependencies=_auth,
)
def measure_measurements_only(
    image: UploadFile = File(...),
    height_cm: float | None = Form(None),
    reference_object_cm: float | None = Form(None),
    gender_profile: str = Form("ladies"),
    enable_smpl: bool = Form(True),
    enable_claude: bool = Form(True),
    subject_id: str | None = Form(None),
    service=Depends(get_measurement_service),
):
    """Measurements only — skips avatar generation entirely.

    `gender_profile`, `enable_smpl`, and `enable_claude` are now wired
    through to the pipeline (FS-NIDLO-MEAS-05). Previously they were
    accepted as form fields but silently dropped on this route.
    """
    if not height_cm and not reference_object_cm:
        raise HTTPException(400, "Provide height_cm or reference_object_cm.")
    tmp_path = _stream_to_tempfile(image)
    try:
        result = service.measure(
            image_path=tmp_path,
            height_cm=height_cm,
            reference_object_cm=reference_object_cm,
            gender_profile=gender_profile,
            subject_id=subject_id,
            enable_smpl=enable_smpl,
            enable_claude=enable_claude,
        )
        response = MeasurementsResponse(status="success", **result)
        response.avatar_id = None  # explicitly None — no avatar generated
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise _safe_500("measure_measurements", e) from e
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post(
    "/measure/avatar",
    tags=["Measure"],
    summary="Avatar only — returns GLB binary directly",
    description="Returns the raw GLB binary. Response Content-Type is model/gltf-binary. No measurements JSON.",
    dependencies=_auth,
)
def measure_avatar_only(
    image: UploadFile = File(...),
    height_cm: float | None = Form(None),
    reference_object_cm: float | None = Form(None),
    gender_profile: str = Form("ladies"),
    subject_id: str | None = Form(None),
    service=Depends(get_measurement_service),
):
    """Avatar only — returns GLB file directly as binary response."""
    if not height_cm and not reference_object_cm:
        raise HTTPException(400, "Provide height_cm or reference_object_cm.")

    tmp_path = _stream_to_tempfile(image)
    content = tmp_path.read_bytes()

    try:
        result, mesh, betas, thetas = service.measure_with_mesh(
            image_path=tmp_path,
            height_cm=height_cm,
            reference_object_cm=reference_object_cm,
            gender_profile=gender_profile,
            subject_id=subject_id,
        )
        if mesh is None:
            raise HTTPException(500, "3D mesh generation failed.")


        source_img = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        gen = AvatarGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            avatar = gen.generate(
                mesh=mesh,
                source_image=source_img,
                measurements=result,
                betas=betas or [],
                thetas=thetas or [],
                gender_profile=gender_profile,
                output_dir=Path(tmpdir),
            )
            glb_bytes = Path(avatar.glb_path).read_bytes()
            height_val = result.height_cm.value_cm if result.height_cm else 0

        return Response(
            content=glb_bytes,
            media_type="model/gltf-binary",
            headers={
                "X-Avatar-Id": str(uuid.uuid4()),
                "X-Height-Cm": str(height_val),
                "Content-Disposition": "attachment; filename=avatar.glb",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise _safe_500("measure_avatar", e) from e
    finally:
        tmp_path.unlink(missing_ok=True)
