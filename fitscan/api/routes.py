"""FastAPI routes for the FitScan API."""

from __future__ import annotations

import hashlib
import secrets
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .dependencies import get_measurement_service
from .schemas import MeasurementsResponse, OverrideResponse

router = APIRouter(prefix="/api/v1", tags=["measurements"])


@router.post("/measure", response_model=MeasurementsResponse)
async def measure_body(
    image: UploadFile = File(..., description="Person photo (JPEG/PNG, min 720p)"),
    height_cm: float | None = Form(None),
    reference_object_cm: float | None = Form(None),
    gender_profile: str = Form("ladies"),
    enable_smpl: bool = Form(True),
    enable_claude: bool = Form(True),
    service=Depends(get_measurement_service),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")
    if height_cm is None and reference_object_cm is None:
        raise HTTPException(
            400,
            "Provide either height_cm or reference_object_cm.",
        )
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(await image.read())
        tmp_path = Path(tmp.name)
    try:
        result = service.measure(
            image_path=tmp_path,
            height_cm=height_cm,
            reference_object_cm=reference_object_cm,
            gender_profile=gender_profile,
        )
        return MeasurementsResponse(status="success", **result)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@router.get("/avatar/{avatar_id}/glb")
async def get_avatar_glb(avatar_id: str):
    """Stream the GLB file for a stored avatar."""
    from fitscan.infrastructure.result_store import AvatarStore

    store = AvatarStore()
    avatar = store.get_avatar(avatar_id)
    if not avatar:
        raise HTTPException(404, f"Avatar {avatar_id} not found.")
    glb_path = Path(avatar.glb_path)
    if not glb_path.exists():
        raise HTTPException(404, "GLB file not found on disk.")
    return FileResponse(str(glb_path), media_type="model/gltf-binary")


@router.get("/avatar/{avatar_id}/thumbnail")
async def get_avatar_thumbnail(avatar_id: str):
    """Return a 512x512 PNG preview of the avatar."""
    from fitscan.infrastructure.result_store import AvatarStore

    store = AvatarStore()
    avatar = store.get_avatar(avatar_id)
    if not avatar:
        raise HTTPException(404, f"Avatar {avatar_id} not found.")
    thumb_path = Path(avatar.thumbnail_path)
    if not thumb_path.exists():
        raise HTTPException(404, "Thumbnail not found on disk.")
    return FileResponse(str(thumb_path), media_type="image/png")


@router.post("/avatar/{avatar_id}/fit-garment")
async def fit_garment(
    avatar_id: str,
    garment_type: str = Form(..., description="dress|blouse|kaba|skirt|trousers"),
):
    """Drape a virtual garment onto a stored avatar. Returns draped GLB."""
    from fitscan.avatar.clothing_fitter import ClothingFitter
    from fitscan.infrastructure.result_store import AvatarStore

    store = AvatarStore()
    avatar = store.get_avatar(avatar_id)
    if not avatar:
        raise HTTPException(404, f"Avatar {avatar_id} not found.")
    fitter = ClothingFitter()
    output_dir = Path(f"data/avatars/{avatar_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = fitter.fit_garment(avatar, garment_type, output_dir)
    return FileResponse(result.draped_glb_path, media_type="model/gltf-binary")


@router.post("/avatar/{avatar_id}/override", response_model=OverrideResponse)
async def override_measurements(
    avatar_id: str,
    overrides: dict[str, float],
):
    """Accept manual measurement corrections via Inverse SMPL."""
    from fitscan.infrastructure.result_store import AvatarStore

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


@router.post("/keys/register", tags=["SaaS"])
async def register_api_key(
    owner_name: str = Form(...),
    owner_email: str = Form(...),
    tier: str = Form("free"),
):
    """Register for a FitScan API key."""
    from fitscan.infrastructure.result_store import APIKeyRecord, engine

    raw_key = f"fsk_{secrets.token_hex(24)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    monthly_limit = {"free": 50, "pro": 1000, "enterprise": 999999}.get(tier, 50)

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

    return {
        "api_key": raw_key,
        "tier": tier,
        "monthly_limit": monthly_limit,
        "note": "Store this key securely. It will not be shown again.",
    }
