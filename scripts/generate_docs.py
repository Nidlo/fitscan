"""
FitScan API Documentation Generator
=====================================
Run this script with Claude Code to:
1. Add the three isolated API endpoints (measurements-only, avatar-only, both)
2. Export the live OpenAPI schema from the running FastAPI app
3. Copy docs files into the correct location
4. Serve the docs locally

Usage (from stitchub/fitscan/ directory):
  python scripts/generate_docs.py

Or via Claude Code:
  claude --dangerously-skip-permissions "Run scripts/generate_docs.py"
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

FITSCAN_DIR = Path(__file__).parent.parent
DOCS_DIR = FITSCAN_DIR / "docs"
API_ROUTES = FITSCAN_DIR / "fitscan" / "api" / "routes.py"
SCHEMAS = FITSCAN_DIR / "fitscan" / "api" / "schemas.py"

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
NC = "\033[0m"

def log(msg): print(f"{GREEN}[DOCS]{NC} {msg}")
def warn(msg): print(f"{YELLOW}[WARN]{NC} {msg}")
def err(msg): print(f"{RED}[ERROR]{NC} {msg}")


# ─── Step 1: Add isolated endpoints to routes.py ──────────────────────────────

MEASUREMENTS_ONLY_ROUTE = '''

@router.post(
    "/measure/measurements",
    response_model=MeasurementsResponse,
    tags=["Measure"],
    summary="Measurements only — no avatar generated",
    description="Faster than /measure. Returns only the 24-field measurement JSON. No GLB is generated.",
)
async def measure_measurements_only(
    image: UploadFile = File(...),
    height_cm: float | None = Form(None),
    reference_object_cm: float | None = Form(None),
    gender_profile: str = Form("ladies"),
    enable_smpl: bool = Form(True),
    enable_claude: bool = Form(True),
    service=Depends(get_measurement_service),
):
    """Measurements only — skips avatar generation entirely."""
    if not height_cm and not reference_object_cm:
        raise HTTPException(400, "Provide height_cm or reference_object_cm.")
    import tempfile
    content = await image.read()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        result = service.measure(
            image_path=tmp_path,
            height_cm=height_cm,
            reference_object_cm=reference_object_cm,
        )
        response = MeasurementsResponse(status="success", **result)
        response.avatar_id = None  # explicitly None — no avatar generated
        return response
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post(
    "/measure/avatar",
    tags=["Measure"],
    summary="Avatar only — returns GLB binary directly",
    description="Returns the raw GLB binary. Response Content-Type is model/gltf-binary. No measurements JSON.",
)
async def measure_avatar_only(
    image: UploadFile = File(...),
    height_cm: float | None = Form(None),
    reference_object_cm: float | None = Form(None),
    gender_profile: str = Form("ladies"),
    service=Depends(get_measurement_service),
):
    """Avatar only — returns GLB file directly as binary response."""
    from fastapi.responses import Response
    import tempfile
    import numpy as np
    import cv2

    if not height_cm and not reference_object_cm:
        raise HTTPException(400, "Provide height_cm or reference_object_cm.")

    content = await image.read()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result, mesh, betas, thetas = service.measure_with_mesh(
            image_path=tmp_path,
            height_cm=height_cm,
            reference_object_cm=reference_object_cm,
            gender_profile=gender_profile,
        )
        if mesh is None:
            raise HTTPException(500, "3D mesh generation failed.")

        from fitscan.avatar.generator import AvatarGenerator
        import uuid
        source_img = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB
        )
        gen = AvatarGenerator()
        import tempfile as tf
        with tf.TemporaryDirectory() as tmpdir:
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
            }
        )
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)
'''

def add_isolated_routes():
    """Add measurements-only and avatar-only endpoints to routes.py."""
    content = API_ROUTES.read_text()
    if "/measure/measurements" in content:
        log("Isolated routes already present in routes.py — skipping.")
        return
    # Append before the last line of the file
    content = content.rstrip() + "\n" + MEASUREMENTS_ONLY_ROUTE + "\n"
    API_ROUTES.write_text(content)
    log("Added /measure/measurements and /measure/avatar endpoints to routes.py")


# ─── Step 2: Export live OpenAPI schema from running FastAPI ──────────────────

def export_live_schema():
    """Start the FastAPI app briefly to export the live OpenAPI schema."""
    log("Exporting live OpenAPI schema from FastAPI...")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import json, sys; sys.path.insert(0, '.'); "
                "from fitscan.api.main import app; "
                "print(json.dumps(app.openapi(), indent=2))"
            ],
            capture_output=True, text=True, cwd=FITSCAN_DIR, timeout=30
        )
        if result.returncode != 0:
            warn(f"Could not export live schema: {result.stderr[:200]}")
            warn("Using pre-built openapi.json instead.")
            return False

        schema = json.loads(result.stdout)
        output = DOCS_DIR / "openapi.json"
        output.write_text(json.dumps(schema, indent=2))
        log(f"Live OpenAPI schema exported to {output}")
        return True
    except Exception as e:
        warn(f"Schema export failed: {e}. Using pre-built openapi.json.")
        return False


# ─── Step 3: Copy static docs files ──────────────────────────────────────────

def copy_docs_files():
    """Copy the pre-built HTML, Postman collection to docs/."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if pre-built files exist (placed by the zip download)
    for filename in ["index.html", "FitScan.postman_collection.json"]:
        src = FITSCAN_DIR / "docs_source" / filename
        dst = DOCS_DIR / filename
        if src.exists():
            shutil.copy(src, dst)
            log(f"Copied {filename} → docs/")
        else:
            warn(f"{filename} not found at {src}. Skipping copy.")

    # Write a minimal index.html if not present
    index = DOCS_DIR / "index.html"
    if not index.exists():
        index.write_text(
            '<!DOCTYPE html><html><head><title>FitScan API Docs</title></head>'
            '<body><div id="redoc-container"></div>'
            '<script src="https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js"></script>'
            '<script>Redoc.init("openapi.json",{},document.getElementById("redoc-container"))</script>'
            '</body></html>'
        )
        log("Created minimal index.html")


# ─── Step 4: Add docs route to FastAPI ────────────────────────────────────────

def add_docs_static_route():
    """Serve the docs directory as a static route in FastAPI."""
    main_py = FITSCAN_DIR / "fitscan" / "api" / "main.py"
    content = main_py.read_text()

    if "StaticFiles" in content and "docs" in content:
        log("Static docs route already in main.py — skipping.")
        return

    static_mount = '''
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Serve API documentation at /api-docs
_docs_dir = Path(__file__).parent.parent.parent / "docs"
if _docs_dir.exists():
    app.mount("/api-docs", StaticFiles(directory=str(_docs_dir), html=True), name="api-docs")
'''
    # Insert after app creation
    content = content.replace(
        'app.include_router(router)',
        'app.include_router(router)' + static_mount
    )
    main_py.write_text(content)
    log("API docs will be served at http://localhost:8000/api-docs")


# ─── Step 5: Print summary ────────────────────────────────────────────────────

def print_summary():
    print()
    log("Documentation generation complete.")
    print()
    print(f"  {GREEN}Local docs:{NC}       http://localhost:8000/api-docs")
    print(f"  {GREEN}Swagger UI:{NC}       http://localhost:8000/docs")
    print(f"  {GREEN}OpenAPI JSON:{NC}     http://localhost:8000/api-docs/openapi.json")
    print(f"  {GREEN}Postman import:{NC}   docs/FitScan.postman_collection.json")
    print()
    print(f"  {YELLOW}To share docs publicly:{NC}")
    print(f"    Upload docs/ folder to Netlify, Vercel, or GitHub Pages")
    print(f"    Or share openapi.json at https://editor.swagger.io")
    print()
    print(f"  {YELLOW}Postman quick start:{NC}")
    print(f"    1. Open Postman → Import → Upload docs/FitScan.postman_collection.json")
    print(f"    2. Run 'Keys / Register API Key'")
    print(f"    3. The api_key is auto-saved to collection variables")
    print(f"    4. Run any other request")
    print()


if __name__ == "__main__":
    log("Starting FitScan docs generation...")
    add_isolated_routes()
    exported = export_live_schema()
    if not exported:
        # Try to copy pre-built openapi.json if it exists
        prebuilt = FITSCAN_DIR / "docs_source" / "openapi.json"
        if prebuilt.exists():
            shutil.copy(prebuilt, DOCS_DIR / "openapi.json")
            log("Using pre-built openapi.json")
    copy_docs_files()
    add_docs_static_route()
    print_summary()
