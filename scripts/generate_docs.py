"""
FitScan API Documentation Generator
=====================================
Pure-read documentation builder. Run from the fitscan project root:

  python scripts/generate_docs.py

What it does:
  1. Imports the FastAPI app (read-only) and exports the live OpenAPI
     schema to docs/openapi.json.
  2. Copies the pre-built Redoc viewer + Postman collection from
     docs_source/ into docs/ (creates a minimal index.html if missing).
  3. Prints a quickstart summary.

What it deliberately does NOT do (audit FS-C7):
  - Append routes to fitscan/api/routes.py
  - Rewrite fitscan/api/main.py
  - Modify any committed source. Routes and the static-mount block live
    exclusively in their canonical files and are code-reviewed.

If you need to regenerate the OpenAPI surface, edit routes.py / schemas.py
and re-run this script — it observes the result, never produces it.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

FITSCAN_DIR = Path(__file__).parent.parent
DOCS_DIR = FITSCAN_DIR / "docs"

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
NC = "\033[0m"


def log(msg: str) -> None:
    print(f"{GREEN}[DOCS]{NC} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{NC} {msg}")


def err(msg: str) -> None:
    print(f"{RED}[ERROR]{NC} {msg}")


# ─── Step 1: Export live OpenAPI schema (pure read) ──────────────────────────


def export_live_schema() -> bool:
    """Spawn a subprocess to import the FastAPI app and dump its OpenAPI
    JSON. Subprocess isolation avoids polluting our process with the full
    ML stack just to read a route table.
    """
    log("Exporting live OpenAPI schema from FastAPI...")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json, sys; sys.path.insert(0, '.'); "
                "from fitscan.api.main import app; "
                "print(json.dumps(app.openapi(), indent=2))",
            ],
            capture_output=True,
            text=True,
            cwd=FITSCAN_DIR,
            timeout=30,
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


# ─── Step 2: Copy static docs files (pure read of docs_source/) ──────────────


def copy_docs_files() -> None:
    """Copy the pre-built HTML + Postman collection from docs_source/ into
    docs/. Source files come from the docs-source bundle; nothing about
    the runtime API is touched.
    """
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    for filename in ["index.html", "FitScan.postman_collection.json"]:
        src = FITSCAN_DIR / "docs_source" / filename
        dst = DOCS_DIR / filename
        if src.exists():
            shutil.copy(src, dst)
            log(f"Copied {filename} → docs/")
        else:
            warn(f"{filename} not found at {src}. Skipping copy.")

    index = DOCS_DIR / "index.html"
    if not index.exists():
        index.write_text(
            "<!DOCTYPE html><html><head><title>FitScan API Docs</title></head>"
            '<body><div id="redoc-container"></div>'
            '<script src="https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js"></script>'
            '<script>Redoc.init("openapi.json",{},document.getElementById("redoc-container"))</script>'
            "</body></html>"
        )
        log("Created minimal index.html")


# ─── Step 3: Print summary ────────────────────────────────────────────────────


def print_summary() -> None:
    print()
    log("Documentation generation complete.")
    print()
    print(f"  {GREEN}Local docs:{NC}       http://localhost:8000/api-docs")
    print(f"  {GREEN}Swagger UI:{NC}       http://localhost:8000/docs")
    print(f"  {GREEN}OpenAPI JSON:{NC}     http://localhost:8000/api-docs/openapi.json")
    print(f"  {GREEN}Postman import:{NC}   docs/FitScan.postman_collection.json")
    print()
    print(f"  {YELLOW}To share docs publicly:{NC}")
    print("    Upload docs/ folder to Netlify, Vercel, or GitHub Pages")
    print("    Or share openapi.json at https://editor.swagger.io")
    print()


if __name__ == "__main__":
    log("Starting FitScan docs generation (read-only)...")
    exported = export_live_schema()
    if not exported:
        prebuilt = FITSCAN_DIR / "docs_source" / "openapi.json"
        if prebuilt.exists():
            shutil.copy(prebuilt, DOCS_DIR / "openapi.json")
            log("Using pre-built openapi.json")
    copy_docs_files()
    print_summary()
