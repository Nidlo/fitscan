"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

app = FastAPI(
    title="FitScan Body Measurement API",
    description="AI-powered body measurement from photographs",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Serve API documentation at /api-docs
_docs_dir = Path(__file__).parent.parent.parent / "docs"
if _docs_dir.exists():
    app.mount("/api-docs", StaticFiles(directory=str(_docs_dir), html=True), name="api-docs")

