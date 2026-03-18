# CHANGELOG — FitScan

Located at `stitchub/fitscan/CHANGELOG.md`

---

## [0.1.0] — 2026-03-18

### Added

- 24-field body measurement set matching Ghanaian tailor book standard
- Ladies / men / children gender profiles
- SMPL 3D body fitting via HMR2.0 with SMPLify fallback
- MediaPipe pose detection (33 landmarks)
- Background removal via rembg
- Scale calibration from user height or reference object
- Claude Vision validation layer
- Anatomical rule-based sanity checks (ANSUR II population bounds)
- Weighted multi-source fusion with per-measurement confidence scoring
- 3D avatar GLB export with photo texture
- Three.js HTML spin viewer
- TailorNet virtual clothing fitting (blouse, skirt, dress, kaba, trousers)
- Inverse SMPL manual override with full recalculation
- FastAPI REST service — interactive docs at `/docs`
- API key authentication (free 50/mo, pro 1000/mo)
- Python SDK (`FitScanClient`)
- CLI interface (`fitscan_cli.py`)
- PostgreSQL + pgvector storage
- Object storage abstraction (local / MinIO / S3)
- Celery async task queue
- 68 tests — 54 unit, 12 integration, 2 benchmark
- GitHub Actions CI/CD with auto-deploy

### Known limitations in 0.1.0

- Children profile uses adult SMPL model (no child-specific weights yet)
- TailorNet limited to basic garment types
- Video input not yet supported
- Test coverage at 60% (target 80% for 1.0.0)

---

## Roadmap

| Version | Target | Focus |
|---------|--------|-------|
| 0.2.0 | Q2 2026 | Video input, 80% test coverage |
| 0.3.0 | Q3 2026 | Child SMPL model, more garments |
| 1.0.0 | Q4 2026 | Stable API, production hardened |
