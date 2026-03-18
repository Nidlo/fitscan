# FitScan — AI Body Measurement Service

Part of the **StitchHub** monorepo. Lives at `stitchub/fitscan/`.  
Runs as a standalone Python microservice on port **8000**.  
The Laravel backend (`stitchub/backend/`) and Next.js frontend (`stitchub/frontend/`) call it over HTTP.

**Version:** 0.1.0 · **Python:** 3.11+ · **Status:** Beta

---

## What it does

Takes a photograph of a person and returns:
- 24 body measurements in centimetres (ladies/men/children profiles)
- A 3D avatar GLB file for virtual try-on
- Confidence score per measurement
- Manual override with full Inverse SMPL recalculation

Measurements match the **Ghanaian professional tailor standard** — every field a tailor
captures: bust, waist, hip, shoulder, across back, kaba length, nape-to-waist, and 20 more.

---

## Quick start (from stitchub root)

```bash
cd fitscan
make setup        # venv + packages + .env copy
make db-migrate   # run PostgreSQL migrations
make models       # download ML model weights (~2GB)
make dev          # start on http://localhost:8000
```

Interactive API docs: **http://localhost:8000/docs**

---

## Accounts & keys required

| Service | Purpose | Cost | Link |
|---------|---------|------|------|
| **Anthropic** | Claude Vision validation | ~$0.003/photo | [console.anthropic.com](https://console.anthropic.com) |
| **SMPL model** | 3D body model weights | Free (register) | [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) |

Add to `fitscan/.env`:
```env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://localhost/fitscan_dev
```

---

## How StitchHub connects to FitScan

```
stitchub/backend/   →  app/Services/FitScanService.php  →  HTTP :8000
stitchub/frontend/  →  hooks/useAvatarGlb.ts            →  HTTP :8000
```

StitchHub never imports Python code. It only calls the REST API.
See `stitchub/backend/app/Services/FitScanService.php` for the integration.

---

## API in 30 seconds

```bash
# Register for an API key
curl -X POST http://localhost:8000/api/v1/keys/register \
  -F "owner_name=StitchHub" -F "owner_email=dev@stitchub.com" -F "tier=free"

# Measure a person
curl -X POST http://localhost:8000/api/v1/measure \
  -H "X-API-Key: fsk_your_key" \
  -F "image=@photo.jpg" \
  -F "height_cm=165" \
  -F "gender_profile=ladies"

# Download the 3D avatar
curl -H "X-API-Key: fsk_your_key" \
  http://localhost:8000/api/v1/avatar/{avatar_id}/glb --output avatar.glb
```

All endpoints documented at **http://localhost:8000/docs**

---

## Versioning

Follows [Semantic Versioning](https://semver.org/). See [CHANGELOG.md](CHANGELOG.md).

---

## Deployment

FitScan deploys as a separate systemd service on the same server as StitchHub.
Both share the server — FitScan on `127.0.0.1:8000`, StitchHub on `:443`.
FitScan is **not** exposed to the internet directly.

See [OPERATIONS.md](OPERATIONS.md) for full local and server deployment guide.
