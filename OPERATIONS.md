# OPERATIONS — FitScan

**Location in monorepo:** `stitchub/fitscan/`  
**Runs on:** port 8000  
**Talks to:** `stitchub/backend/` (Laravel) and `stitchub/frontend/` (Next.js)

---

## CONTENTS

1. Running locally alongside StitchHub
2. Environment files
3. API endpoints — what to call, what you get back
4. Testing guide
5. Deploying to a server
6. Database migrations
7. Accounts & keys
8. Nginx + systemd on the server
9. Updating the server
10. Troubleshooting

---

## 1. RUNNING LOCALLY ALONGSIDE STITCHUB

You run FitScan and StitchHub at the same time in VS Code. Open two terminal panels.

**Terminal 1 — FitScan:**
```bash
cd fitscan
make dev
# Running on http://localhost:8000
# Docs at  http://localhost:8000/docs
```

**Terminal 2 — StitchHub backend:**
```bash
cd backend
php artisan serve
# Running on http://localhost:8001
```

**Terminal 3 — StitchHub frontend:**
```bash
cd frontend
npm run dev
# Running on http://localhost:3000
```

Laravel calls FitScan at `http://localhost:8000`.
Next.js fetches GLB files directly from `http://localhost:8000`.

### First-time setup for FitScan

```bash
cd fitscan

# 1. Install Python dependencies
make setup

# 2. Install and start PostgreSQL locally
#    Mac:   brew install postgresql@15 && brew services start postgresql@15
#    Linux: sudo apt install postgresql-15 && sudo service postgresql start

# 3. Create the databases
createdb fitscan_dev
createdb fitscan_test
psql -d fitscan_dev -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -d fitscan_test -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 4. Run migrations
make db-migrate

# 5. Download ML models (~2GB, takes 5-10 min)
make models

# 6. Start
make dev
```

---

## 2. ENVIRONMENT FILES

```
fitscan/.env           ← your local config (gitignored — never commit)
fitscan/.env.test      ← test config (committed — no real keys)
fitscan/.env.example   ← template documenting every variable (committed)
```

### Minimum required in fitscan/.env

```env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://localhost/fitscan_dev
APP_ENV=development
```

### Full .env.example

```env
# ── Core ──────────────────────────────────────────────
ANTHROPIC_API_KEY=your_key_here
APP_ENV=development          # development | test | production
LOG_LEVEL=INFO
API_PORT=8000

# ── Database ──────────────────────────────────────────
DATABASE_URL=postgresql://localhost/fitscan_dev

# ── Object storage (avatars/GLB files) ────────────────
STORAGE_BACKEND=local        # local | minio | s3

# ── ML models ─────────────────────────────────────────
SMPL_MODEL_PATH=data/models/smpl/SMPL_NEUTRAL.pkl
HMR2_CHECKPOINT_PATH=data/models/hmr2/epoch=35-step=1000000.ckpt

# ── Feature flags ─────────────────────────────────────
ENABLE_CLAUDE_VALIDATION=true
ENABLE_SMPL_FITTING=true
ENABLE_AVATAR_GENERATION=true
CONFIDENCE_THRESHOLD=0.65
MAX_IMAGE_SIZE_MB=10

# ── SaaS tiers ────────────────────────────────────────
FREE_TIER_MONTHLY_LIMIT=50
PRO_TIER_MONTHLY_LIMIT=1000

# ── Internal key (StitchHub backend uses this) ────────
FITSCAN_INTERNAL_KEY=generate_with_openssl_rand_hex_32
```

### What to add in stitchub/backend/.env

```env
FITSCAN_API_URL=http://127.0.0.1:8000
FITSCAN_INTERNAL_KEY=same_value_as_fitscan_env
```

### What to add in stitchub/frontend/.env.local

```env
NEXT_PUBLIC_FITSCAN_URL=http://localhost:8000
```

---

## 3. API ENDPOINTS

All endpoints require `X-API-Key: fsk_your_key` header (except `/health` and `/keys/register`).

### Get an API key

```bash
curl -X POST http://localhost:8000/api/v1/keys/register \
  -F "owner_name=StitchHub Dev" \
  -F "owner_email=dev@stitchub.com" \
  -F "tier=free"
```

Response:
```json
{ "api_key": "fsk_abc123...", "tier": "free", "monthly_limit": 50 }
```

Store this key. Use it in the `X-API-Key` header for all other requests.

---

### POST /api/v1/measure — submit a photo

**Request (multipart/form-data):**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `image` | file | YES | JPEG/PNG, min 720p, full body in frame |
| `height_cm` | float | YES* | Person's height, e.g. `165.0` |
| `reference_object_cm` | float | NO* | A4 = 29.7, if held in photo |
| `gender_profile` | string | NO | `ladies` (default) · `men` · `children` |
| `enable_smpl` | bool | NO | `true` (default) — set `false` for speed |
| `enable_claude` | bool | NO | `true` (default) — set `false` to skip validation |
| `generate_avatar` | bool | NO | `true` (default) — set `false` to skip GLB |

*One of `height_cm` or `reference_object_cm` is required.

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/measure \
  -H "X-API-Key: fsk_your_key" \
  -F "image=@photo.jpg" \
  -F "height_cm=165" \
  -F "gender_profile=ladies"
```

**Response:**
```json
{
  "status": "success",
  "avatar_id": "3f8a2b1c-4d5e-...",
  "gender_profile": "ladies",

  "bust_cm":   { "value_cm": 92.4, "confidence": 0.88, "source": "smpl" },
  "waist_cm":  { "value_cm": 71.2, "confidence": 0.85, "source": "smpl" },
  "hip_cm":    { "value_cm": 97.8, "confidence": 0.87, "source": "smpl" },
  "shoulder_cm": { "value_cm": 38.1, "confidence": 0.84, "source": "smpl" },
  "across_back_cm":    { "value_cm": 34.2, "confidence": 0.80, "source": "smpl" },
  "across_chest_cm":   { "value_cm": 32.8, "confidence": 0.80, "source": "silhouette" },
  "shoulder_to_waist_cm": { "value_cm": 38.9, "confidence": 0.86, "source": "smpl" },
  "shoulder_to_hip_cm":   { "value_cm": 56.4, "confidence": 0.85, "source": "smpl" },
  "nape_to_waist_cm":     { "value_cm": 40.2, "confidence": 0.78, "source": "smpl" },
  "sleeve_length_cm":     { "value_cm": 58.7, "confidence": 0.80, "source": "smpl" },
  "neck_cm":    { "value_cm": 33.4, "confidence": 0.87, "source": "smpl" },
  "wrist_cm":   { "value_cm": 15.1, "confidence": 0.80, "source": "smpl" },
  "height_cm":  { "value_cm": 165.0, "confidence": 0.92, "source": "smpl" },
  "inseam_cm":  { "value_cm": 74.3, "confidence": 0.82, "source": "smpl" },
  "blouse_length_cm": { "value_cm": 56.4, "confidence": 0.85, "source": "smpl" },
  "kaba_length_cm":   { "value_cm": 86.4, "confidence": 0.75, "source": "smpl" },
  "skirt_length_cm":  { "value_cm": 90.7, "confidence": 0.78, "source": "smpl" },
  "dress_length_cm":  { "value_cm": 156.8, "confidence": 0.72, "source": "smpl" }
}
```

**Confidence scores:**

| Range | Meaning |
|-------|---------|
| 0.85 – 1.0 | Use directly |
| 0.70 – 0.85 | Good, verify for bespoke tailoring |
| 0.50 – 0.70 | Moderate — consider retaking photo |
| Below 0.50 | Low — retake photo |

---

### GET /api/v1/avatar/{id}/glb — download 3D avatar

Returns binary GLB file. Use directly in a Three.js or React Three Fiber viewer.

```bash
curl -H "X-API-Key: fsk_your_key" \
  http://localhost:8000/api/v1/avatar/3f8a2b1c-.../glb \
  --output avatar.glb
```

In Next.js — fetch directly (do NOT go through GraphQL for binary):
```typescript
const r = await fetch(`${process.env.NEXT_PUBLIC_FITSCAN_URL}/api/v1/avatar/${id}/glb`,
  { headers: { 'X-API-Key': process.env.NEXT_PUBLIC_FITSCAN_KEY } })
const blob = await r.blob()
const url = URL.createObjectURL(blob)
// pass url to <Canvas> / drei <useGLTF>
```

---

### POST /api/v1/avatar/{id}/override — correct a measurement

When tailor says "wrist should be 10cm not 15cm" — send the correction and FitScan
recalculates all measurements using Inverse SMPL and regenerates the GLB.

```bash
curl -X POST http://localhost:8000/api/v1/avatar/3f8a2b1c-.../override \
  -H "X-API-Key: fsk_your_key" \
  -H "Content-Type: application/json" \
  -d '{"overrides": {"wrist_cm": 10.0, "waist_cm": 72.0}}'
```

Response returns all recalculated measurements and the new avatar GLB is stored.

---

### POST /api/v1/avatar/{id}/fit-garment — virtual clothing

```bash
curl -X POST http://localhost:8000/api/v1/avatar/3f8a2b1c-.../fit-garment \
  -H "X-API-Key: fsk_your_key" \
  -F "garment_type=blouse"
# Returns: binary GLB of avatar wearing the garment
```

Supported: `blouse`, `dress`, `kaba`, `skirt`, `trousers`

---

### Expected response times

| Operation | Approx time |
|-----------|------------|
| Health check | < 50ms |
| Measure (SMPL off) | 3 – 8s |
| Measure (SMPL on) | 20 – 40s |
| Download GLB | < 200ms |
| Manual override | 15 – 30s |
| Fit garment | 10 – 20s |

---

## 4. TESTING GUIDE

### Unit tests (fast, no running services needed)

```bash
cd fitscan
make test-unit
# Runs in ~30 seconds
# Tests: scale calibration, measurement fusion, confidence scoring, rule validation
```

### Integration tests (needs DB)

```bash
cd fitscan
make db-migrate
make test-integration
```

### Smoke test (needs make dev running)

```bash
# Terminal 1
cd fitscan && make dev

# Terminal 2
cd fitscan && make test-smoke
```

This hits real endpoints and checks that the API responds correctly.

### All tests + coverage

```bash
cd fitscan
make test
# Must reach 55% coverage to pass (target 80% for 1.0)
```

---

## 5. DEPLOYING TO A SERVER

FitScan and StitchHub deploy to the same server. FitScan runs as a separate
systemd service on `127.0.0.1:8000`. Only StitchHub can reach it — port 8000
is not exposed to the internet.

### One-command install (Ubuntu 22.04)

```bash
# From your local machine
scp deploy/install.sh user@your-server:/tmp/
ssh user@your-server "sudo bash /tmp/install.sh"
```

### What the installer does automatically

1. Installs Python 3.11, PostgreSQL 15, pgvector, Redis, Nginx
2. Creates `fitscan` system user
3. Creates PostgreSQL databases with a generated password
4. Sets up systemd service (`fitscan.service`)
5. Configures Nginx reverse proxy
6. Optionally installs SSL via Let's Encrypt

### After install — add your keys

```bash
ssh fitscan@your-server
nano /home/fitscan/app/.env
# Set: ANTHROPIC_API_KEY, APP_ENV=production
```

Then download the SMPL model manually (requires free registration):
1. Register at https://smpl.is.tue.mpg.de/
2. Download `SMPL_python_v.1.1.0.zip`
3. Copy `SMPL_NEUTRAL.pkl` to `/home/fitscan/app/data/models/smpl/`

Then restart:
```bash
sudo systemctl restart fitscan
curl http://localhost:8000/api/v1/health
```

---

## 6. DATABASE MIGRATIONS

```bash
cd fitscan

make db-migrate          # apply all pending migrations
make db-rollback         # roll back one step
make db-status           # show current version
make db-create-migration NAME=add_new_field
```

On the server, always back up before migrating:
```bash
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M).sql
alembic upgrade head
```

---

## 7. ACCOUNTS & KEYS

### Must have before running

**Anthropic API key** — https://console.anthropic.com  
Create a project → API Keys → New key  
Add to `fitscan/.env`: `ANTHROPIC_API_KEY=sk-ant-...`

**SMPL body model** — https://smpl.is.tue.mpg.de/  
Free registration. Download `SMPL_python_v.1.1.0.zip`.  
Extract `SMPL_NEUTRAL.pkl` → place at `fitscan/data/models/smpl/SMPL_NEUTRAL.pkl`

### For production (optional)

**AWS S3 or Cloudflare R2** — for storing GLB files  
R2 is free up to 10GB: https://dash.cloudflare.com → R2

**GitHub repo secrets** — for CI/CD auto-deploy:
- `ANTHROPIC_API_KEY`
- `DEPLOY_HOST` (server IP)
- `DEPLOY_USER` (SSH username)
- `DEPLOY_SSH_KEY` (your private SSH key)

---

## 8. NGINX + SYSTEMD ON THE SERVER

FitScan's systemd service file (created by `install.sh`):
```
/etc/systemd/system/fitscan.service
```

Nginx config (created by `install.sh`):
```
/etc/nginx/sites-available/fitscan
```

The Nginx config for the full StitchHub + FitScan setup:

```nginx
# StitchHub main app (public)
server {
    listen 443 ssl;
    server_name app.yourdomain.com;
    location / { proxy_pass http://127.0.0.1:9000; }
}

# FitScan API docs (optional — only if exposing to third-party devs)
server {
    listen 443 ssl;
    server_name api.yourdomain.com;
    client_max_body_size 15M;
    proxy_read_timeout 90s;
    location / { proxy_pass http://127.0.0.1:8000; }
}
```

If FitScan is internal-only (StitchHub consumes it privately), skip the second block.

---

## 9. UPDATING THE SERVER

```bash
# From your local machine
ssh fitscan@your-server "cd /home/fitscan/app && make server-deploy"
```

Or push to GitHub and let CI/CD handle it automatically.

---

## 10. TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| `No pose detected` | Full body must be in frame, good lighting |
| `ModelNotFoundError: SMPL_NEUTRAL.pkl` | Download from smpl.is.tue.mpg.de manually |
| `connection refused :5432` | PostgreSQL not running — `brew services start postgresql@15` |
| Measure takes > 60s | Set `enable_smpl=false` in request for speed |
| `403 Invalid API key` | Register first: `POST /api/v1/keys/register` |
| Tests fail with wrong DB | Check `fitscan/.env.test` has `APP_ENV=test` |
| GLB won't load in browser | Use `fitscan/cli/avatar_viewer.html` — pins Three.js r128 |
| `ANTHROPIC_API_KEY not set` | Add to `fitscan/.env` and restart |
