# Fitscan server setup

Per-server runbook for getting Fitscan **fully working** on a fresh box — not just running, but with all features (Claude vision validation, SMPL body fitting, avatar generation) actually enabled.

If you're standing up a brand-new server, follow [`warp/docs/deploy-runbook.md`](https://github.com/Nidlo/warp/blob/development/docs/deploy-runbook.md) Phase B first to provision the OS / Python / Postgres / systemd, then come back here for the env audit + ML model dance.

---

## 1. Environment variables — full reference

The canonical list lives in [`.env.example`](../.env.example). Below is every variable, what it does, the safe default, and **when** you should put a real value in.

### API auth

| Var | Purpose | Default | When to set |
|---|---|---|---|
| `FITSCAN_REQUIRE_API_KEY` | Gate `/api/v1/measure` + friends behind `X-API-Key` | `true` (prod) / `false` (tests) | Always `true` on production servers. CI overrides via the workflow env. |
| `FITSCAN_ADMIN_API_KEY` | Gate `/api/v1/admin/*` + `/api/v1/keys/register` behind `X-Admin-API-Key` | empty (= admin endpoints unusable) | **Set on every server.** Mint with `openssl rand -hex 32`; same value must be set on warp's `.env` (`FITSCAN_ADMIN_API_KEY=`). |
| `FITSCAN_ALLOW_UNAUTHED_LOCALHOST` | DEV-ONLY bypass: loopback callers (127.0.0.1 / ::1) skip admin auth | `false` | Leave **false** on production. Only flip for local dashboard wiring. |
| `FITSCAN_API_KEY_SECRET` | Reserved for future HMAC-signed key issuance — currently unused | empty | Skip until a feature lands. |

### Anthropic / Claude vision

| Var | Purpose | Default | When to set |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | API key from console.anthropic.com — used for Claude-vision landmark verification + style classification | empty | **Set before flipping `ENABLE_CLAUDE_VALIDATION=true`**. Doc: see Section 3. |

### Database

| Var | Purpose | Default | When to set |
|---|---|---|---|
| `DATABASE_URL` | Postgres DSN. `postgresql://USER:PASS@HOST:PORT/DB` | sqlite (dev) | **Always set on the server** to the local postgres URL. The `nidlo-db-creds` file written by Phase B of the deploy runbook has the credentials. |
| `TEST_DATABASE_URL` | Same shape, separate DB for tests | empty | CI only — set by the workflow. |

### Object storage

| Var | Purpose | Default | When to set |
|---|---|---|---|
| `STORAGE_BACKEND` | `local` / `minio` / `s3` | `local` | Beta = `local` (writes to /home/fitscan/app/uploads). Once traffic grows or you need durability, switch to `s3` + populate the AWS_* keys. |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` / `MINIO_BUCKET` | Local MinIO dev cluster | defaults work locally | Skip on production unless running a MinIO-on-server pattern. |
| `S3_BUCKET` / `S3_REGION` | Real S3 / Backblaze B2 target | empty / `us-east-1` | Only when `STORAGE_BACKEND=s3`. |

### Queue (Celery)

| Var | Purpose | Default | When to set |
|---|---|---|---|
| `CELERY_BROKER_URL` | Redis URL for Celery tasks (avatar generation runs async) | `redis://127.0.0.1:6379/0` | Already set by deploy-runbook Phase B. Verify it matches the server's Redis. |
| `CELERY_RESULT_BACKEND` | Where Celery stores task results | `redis://127.0.0.1:6379/1` | Same Redis instance, different logical DB. |

### ML models

| Var | Purpose | Default | When to set |
|---|---|---|---|
| `SMPL_MODEL_PATH` | Path to SMPL_NEUTRAL.pkl (and optionally _MALE / _FEMALE) | `data/models/smpl/SMPL_NEUTRAL.pkl` | Path is fine as-is; the .pkl file itself needs manual download. See Section 2. |
| `HMR2_CHECKPOINT_PATH` | Path to HMR2 pose-estimation checkpoint | `data/models/hmr2/epoch=35-step=1000000.ckpt` | Same — path fine, file needs download. See Section 2.2. |

### Feature flags — the big ones

| Var | Purpose | Default | When to set |
|---|---|---|---|
| `ENABLE_CLAUDE_VALIDATION` | Use Claude vision to verify low-confidence landmarks | `true` | Requires `ANTHROPIC_API_KEY` set. Leave `false` until the key is provisioned. |
| `ENABLE_SMPL_FITTING` | Fit a 3D SMPL body to detected landmarks → real circumferences | `true` | Requires SMPL .pkl files on disk. Leave `false` until downloaded. |
| `ENABLE_AVATAR_GENERATION` | Export a textured GLB avatar from the fitted body | `true` | Requires both SMPL + HMR2 assets. Leave `false` until both are in place. |
| `CONFIDENCE_THRESHOLD` | Landmark visibility threshold below which Claude vision is consulted | `0.65` | Tune as you collect data. Lower = more Claude calls = better accuracy but higher cost. |
| `MAX_IMAGE_SIZE_MB` | Reject uploads larger than this | `10` | Bump if you accept high-res phone photos (modern iPhones can hit 4-5MB at default JPEG quality). |

### Server / runtime

| Var | Purpose | Default | When to set |
|---|---|---|---|
| `APP_ENV` | `development` / `staging` / `production` | `development` | Always set on the server to match env. |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO` | Production = `INFO`. Staging can be `DEBUG` for richer logs. |
| `API_PORT` | Uvicorn bind port | `8000` | Already pinned by systemd unit. |

### SaaS / quotas

| Var | Purpose | Default | When to set |
|---|---|---|---|
| `FREE_TIER_MONTHLY_LIMIT` | Quota for free-tier API keys | `50` | Override if your beta cohort has higher needs. |
| `PRO_TIER_MONTHLY_LIMIT` | Quota for pro-tier keys | `1000` | Same. |
| `STRIPE_SECRET_KEY` | Stripe — for paid-tier billing | empty | Skip until billing ships. |

### Email (Resend)

| Var | Purpose | Default | When to set |
|---|---|---|---|
| `RESEND_API_KEY` | Resend API key (separate sender identity from warp's Resend setup) | empty | Once provisioned, set + verify DKIM on the Resend domain matching `RESEND_FROM_ADDRESS`. |
| `RESEND_FROM_ADDRESS` | Sender address for fitscan's transactional emails | `noreply@nidlo.test` | Update to a real verified sender. |
| `RESEND_FROM_NAME` | Display name | `Nidlo Fitscan` | Optional. |

### Observability

| Var | Purpose | Default | When to set |
|---|---|---|---|
| `SENTRY_DSN` | Error reporting (`sentry-sdk[fastapi]` already wired) | empty (no-op) | Set once a Sentry project exists. Until then it's silent. |
| `SENTRY_TRACES_SAMPLE_RATE` | What fraction of requests get APM-traced | `0.1` | Leave at `0.1` (10%) on beta. Lower in prod to reduce Sentry cost. |

### CORS

| Var | Purpose | Default | When to set |
|---|---|---|---|
| `CORS_ALLOWED_ORIGINS` | Comma-separated origins allowed to call fitscan directly from a browser | `http://localhost:8000` | Set to your weft origins, e.g. `https://nidlo.app,https://staging.nidlo.app,http://localhost:3000`. |

### Upload temp storage

| Var | Purpose | Default | When to set |
|---|---|---|---|
| `UPLOAD_TEMPDIR` | Where multipart bodies land while being processed | `/tmp/fitscan_uploads` | OK as-is. Override if `/tmp` is too small. |
| `UPLOAD_TEMPDIR_MAX_AGE_SECONDS` | TTL before stale uploads get garbage-collected | `3600` (1h) | Fine. |

---

## 2. ML model setup (license-gated, manual)

### 2.1 SMPL — body shape model

Required for `ENABLE_SMPL_FITTING=true`. The SMPL `.pkl` files are license-gated by Max-Planck-Gesellschaft — you must register before downloading.

```text
1. Go to https://smpl.is.tue.mpg.de
2. Register (free) → confirm email → log in
3. Downloads page → "SMPL for Python users — Version 1.1.0 (10 shape PCs)"
   (~100 MB zipped)
4. Unzip the archive locally
5. Rename + transfer to the server:
     basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl  →  SMPL_NEUTRAL.pkl
6. (Optional) MALE + FEMALE variants for gender-specific fits:
     basicmodel_m_lbs_10_207_0_v1.1.0.pkl  →  SMPL_MALE.pkl
     basicmodel_f_lbs_10_207_0_v1.1.0.pkl  →  SMPL_FEMALE.pkl
```

Server-side, the SMPL directory must exist before scp:

```bash
ssh -i ~/.ssh/nidlo_deploy deploy@<FITSCAN_HOST> \
  'mkdir -p /home/fitscan/app/data/models/smpl'

# Transfer (from your local machine — adjust ~/Downloads/smpl/ to wherever you unzipped):
scp -i ~/.ssh/nidlo_deploy \
  ~/Downloads/smpl/SMPL_NEUTRAL.pkl \
  ~/Downloads/smpl/SMPL_MALE.pkl \
  ~/Downloads/smpl/SMPL_FEMALE.pkl \
  deploy@<FITSCAN_HOST>:/home/fitscan/app/data/models/smpl/

# Verify
ssh -i ~/.ssh/nidlo_deploy deploy@<FITSCAN_HOST> \
  'ls -lh /home/fitscan/app/data/models/smpl/'
```

Then flip the flag + restart:

```bash
# On the server as root
sed -i 's|^ENABLE_SMPL_FITTING=.*|ENABLE_SMPL_FITTING=true|' /home/fitscan/app/.env
sudo -n /usr/bin/systemctl restart fitscan
sleep 30  # cold start (PyTorch + MediaPipe + SMPL = ~25s)
journalctl -u fitscan -n 20 --no-pager
```

**Look for** `INFO ... SMPL model loaded` (or similar) in the journal. If you see `FileNotFoundError ... SMPL_NEUTRAL.pkl`, the file isn't at the path that `SMPL_MODEL_PATH` points to.

### 2.2 HMR2 — pose estimation checkpoint

Required for `ENABLE_AVATAR_GENERATION=true`. HMR2 ("Humans in 4D") estimates 3D pose from a single image; SMPL then fits the body to that pose.

HMR2 checkpoint isn't license-gated the same way SMPL is — it's a public GitHub release from the [`shubham-goel/4D-Humans`](https://github.com/shubham-goel/4D-Humans) project:

```bash
# Default path expected by fitscan/config.py
DEST=/home/fitscan/app/data/models/hmr2/epoch=35-step=1000000.ckpt
mkdir -p "$(dirname "$DEST")"

# Download (~600 MB) — link from the 4D-Humans repo's releases page:
#   https://github.com/shubham-goel/4D-Humans/releases
# Substitute the actual release URL when you check it:
curl -L -o "$DEST" \
  "<paste the actual release-asset URL here>"

# Verify size (~600 MB)
ls -lh "$DEST"
```

Then enable + restart:

```bash
sed -i 's|^ENABLE_AVATAR_GENERATION=.*|ENABLE_AVATAR_GENERATION=true|' /home/fitscan/app/.env
sudo -n /usr/bin/systemctl restart fitscan
sleep 35
journalctl -u fitscan -n 30 --no-pager
```

Memory note: with both SMPL + HMR2 loaded, fitscan's RSS jumps from ~400 MB to ~1.5–2 GB. If your server has less than 4 GB RAM, run only one feature at a time, or upgrade the box.

---

## 3. Anthropic API key (Claude vision validation)

```text
1. Go to https://console.anthropic.com
2. Generate an API key (or pick an existing one — fitscan can share with other Anthropic-using services)
3. Copy the key (starts with `sk-ant-`)
```

Then on the server as root:

```bash
# Replace <KEY> with the value you copied
sed -i 's|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=<KEY>|' /home/fitscan/app/.env
sed -i 's|^ENABLE_CLAUDE_VALIDATION=.*|ENABLE_CLAUDE_VALIDATION=true|' /home/fitscan/app/.env

sudo -n /usr/bin/systemctl restart fitscan
sleep 30

# Smoke test: ask fitscan to verify a landmark — should not error out on auth
journalctl -u fitscan -n 20 --no-pager | grep -i anthropic
```

If you see `anthropic.AuthenticationError`, the key is wrong or the Anthropic billing isn't set up. Visit the Anthropic console → API Keys page.

---

## 4. End-to-end audit

Once you've done sections 2 + 3, run this on the server as root to confirm everything is wired:

```bash
echo "=== fitscan .env audit ==="
grep -E '^[A-Z_]+=' /home/fitscan/app/.env | awk -F= '{
    k=$1; v=substr($0, length(k)+2)
    if (v == "") print "  EMPTY: " k
    else if (length(v) > 40) print "  SET  : " k " = " substr(v,1,8) "...(masked)"
    else print "  SET  : " k " = " v
}' | sort

echo ""
echo "=== model files ==="
ls -lh /home/fitscan/app/data/models/smpl/ 2>/dev/null || echo "  (no SMPL dir)"
ls -lh /home/fitscan/app/data/models/hmr2/ 2>/dev/null || echo "  (no HMR2 dir)"

echo ""
echo "=== service ==="
systemctl is-active fitscan
ss -tlnp | grep ':8000 ' | head -1

echo ""
echo "=== healthcheck ==="
curl -sI http://127.0.0.1:8000/api/v1/health | head -2
```

What "fully working" looks like:

- All variables from `.env.example` present (none `EMPTY`)
- SMPL_NEUTRAL.pkl + (optional) SMPL_MALE.pkl + SMPL_FEMALE.pkl in `data/models/smpl/`
- HMR2 checkpoint in `data/models/hmr2/`
- `ENABLE_CLAUDE_VALIDATION=true`, `ENABLE_SMPL_FITTING=true`, `ENABLE_AVATAR_GENERATION=true`
- systemd unit `active (running)`
- Port 8000 listener present
- Healthcheck returns `200 OK`

---

## 5. Pitfalls

### Cold-start RAM spike

First request after a restart loads PyTorch + MediaPipe + SMPL + (HMR2) into one worker. RSS jumps from ~400 MB idle → ~1.5–2 GB with all features on. If the box only has 2 GB RAM, the worker may OOM-kill. Fixes:

- Run with `--workers 1` (systemd unit already does this)
- Disable HMR2 until you upgrade RAM (`ENABLE_AVATAR_GENERATION=false`)
- Add a swapfile as a band-aid (not for sustained traffic)

### `Base.metadata.create_all` racing with alembic

`fitscan/infrastructure/result_store.py` calls `Base.metadata.create_all(engine)` at module import. On a fresh server **without** `alembic stamp head`, you may see `DuplicateTable` errors during the first deploy. Stamp once per DB:

```bash
ssh -i ~/.ssh/nidlo_deploy deploy@<FITSCAN_HOST> \
  'cd /home/fitscan/app && venv/bin/alembic stamp head'
```

See [`warp/docs/deploy-runbook.md`](https://github.com/Nidlo/warp/blob/development/docs/deploy-runbook.md) Phase C for full context.

### `.env.test` vs `.env` priority

`alembic/env.py` loads `.env` first, falls back to `.env.test`. **Don't reverse the order** — CI workflows rely on `.env.test` being the secondary; production servers ship with both files and `.env` must win.

### MAX_IMAGE_SIZE_MB and Nginx

Even if fitscan accepts 10 MB uploads, nginx in front of it has its own `client_max_body_size` (default 1 MB on stock nginx). The deploy runbook's nginx snippets bump it to 50 MB. If you're running fitscan behind a different reverse proxy, mirror that.

### Restart vs reload

`systemctl reload fitscan` doesn't reload `.env` — there's no SIGHUP handler in uvicorn. **Always `restart`** after `.env` changes.

### Tier limits

`FREE_TIER_MONTHLY_LIMIT` / `PRO_TIER_MONTHLY_LIMIT` / `ENTERPRISE_TIER_MONTHLY_LIMIT` are read at key-mint time and stored in the `api_keys` row's `monthly_limit`. Changing the env var later doesn't retroactively update existing keys — you'd `UPDATE api_keys SET monthly_limit = ... WHERE tier = ...` directly.

### Rotating FITSCAN_ADMIN_API_KEY

The admin key must match between fitscan and warp. Rotate both at once:

```bash
NEW=$(openssl rand -hex 32)

# On the fitscan server:
sed -i "s|^FITSCAN_ADMIN_API_KEY=.*|FITSCAN_ADMIN_API_KEY=fsk_admin_${NEW}|" /home/fitscan/app/.env
sudo -n /usr/bin/systemctl restart fitscan

# On the warp server (both staging + production checkouts):
sed -i "s|^FITSCAN_ADMIN_API_KEY=.*|FITSCAN_ADMIN_API_KEY=fsk_admin_${NEW}|" /var/www/nidlo/warp-staging/.env
sed -i "s|^FITSCAN_ADMIN_API_KEY=.*|FITSCAN_ADMIN_API_KEY=fsk_admin_${NEW}|" /var/www/nidlo/warp-production/.env
su - deploy -c 'cd /var/www/nidlo/warp-staging && php artisan config:cache'
su - deploy -c 'cd /var/www/nidlo/warp-production && php artisan config:cache'
```

The CLI tool `python -m fitscan.cli set-admin-key <KEY>` automates the fitscan side of this.

---

## Quick-reference cheat sheet

| Want to... | Run |
|---|---|
| Add Anthropic key + enable Claude vision | sed `ANTHROPIC_API_KEY=...` + `ENABLE_CLAUDE_VALIDATION=true` + restart |
| Enable SMPL fitting | mkdir + scp .pkl files + sed `ENABLE_SMPL_FITTING=true` + restart |
| Enable avatar generation | Above + HMR2 checkpoint + sed `ENABLE_AVATAR_GENERATION=true` + restart |
| Rotate admin key | sed on fitscan + both warp envs + restart fitscan + re-cache warp |
| Audit env state | Run the awk block in Section 4 |
| Check what fitscan thinks | `journalctl -u fitscan -n 50 --no-pager` |

---

**Last updated:** 2026-05-12
