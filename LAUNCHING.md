# Launching Fitscan — hosting, integration, pricing, monetisation

> **Audience:** Snad + future ops team.
> **Purpose:** the single document that ties together "what is Fitscan", "how do I get it on the server", "how does my Laravel backend talk to it", and "how do I make money from it as a SaaS product".
>
> Companion to:
> - [README.md](README.md) — 30-second product pitch + dev quick-start
> - [OPERATIONS.md](OPERATIONS.md) — local + server runbook (env vars, systemd, nginx, troubleshooting)
> - [TASKS.md](TASKS.md) — granular task ledger
> - [ROADMAP.md](ROADMAP.md) — exec summary of what's in code today
>
> This doc covers the **product + business + integration** angle that those don't.

---

## Table of contents

1. [What Fitscan actually is — three ways it gets used](#1-what-fitscan-actually-is)
2. [Hosting it on the server (Snad checklist)](#2-hosting-it-on-the-server)
3. [Linking it to the Nidlo backend](#3-linking-it-to-the-nidlo-backend)
4. [Pricing model — tiers, limits, costs](#4-pricing-model)
5. [Unit economics — what each Fitscan call costs us](#5-unit-economics)
6. [The customer signup flow (today vs target)](#6-the-customer-signup-flow)
7. [Public landing page brief — what fitscan.com should pitch](#7-public-landing-page-brief)
8. [Profitability path — tokens, Stripe, billing](#8-profitability-path)
9. [Pre-launch checklist](#9-pre-launch-checklist)

---

## 1. What Fitscan actually is

Fitscan is a **Python FastAPI service** that takes a photo of a person and returns:

- **24 body measurements in centimetres** (bust, waist, hip, shoulder, kaba length, sleeve, etc. — the full Ghanaian-tailor field set)
- **A 3D body avatar** (GLB file, viewable in any glTF viewer or three.js)
- **Per-field confidence scores** (so the UI can flag low-confidence fields for manual override)
- **Layered AI verification**: rule-based bounds check + Claude vision verification of occluded landmarks + per-field validation (FS-NIDLO-VALID-03 surfaces disagreement)

**The same Fitscan runtime serves three audiences:**

| Audience | How they reach it | Auth | Pricing |
|---|---|---|---|
| **Nidlo's own users** (clients capturing measurements for their orders) | Laravel backend → HTTP → Fitscan, keyed by an internal admin key | `FITSCAN_INTERNAL_API_KEY` on the Laravel side | Bundled — no per-call charge to the client |
| **Other fashion businesses** (SaaS customers) | Direct HTTP `POST /api/v1/measure` from their own backend, keyed per-tenant | `X-API-Key: fsk_…` minted by `/keys/register` | Tier-based (free/pro/enterprise) — see [§4](#4-pricing-model) |
| **Curious developers / "show me what it does"** | Frontend playground at `fitscan.nidlo.com` (TBD) — drag a photo in, see measurements + GLB | None (rate-limited free trial), or bring-your-own free-tier API key | Free with a hard ceiling (e.g. 10 lifetime calls) |

The service code is the same for all three; the difference is **which API key the request carries** and **who pays for it**.

---

## 2. Hosting it on the server

### One-time deploy (Ubuntu 22.04 VPS — Contabo, RunPod, Hetzner, Linode all work)

1. **Provision the box.** Minimum sane spec for CPU-only beta:
   - 4 vCPU, 8 GB RAM, 60 GB SSD, Ubuntu 22.04 LTS
   - **Snad's current Contabo VPS is already this size** — Fitscan and the Laravel backend share it
   - For public launch, see [§5](#5-unit-economics) — you'll likely want a GPU node for inference

2. **Run the installer** (already exists at [`deploy/install.sh`](deploy/install.sh)):
   ```bash
   ssh user@your-server
   git clone https://github.com/snad/stitchub.git nidlo
   cd nidlo/fitscan
   sudo ./deploy/install.sh
   ```
   The installer handles: Python 3.11, Postgres, Redis, venv, model downloads, systemd service, nginx config. See [OPERATIONS.md §5](OPERATIONS.md#5-deploying-to-a-server) for the full step list.

3. **Provision env secrets** in `/etc/fitscan/.env` (NOT in git):
   ```bash
   # ── Core ──
   APP_ENV=production
   ANTHROPIC_API_KEY=sk-ant-…              # https://console.anthropic.com → API Keys
   DATABASE_URL=postgresql://fitscan:<pw>@127.0.0.1:5432/fitscan?sslmode=require
   FITSCAN_DATA_DIR=/var/lib/fitscan/data
   FITSCAN_REQUIRE_API_KEY=true            # always true in prod

   # ── Auth gates ──
   FITSCAN_ADMIN_API_KEY=<random 64 hex>   # for /api/v1/admin/* endpoints
   FITSCAN_INTERNAL_API_KEY=<random 64 hex> # the one Laravel uses

   # ── Storage (FS-NIDLO-MEAS-03) ──
   FITSCAN_CORS_ORIGINS=https://api.nidlo.com,https://nidlo.com
   MEASUREMENTS_PHOTO_DISK=local           # or 's3' for Backblaze later

   # ── Observability (FS-NIDLO-OPS-01) ──
   SENTRY_DSN=https://…@sentry.io/…        # optional but strongly recommended

   # ── SaaS tier limits (FS-NIDLO-AUTH-05) ──
   FREE_TIER_MONTHLY_LIMIT=50              # tweak without code change
   PRO_TIER_MONTHLY_LIMIT=1000
   ENTERPRISE_TIER_MONTHLY_LIMIT=999999
   ```
   `openssl rand -hex 32` generates a random key. Document each rotation in `docs/journeys/03-admin.md` §4.

4. **Start the services**:
   ```bash
   sudo systemctl enable --now fitscan-api fitscan-worker fitscan-beat
   sudo systemctl status fitscan-api      # should show 'active (running)'
   ```
   - `fitscan-api` is the FastAPI app (uvicorn, port 8000)
   - `fitscan-worker` is the Celery retention/quota-reset worker
   - `fitscan-beat` is the cron-like scheduler (FS-NIDLO-OPS-07 — make sure both worker AND beat are supervised)

5. **Verify**:
   ```bash
   curl http://127.0.0.1:8000/api/v1/health
   # → {"status":"ok","version":"0.1.0"}

   curl -H "X-Admin-API-Key: <FITSCAN_ADMIN_API_KEY>" \
        http://127.0.0.1:8000/api/v1/admin/celery/status
   # → {"workers":{"count":1,…},"broker_reachable":true,…}
   ```

6. **Reverse-proxy via Nginx** (templates in [OPERATIONS.md §8](OPERATIONS.md#8-nginx--systemd-on-the-server)). Two route patterns:
   - `api.fitscan.nidlo.com/api/v1/*` → public SaaS surface (TLS, rate limits, `X-API-Key` required)
   - `127.0.0.1:8000/api/v1/admin/*` → internal-only (bind localhost, never expose externally)

7. **Confirm Sentry receives a test event** (if you set `SENTRY_DSN`):
   ```bash
   sudo systemctl restart fitscan-api
   cd /home/fitscan/app && make sentry-smoke
   # → "Sentry test event sent. Event ID: …"
   # Open https://sentry.io/issues/?query=is:unresolved within ~30s.
   ```

   The same probe exists on the backend: `cd backend && php artisan sentry:smoke-test`.
   Both skip with a friendly message (exit 0) when their DSN is unset, so it's
   safe to run anywhere.

### Updating

```bash
ssh user@server
cd nidlo
git pull
cd fitscan
sudo ./deploy/install.sh --update     # runs alembic upgrade + restarts services
```

### Backup

The only thing that needs backing up is **the Postgres database** (avatars + API keys). Photo storage with `MEASUREMENTS_PHOTO_DISK=local` lives on disk — back up `/var/lib/fitscan/data/` if you want photo retention beyond the 30-day auto-purge. Once you flip to Backblaze B2 (`s3` disk), backups happen via B2's lifecycle policies.

---

## 3. Linking it to the Nidlo backend

The Laravel backend treats Fitscan as **an upstream API** — same pattern as Stripe or Twilio. There is no shared code; it's pure HTTP.

### What the Laravel side has

[`backend/app/Services/Measurement/AiMeasurementService.php`](../backend/app/Services/Measurement/AiMeasurementService.php) — the only file that talks to Fitscan. Calls `POST /upload_images` (legacy Flask shape; will swap to the Fitscan-native `/api/v1/measure` when MEAS-14 async pipeline lands).

It now forwards `degraded_modes` from Fitscan into the GraphQL `extractAiMeasurements` mutation result so the frontend can flag manual review (FS-NIDLO-VALID-03, ✅ shipped).

### Env vars Laravel needs

```bash
# backend/.env
FITSCAN_URL=https://api.fitscan.nidlo.com           # or http://127.0.0.1:8000 in dev
FITSCAN_API_KEY=<value of FITSCAN_INTERNAL_API_KEY> # the same one set in /etc/fitscan/.env
FITSCAN_ADMIN_API_KEY=<value of FITSCAN_ADMIN_API_KEY> # for the admin Filament reports page
FITSCAN_TIMEOUT=30                                  # seconds
```

The two keys MUST match across the two `.env` files. When you rotate, do both at once (FS-NIDLO-OPS-04 documents this).

### What the Laravel side does on a measurement request

1. Frontend uploads a photo via the GraphQL `extractAiMeasurements` mutation
2. Laravel `MediaValidator` checks size/type (10 MB cap, JPEG/PNG)
3. `AiMeasurementService` streams the file via `fopen()` + Guzzle to Fitscan (FS-C4 / BE-NIDLO-MEAS-03 — chunked, no memory blow-up)
4. Fitscan runs the full pipeline, returns measurements + `degraded_modes`
5. Laravel maps the response to the canonical `MeasurementType` GraphQL shape
6. Frontend shows the result; if `degradedModes` is non-empty, renders the "Manual review recommended" band

### The admin-only operational channel (`/api/v1/admin/*`)

Used by the **Filament admin panel** in the Laravel backend to render the Fitscan health + usage pages. Auth via `X-Admin-API-Key` (constant-time compare via `hmac.compare_digest`). The `FitscanService::celeryStatus()` and `FitscanService::usage()` methods bucket connection failures into ops-readable categories (FS-NIDLO-VALID-03 / B-LV-10) so admins see "Connection refused" rather than raw cURL errors.

### When to break the link

- **Fitscan is down**: the GraphQL mutation surfaces a friendly "AI measurement service is currently unavailable. Please try manual entry" — clients can still proceed with manual measurement entry. Don't 500 the frontend.
- **Anthropic is down**: the Claude validator silently falls back to a default-pass result. Measurements still come back; rule validation still runs. No user-visible failure.

---

## 4. Pricing model

### Today (in code, FS-NIDLO-AUTH-05 ✅ shipped)

Three tiers, monthly limits configurable via env:

| Tier | Monthly limit | Meant for |
|---|---|---|
| `free` | 50 calls | Trials, individual designers, students |
| `pro` | 1000 calls | Small fashion businesses, multi-designer studios |
| `enterprise` | 999 999 calls | Marketplaces, SaaS resellers, white-label customers |

The limits live in `Settings.{tier}_tier_monthly_limit` so you can flex them via env without redeploying. The `_TIER_LIMITS` dict in [`fitscan/api/routes.py`](fitscan/api/routes.py) is the single source of truth — both `/keys/register` validation and the DB row's `monthly_limit` column read from it.

### Suggested public pricing (USD)

These are conservative starting points — adjust after the first 90 days of real usage data.

| Tier | Monthly fee | Per call after limit | Sells to |
|---|---|---|---|
| **Free trial** | $0 | Hard stop at 50 | Individual designers, "let me try this for a project" |
| **Starter** | $29/mo | n/a (cap at 1000) | Small fashion business doing ≤30 measurements/day |
| **Pro** | $99/mo | $0.05/call beyond 1000 | Studio, ~3000 measurements/mo |
| **Enterprise** | Custom (from $499/mo) | Volume negotiated | Marketplaces, white-label, ≥10k/mo |

**One-shot model (alternative or in addition):** sell **Fitscan credits** — 1 credit = 1 `/measure` call — in packs (100 / 500 / 5000) at $0.40 / $0.30 / $0.20 per credit. Credits never expire. Easier sell to customers who don't want a subscription.

### What the customer gets at every tier

- Same `/api/v1/measure` endpoint shape — measurements + GLB + landmarks
- Same Claude vision validation
- Same RTBF (`DELETE /api/v1/avatar/{id}`) on demand
- Same OpenAPI docs + Postman collection
- Differs **only** in monthly call ceiling — no feature gating

This matters: feature-gating SaaS APIs is a maintenance burden that doesn't pay off at small scale. Just throttle on volume.

---

## 5. Unit economics

### What each `/measure` call costs us

Three cost lines:

1. **Anthropic Claude vision** — ~700-2500 tokens per call (image + prompt) at the Sonnet 4.6 rate. **Approx $0.003–$0.015/call** depending on image size + how many landmarks needed verification. The OPS-05 client memoization saves ~40% on connection overhead but the per-call token cost is fixed.
2. **Compute** — on a CPU-only Contabo VPS (8 GB RAM, 4 vCPU), one `/measure` call takes 20–60 s of full-tilt CPU. At Contabo pricing (~$15/month for the box) this is roughly **$0.0001/call** if the box is fully utilised — basically free, **but** the box can only handle ~2-3 concurrent calls before queueing kicks in. GPU acceleration (FS-NIDLO-OPS-02 — pending decision) would cut latency to 2-5 s per call and let one node handle 20+ concurrent calls.
3. **Storage** — body photos auto-purge at 30 days (FS-NIDLO-CELERY-01). Per-photo cost on Backblaze B2 (when we flip from local) is ~$0.0001/photo/month. Negligible.

**Floor cost per call: ~$0.005** (mostly Anthropic). Free-tier 50 calls/month = $0.25 lifetime cost per signup. Even if 99% of free signups never convert, that's still profitable.

### Break-even math

- Pro at $99/mo with 1000 calls = **$99 - (1000 × $0.005) = $94/mo gross margin** per Pro customer
- The Contabo box at $15/mo can probably handle ~50 active Pro customers before you hit queue saturation (50 × ~3 calls/day average = 150 concurrent risk windows)
- So one $15/mo box generates up to **$4700/mo in Pro revenue** at saturation — **313× cost coverage** at full utilisation

The bottleneck is therefore not money; it's per-call latency. GPU planning (OPS-02) decides when to scale beyond one CPU box.

### When to flip to GPU

Trigger: when concurrent `/measure` calls regularly exceed 3 during business hours, or when p95 latency crosses 60 s. RunPod / Vast.ai / Lambda Labs all rent A40 / A6000 cards by the hour. Migration plan: keep Contabo as the FastAPI orchestration node (cheap, always-on) and have it RPC to a GPU pool node for the SMPL fit step only. Fits inside the existing `MeasurementService.measure_with_mesh()` boundary — it's a one-method swap.

---

## 6. The customer signup flow

### Today (working but locked-down-pending)

```
Customer fills web form ─────► POST /api/v1/keys/register
                                 │
                                 ▼
                          tier allowlist + audit log
                                 │
                                 ▼
                          DB writes APIKeyRecord
                                 │
                                 ▼
                          Returns plaintext fsk_… key
```

What the response looks like:

```json
{
  "api_key": "fsk_3a7b8c…",
  "tier": "free",
  "monthly_limit": 50,
  "note": "Store this key securely. It will not be shown again."
}
```

The customer pastes the key into their backend env, starts calling `/api/v1/measure` with `X-API-Key: fsk_…`. Every call increments `usage_this_month`; over the limit returns 429.

### What's pending (FS-NIDLO-AUTH-03)

`/keys/register` is currently **publicly callable** — anyone can mint a free-tier key with no email verification, no rate limit, no abuse protection. Three options on the table (Snad-decision):

1. **Admin-only**: drop the public route. Snad mints keys via an admin Filament UI, hands them out to customers manually. Best for the curated-beta phase.
2. **Trusted-network**: keep the route, bind it to the box's local interface only, front it with a webhook from the marketing site that's protected by Cloudflare Turnstile / hCaptcha. Best for "scale me to 100 customers without me being in the loop".
3. **Email-confirm + rate-limit**: keep the route public but require email confirmation before issuing the key, and throttle to 3 keys/IP/hour via slowapi + Redis. Best for "pure self-serve SaaS".

Every option is a 1-day swap. **My recommendation for the next 6 months: start with #1 (admin-only).** You're curating the beta anyway; manual key issuance gives you direct conversation with each customer = qualitative feedback you'd otherwise miss.

### What the SaaS dashboard should look like (pre-launch design)

A separate Filament panel at `admin.nidlo.com/fitscan-saas` (or its own subdomain) showing:

- **Customers**: rows = each `APIKeyRecord`. Columns: owner_email, tier, usage_this_month / monthly_limit, last_used_at, total_lifetime_calls, total_revenue_to_date.
- **Issue key**: button → form (owner_email, owner_name, tier, optional 1-time multi-month override). On submit, mints the key and emails it to the owner via Resend. Logs the issuance to the existing audit-log table.
- **Suspend / reactivate**: toggle `is_active`. Suspending stops measurement calls but doesn't delete the row (pending-payment workflow).
- **Billing**: row → Stripe customer link → invoice list. Stripe webhook on `invoice.payment_failed` auto-suspends the key.

None of this is built. It's the **Phase 1 product layer above the existing Fitscan API**. Estimate: 5-7 days of Laravel + Filament work once you decide the trust model.

---

## 7. Public landing page brief

When you're ready to publicly market Fitscan as its own product (separate from Nidlo), here's what `fitscan.com` (or `fitscan.nidlo.com`) should pitch:

### Hero

> **Body measurements from a single photo.**
> 24 fields. 30 seconds. ±5cm accuracy.
> Built for fashion businesses who can't afford a tape measure at every customer's door.

### Three trust-builders

1. **Privacy by design** — body photos auto-delete after 30 days. Per-customer right-to-be-forgotten endpoint. No model training on your data.
2. **Tailor-grade fields** — bust, waist, hip, shoulder, kaba length, sleeve, nape-to-waist, and 17 more. The full Ghanaian-tailor field set, not just the 5-field "size" approximations other APIs return.
3. **Confidence per measurement** — every field carries a 0–1 confidence score. Your UI knows which fields to show as "AI-suggested" vs "needs manual review".

### What a buyer signs up for

A button: **"Start free — 50 measurements"** → form (name, email, company, expected use case dropdown) → admin review (current trust model) → email with API key + 3 working code snippets (PHP / Node / Python).

### Below the fold

- **Pricing table** (the 4 tiers from §4)
- **Code samples** — show `curl -X POST /measure -F image=@photo.jpg -F height_cm=165` and the JSON response
- **Comparison table** vs. competitors (3DLook, Bodygram, custom-tape-measure-form)
- **OpenAPI link** — "complete API reference" → `fitscan.com/docs`
- **Customer logos** when you have them

### Doc surface

| Page | Source | Audience |
|---|---|---|
| `/docs` | Generated from FastAPI OpenAPI | Developers integrating |
| `/pricing` | Manual MD render | Buyers comparing |
| `/playground` | A live "drag photo, see measurements" demo (rate-limited free trial) | "Show me what it does" |
| `/changelog` | [`CHANGELOG.md`](CHANGELOG.md) auto-rendered | Existing customers tracking changes |
| `/status` | Sentry-backed status page | Existing customers when something breaks |

The marketing site itself is out of scope for the Fitscan codebase — build it as a separate Next.js project that calls Fitscan's public OpenAPI for live examples.

---

## 8. Profitability path

### Phase A — Beta on Nidlo (now → 3 months out)

- Fitscan ONLY serves Nidlo's own measurement flow
- No external customers, no revenue from Fitscan directly
- Fitscan's value is enabling Nidlo to undercut the "manual measurement" friction other Ghanaian platforms have
- Cost: Anthropic ~$0.005 per Nidlo client × ~50 clients/month = ~$0.25/mo. Lost in the noise.

### Phase B — Soft external launch (3 → 6 months)

- AUTH-03 lockdown decision lands → admin-only key issuance
- Onboard 3-5 hand-picked external customers (other Ghanaian / West African fashion businesses) at flat $99/mo Pro tier
- Use their feedback to harden edge cases the Nidlo flow doesn't hit (different camera setups, more diverse body types, etc.)
- Revenue: ~$300-500/mo. Still loss-leader; pays the Anthropic bill plus a slice of the Contabo cost.
- Critical metrics: per-customer monthly call volume (do they actually use it?), latency p95 (do they complain?), retention at 90 days.

### Phase C — Public SaaS (6 months out, after #1 lockdown decision + #2 GPU planning + #3 billing integration)

- Public `fitscan.com` landing
- Self-serve signup flow (whatever AUTH-03 lockdown landed at)
- Stripe subscriptions for tier pricing + Stripe metered billing for per-call overage
- Fitscan becomes its own P&L line, separate from Nidlo
- Target: 30 paying customers in the first 6 months public = ~$3000 MRR. Covers the GPU node + 2 days/month of ops time.

### Token model (alternative to subscriptions)

If you'd rather sell **credits** than subscriptions (as the user message hinted):

- Each `/measure` call burns 1 credit
- Credits are minted by paying via Stripe / mobile money / wallet top-up
- Pricing tiers become **credit pack discounts**: $40 for 100, $150 for 500, $1000 for 5000
- Credits never expire (unless you set a 24-month dormancy clause)
- Implementation: add a `credits` column to `APIKeyRecord`, decrement on each successful `/measure`, return `402 Payment Required` when zero

This is an **either/or vs the subscription model**, not "and". Subscriptions = predictable revenue + better LTV math. Credits = lower friction signup + better fit for spiky usage. Pick one and stick with it.

**My recommendation: subscriptions.** Fashion businesses operate on monthly cycles already; subscriptions match their mental model.

### What you'd need to build for either model

| Need | Subscription | Credits |
|---|---|---|
| Stripe customer + product setup | Yes | Yes |
| Webhook on `invoice.payment_succeeded` | Yes (resets quota) | n/a |
| Webhook on `payment_intent.succeeded` | n/a | Yes (mints credits) |
| Cron to suspend on overdue invoice | Yes | n/a (just runs out of credits) |
| Frontend "buy more" button | n/a | Yes |
| Migration from MoMo to Stripe | Optional | Optional |

Either way: **estimated build effort 3-4 days** once the AUTH-03 trust model is settled.

---

## 9. Pre-launch checklist

These items gate "we open to external customers". Tracked as the pre-beta blockers in [ROADMAP.md §75](ROADMAP.md):

- [x] **AUTH-03** — `/keys/register` lockdown trust model: admin-only via `X-Admin-API-Key` (same gate as `/api/v1/admin/*`). Snad mints keys via curl until the Filament panel ships. ✅ shipped.
- [ ] **TEST-01 fixtures** — 8-10 ground-truth photos so numerical regressions can't slip through silently. The harness is ready ([`tests/integration/test_pipeline_accuracy.py`](tests/integration/test_pipeline_accuracy.py)); fixtures need sourcing. 📐 Snad-decision (synthetic vs consented real).
- [ ] **OPS-02** GPU planning — Contabo CPU is fine for beta scale; public launch needs measurable latency budget. 📐 Snad-decision.
- [ ] **OPS-01** Sentry actually wired — SDK is shipped on both services and a verification probe exists (`make sentry-smoke` in fitscan, `php artisan sentry:smoke-test` in backend). Snad-side: create the Sentry project, set `SENTRY_DSN` in both `.env` files on the server, run each smoke command and confirm the event lands.
- [ ] **OPS-07** Celery beat under systemd — without it the retention + quota-reset tasks never fire. The systemd unit lives in `deploy/install.sh` already; verify with `sudo systemctl is-enabled fitscan-beat && sudo systemctl is-active fitscan-beat` after each deploy.
- [ ] **VALID-02** privacy-copy contract for Claude consent — Snad-decision on the exact wording the frontend shows users before shipping their photo to Anthropic.
- [ ] Marketing landing page (separate Next.js project, ~2 days)
- [ ] Stripe / billing integration (subscription OR credits — see [§8](#8-profitability-path), ~3-4 days)
- [ ] Customer-facing docs (`fitscan.com/docs` from the OpenAPI; templates ready in `docs_source/`, ~1 day to wire)
- [ ] Admin Filament panel for issuing keys + watching usage (`admin.nidlo.com/fitscan-saas`, ~5 days)

**Order of attack** — items that unblock other items first:
1. AUTH-03 decision (everything downstream depends on it)
2. OPS-01 Sentry DSN (zero code, just env)
3. Admin Filament panel (you can't onboard customers without it)
4. Stripe integration (you can't charge without it)
5. Marketing site + customer docs (parallel)
6. TEST-01 fixtures (parallel; nice-to-have for confidence in latency/correctness regressions)
7. OPS-02 GPU (after first 5 paying customers, when you have real usage data to size from)

---

## Reference — file layout in this repo

```
fitscan/
├── README.md          ← 30-sec product pitch, dev quick-start
├── LAUNCHING.md       ← THIS FILE: hosting + integration + pricing
├── OPERATIONS.md      ← detailed runbook: env vars, systemd, nginx, troubleshooting
├── TASKS.md           ← granular task ledger with status icons
├── ROADMAP.md         ← exec summary of what's in code
├── CHANGELOG.md       ← versioned release notes
├── deploy/
│   └── install.sh     ← one-command Ubuntu installer
├── docs/
│   ├── index.html              ← static API reference (built from OpenAPI)
│   ├── openapi.json            ← machine-readable API spec
│   └── FitScan.postman_collection.json   ← Postman collection for testing
├── fitscan/           ← the Python package
│   ├── api/           ← FastAPI routes
│   ├── application/   ← orchestration
│   ├── cv_pipeline/   ← MediaPipe + rembg + scale calibration
│   ├── smpl/          ← HMR2 + SMPL fitting
│   ├── validation/    ← rule + Claude validators
│   ├── infrastructure/← DB + storage backends
│   └── tasks.py       ← Celery
└── tests/             ← 183 passing tests, 85% coverage
```

When in doubt, read in this order: **README → LAUNCHING (this) → OPERATIONS → TASKS for specifics**.
