# Fitscan (Python) — Roadmap

> Last updated 2026-05-01. Companion to `audits/fitscan/99_FINDINGS.md` and `docs/journeys/`.

## What this is

Operational roadmap for the **Fitscan AI body-measurement service** (Python 3.11+ + FastAPI + PyTorch + MediaPipe + SMPL-X + rembg + trimesh + Anthropic SDK + SQLAlchemy + Alembic + Celery). Replaces the older Flask `ai-measurements` service. Backend (Laravel) calls this over HTTP for body measurements + avatar generation.

This is NOT the spec — `docs/section-06-ai-strategy.md` and `docs/stitchhub-complete-documentation/stitchhub-fitscan-3d-integration.md` are. This is what's *in code* today.

---

## Status today

- **41 Python source files** in `fitscan/`, **25 test files**.
- **Test status**: 68 passing / 0 failing, 55.86% coverage (above 55% threshold).
- **Stage**: private beta service, called from the Laravel backend over HTTP. Hosted on Contabo VPS (CPU-only — see GPU planning below).
- **Architectural decisions**:
  - **FastAPI + Pydantic v2** for routes + schemas
  - **`X-API-Key` auth required** on all endpoints except `/health` and `/keys/register` (env `FITSCAN_REQUIRE_API_KEY=true` in prod, `false` in test)
  - **`AvatarStore` (SQLAlchemy)** — `avatars` table with absolute filesystem paths (now sandboxed via `_resolve_inside_avatars_root`)
  - **`ObjectStore`** — local fs / MinIO / S3 backend. MinIO uses `secure=True` by default; S3 puts include `ServerSideEncryption='AES256'`
  - **Pipeline orchestration**: image → preprocess → pose detect (MediaPipe) → scale calibrate (user height OR reference object) → silhouette measure → SMPL fit → mesh measure (rescaled to user height) → fuse → score
  - **Celery app** (`fitscan/tasks.py`) with broker/backend env-driven (Redis dev, AMQP prod). Reliability defaults: `task_acks_late=True`, `task_reject_on_worker_lost=True`, `task_time_limit=15min`, `worker_max_tasks_per_child=200`
  - **Privacy controls**: avatar 30-day retention, `subject_id` consent token required for Claude validation (without it, photo NOT shipped to Anthropic), `delete_avatar()` for right-to-be-forgotten
  - **Numerical safety**: streaming uploads with 10MB cap, `Image.MAX_IMAGE_PIXELS` decompression-bomb guard, mesh rescale via `scale_mesh_to_height` (was returning template ~1.7m for everyone)
  - **Pesewas-aware**: not directly — Fitscan returns measurements, Laravel handles pricing

---

## Closed audit findings (A1–A7)

9 Critical + 2 High closed across Fitscan.

| ID | Sprint | Fix |
|----|--------|-----|
| FS-C1 | A1 | `verify_api_key` FastAPI dependency on every protected route. Validates `X-API-Key`, checks `is_active`, enforces `monthly_limit`, increments `usage_this_month` + `last_used_at`. |
| FS-C3 | A2 + A4 | Avatar lifecycle: `AvatarStore.delete()` + `delete_older_than()`, MinIO `secure=True`, S3 `ServerSideEncryption='AES256'`, Anthropic Claude requires `subject_id` consent token, `metadata={"user_id": ...}` on Claude calls. New `DELETE /avatar/{id}` route. |
| FS-C4 | A4 | Streaming uploads via `_stream_to_tempfile` (64KB chunks, aborts at 10MB), `Image.MAX_IMAGE_PIXELS` guard at app startup. OOM + decompression-bomb both closed. |
| FS-C5 | A1 + A7 | A1 added `usage_this_month` increment + `is_active` check in middleware. A7 added monthly reset task `reset_api_key_quotas` (Celery beat, 1st-of-month 00:05 UTC). |
| FS-C6 | A4 | `MeasurementService.measure_with_mesh()` exists. `PipelineOrchestrator.run_with_mesh()` returns `PipelineResult(measurements, mesh, betas, thetas)`. `/measure/avatar` no longer 500s on every request. |
| FS-C7 | A5 | `scripts/generate_docs.py` is now read-only — exports OpenAPI JSON + copies static files. No longer mutates `routes.py` / `main.py`. Supply-chain risk closed. |
| FS-C9 | A7 | `_resolve_inside_avatars_root()` sandboxes every `FileResponse(...)` path. DB-stored paths can no longer escape `data/avatars/`. 404 (not 403) so the failure mode doesn't leak which case triggered. |
| FS-H2 | A4 | `fitscan/tasks.py` exists. `make worker` no longer fails with `ModuleNotFoundError`. Celery app + `cleanup_old_avatars` daily task + `reset_api_key_quotas` monthly task. |
| FS-H13 | A4 | Mesh rescaled to user's actual `height_cm` via `scale_mesh_to_height` in `PipelineOrchestrator._rescale_mesh_to_height`. `ScaleFactor.target_height_cm` field carries the value from calibrator → analyzer. **Headline measurement now correct** — was returning ~1.7m template for everyone. |

---

## Open audit findings — backlog

Filtered to Fitscan-relevant items still on the list. Severity from `audits/fitscan/99_FINDINGS.md`.

### Critical

- **FS-C8 — `subprocess.run` invocations in CLI use unvalidated cwd** (low practical impact — CLI only, not RCE). Marker for sloppy subprocess hygiene. Pass explicit `cwd=` and `check=True` everywhere.

### High

- **FS-H3 — No idempotency / retry policy on (planned) Celery tasks.** A1's auth dep set `acks_late` + `reject_on_worker_lost` defaults but per-task: when `measure_async` is added, it must accept an `idempotency_key` (e.g. `sha256(image_bytes + params)`), `autoretry_for=(httpx.HTTPError,)` for the Anthropic call, and a DLQ on permanent failure. Foundation is in place; per-task config when measure_async lands.
- **Tempfile cleanup on `uvicorn` hard-kill is best-effort** (FS-C3 sub-item e). If the worker is hard-killed mid-request, the JPEG can linger in `/tmp`. Mitigation: rotate temp dir + sweep on startup.

### Medium

- **16 medium findings** in `audits/fitscan/99_FINDINGS.md`: broad `except`, unclosed file handles, hardcoded magic numbers in measurement math, missing seeds for determinism, missing docstrings on numerical functions, inline `import` statements (5 ruff warnings persist), inconsistent error response shapes.

### Test coverage gaps

- **No measurement regression tests against fixture images.** The audit explicitly called this out: "are measurement values regression-tested against fixture images?" — answer: no. This is the single most important missing test category. Without it, FS-H13 (the rescale fix) has no numerical guarantee — only type safety.
- **`/measure/avatar`, `/avatar/{id}/glb`, `/avatar/{id}/thumbnail`, `/avatar/{id}/fit-garment`, `/avatar/{id}/override` routes have zero test coverage.** A single integration test would have caught FS-C6 ("`measure_with_mesh` does not exist") on day one.
- **Celery task tests** — `cleanup_old_avatars` and `reset_api_key_quotas` are wired but no test exercises them end-to-end with a fake clock.
- **`mesh_analyzer.py` 14% coverage** — none of the existing tests exercise the SMPL path with a real mesh.

---

## Pre-beta must-fix (Fitscan slice)

1. **Numerical regression tests** — pick 5–10 fixture images with known body measurements (curated dataset, ideally from real users with their consent), assert pipeline output is within tolerance. Without this, every refactor risks silently regressing the headline metric.
2. **GPU planning** — Contabo VPS is likely CPU-only. PyTorch + SMPL-X on CPU runs measurement in ~20-60s. For private beta with curated designers it's tolerable; public launch needs a GPU node or model quantization. **Decision needed before beta opens**.
3. **Sentry / observability** — Fitscan currently uses Loguru only. Backend has Sentry SDK; Fitscan should too — Anthropic API errors, SMPL fit failures, measurement value clamps all need visibility.
4. **`/keys/register` admin gate** — currently public. Anyone can register an API key. Fine for the trusted-network-internal model with Laravel as the only caller, but document the trust assumption explicitly OR put it behind an admin-only gate.

---

## Phase 2 — deliberately deferred

Per `~/.claude/projects/-Users-mac-Projects-mine-stitchub/memory/project_phase_and_scope.md`:

- **GPU autoscaling** — single VPS in beta
- **Multi-region deployment** — single Contabo VPS
- **On-device inference fallback** — privacy + cost win, big infra work; Phase 3
- **Video stream support** — image-only today
- **Style classification + garment recommendation from measurements via Claude** — Snad asked for this as a near-term feature task. Anthropic SDK is already a dep; the integration is scoped but not built. Tracked in `~/.claude/projects/-Users-mac-Projects-mine-stitchub/memory/project_vendors.md` "Open items to revisit with Snad" #6.

---

## Tech-debt backlog

- **Inline `import` statements** in `routes.py` (5 ruff warnings persist) — pre-existing style issue. Move to top-level imports for the modules that aren't truly conditional.
- **Magic numbers in `mesh_analyzer.py` `LANDMARK_HEIGHTS`** — well-tuned but undocumented. Add citations to the SMPL paper or reference body proportion tables.
- **`AvatarStore.update_avatar` raises `KeyError`** instead of a domain exception. Inconsistent with the rest of the module.
- **`ClothingFitter.fit_garment` writes to `data/avatars/{id}` directly** rather than through `ObjectStore.put` — bypasses the storage abstraction. Refactor when next touched.
- **`config.get_settings()`** does not enforce required env vars at startup — first-call failure surfaces deep in the pipeline. Use Pydantic Settings validation.
- **Celery beat scheduling lives in `tasks.py`** — fine for a single project but if Fitscan ever spawns sister Python services they'll need a shared `celery_app` package.
- **Anthropic API key is read directly from env into the validator constructor** (`measurement_service.py:46`) and a fresh `Anthropic` client is constructed per call. Fine for security; bad for connection pooling. Memoize the client.
- **No model quantization** — float32 PyTorch weights load fully. Quantize to int8 / fp16 for ~3× perf on CPU.
- **`fitscan_test.db` SQLite** — fine for tests, but the production `DATABASE_URL` should be Postgres + pgvector. Verify the production env actually uses Postgres before launch.

---

## Direction (next 6–8 weeks of Fitscan work)

- **Pre-beta correctness gates**: items #1–4 above (regression tests, GPU planning, Sentry, `/keys/register` decision)
- **Claude style classification + garment recommendation** — Snad's near-term feature ask. Add a new `/measure/with-recommendations` route that returns measurements + Claude-generated style notes + garment fit suggestions
- **Async pipeline** — kick measurement to Celery, return 202 + job ID. Frontend polls or subscribes via WebSocket (Reverb on backend). Resolves the synchronous blocking-inference problem the audit flagged.
- **Webhook callback to Laravel** — when Celery measurement completes, POST results back to backend instead of polling
- **Improve `_smplify_fallback`** — currently a 100-iter Adam loop, slow. Either drop it (HMR2 is the supported path) or speed up
- **Image-prep performance** — `rembg` is slow. Profile + cache bg-removed images for re-measurement
- **OPS: Celery beat process management** — Supervisor / systemd unit for beat + worker on Contabo

After private beta:
- **GPU node** — likely cheap GPU VPS (RunPod / Vast / Lambda Labs) for inference, keep Contabo for FastAPI/orchestration
- **Model size reduction** — distill or quantize HMR2 + SMPL-X to fit on smaller hardware
- **On-device inference for the simplest measurements** — pose detection can run in the browser (TensorFlow.js MediaPipe), only ship the full SMPL fit to server
- **Privacy-preserving compute** — do measurement on-device, send only the measurements, never the photo. Big work; Phase 3 alignment.

---

## Cross-references

- **Audit findings**: `audits/fitscan/99_FINDINGS.md` (10 Critical + 18 High + 16 Medium + 9 Test gaps)
- **Sprint summary**: `~/.claude/projects/-Users-mac-Projects-mine-stitchub/memory/project_audits.md`
- **Architecture decisions**: project root `CLAUDE.md` "Tech Stack" + "Fitscan as separate Python service"
- **Vendor + hosting**: `~/.claude/projects/-Users-mac-Projects-mine-stitchub/memory/project_vendors.md` "Hosting" + "AI" sections
- **Journey docs that touch Fitscan**: `docs/journeys/00-cross-cutting.md` §4 (camera/permission), `01-client.md` §6 (body measurements), `03-admin.md` §4 (Fitscan ops), `05-failure-modes.md` §4 (Fitscan failures)
- **Operations runbook**: `docs/journeys/03-admin.md` §4 (retention, quota reset, deletion)
