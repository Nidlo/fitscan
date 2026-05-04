# Nidlo Fitscan — Implementation Tasks

**Owner:** Snad
**Started:** 2026-05-01
**Goal:** Source-of-truth task catalog for the Fitscan AI body-measurement service. Every backend / frontend / admin dependency that needs Fitscan work appears here as a `FS-NIDLO-*` task. Companion to [frontend/ROADMAP.md](../frontend/ROADMAP.md), [backend/TASKS.md](../backend/TASKS.md), [backend/ROADMAP_ADMIN.md](../backend/ROADMAP_ADMIN.md).

The slim exec summary lives at [ROADMAP.md](ROADMAP.md). This file is the granular per-task ledger.

---

## Status legend

| Icon | Meaning |
|---|---|
| ✅ | Done — landed in code, has tests where applicable |
| 🟡 | Partial — code exists, gaps / mocks / weak coverage |
| ⬜ | Not started |
| 📐 | Needs Snad sign-off (product / contract / consent decision) |
| 🔒 | Phase 2/3 — explicitly deferred, do NOT implement |
| 🐛 | Broken in code today — high-priority fix |

---

## Conventions

- **Python 3.11+** (PyTorch + MediaPipe + SMPL-X compat). 3.13 breaks `mediapipe.solutions` — do NOT bump.
- **FastAPI + Pydantic v2** for routes + schemas. Schema in [fitscan/api/schemas.py](fitscan/fitscan/api/schemas.py).
- **Loguru** for logging — never `print`. Sentry SDK wiring is open ([FS-NIDLO-OPS-01](#phase-5--production-hardening)).
- **Alembic** for migrations; **SQLAlchemy 2.x** typed mappings. Alembic is referenced from [Makefile](fitscan/Makefile) but the scaffold is incomplete — see [FS-NIDLO-DB-01](#phase-2--database-schema-correctness).
- **Tests**: `pytest tests/` runs against `.env.test` via [tests/conftest.py](fitscan/tests/conftest.py). Coverage gate is 55% in [pyproject.toml](fitscan/pyproject.toml); raise to 80% per audit T9.
- **Privacy first**: body photos + 3D scans = sensitive PII. Encrypted at rest (S3 SSE='AES256'), 30-day retention, consent-gated for Anthropic (`subject_id` token).
- **Auth**: `X-API-Key` header. Laravel backend is the only legitimate caller in beta. `FITSCAN_REQUIRE_API_KEY=true` in prod, `false` in `.env.test`.
- **ID prefix**: `FS-NIDLO-<MODULE>-<NN>` where `<MODULE>` ∈ `{AUTH, MEAS, AVATAR, STYLE, CV, SMPL, VALID, STORAGE, DB, CELERY, CLI, OPS, TEST}`.
- **Cross-references** are grep-able: `→ BE-NIDLO-MEAS-04`, `→ FE-NIDLO-MEAS-02`, `→ AD-NIDLO-USER-07`, `(✅ closed A4, FS-C6)`.

---

## Module map

| Module | Owns | Key files |
|---|---|---|
| `AUTH` | API-key registration, header verification, quota | [fitscan/api/dependencies.py](fitscan/fitscan/api/dependencies.py), [fitscan/api/routes.py:226](fitscan/fitscan/api/routes.py#L226) |
| `MEAS` | `/measure`, `/measure/measurements`, `/measure/avatar` | [fitscan/api/routes.py:90](fitscan/fitscan/api/routes.py#L90), [fitscan/application/measurement_service.py](fitscan/fitscan/application/measurement_service.py) |
| `AVATAR` | Avatar generation, GLB export, garment fit, override, retention | [fitscan/avatar/generator.py](fitscan/fitscan/avatar/generator.py), [fitscan/avatar/clothing_fitter.py](fitscan/fitscan/avatar/clothing_fitter.py), [fitscan/avatar/inverse_smpl.py](fitscan/fitscan/avatar/inverse_smpl.py) |
| `STYLE` | Claude style classification + garment recommendations (planned) | _new — TBD_ |
| `CV` | MediaPipe pose, rembg silhouette, scale calibration | [fitscan/cv_pipeline/](fitscan/fitscan/cv_pipeline/) |
| `SMPL` | HMR2 fit, SMPLify fallback, mesh measurement | [fitscan/smpl/body_fitter.py](fitscan/fitscan/smpl/body_fitter.py), [fitscan/smpl/mesh_analyzer.py](fitscan/fitscan/smpl/mesh_analyzer.py) |
| `VALID` | Rule + Claude validators, fusion, confidence | [fitscan/validation/](fitscan/fitscan/validation/), [fitscan/fusion/](fitscan/fitscan/fusion/) |
| `STORAGE` | Local fs / MinIO / S3 backend | [fitscan/infrastructure/object_store.py](fitscan/fitscan/infrastructure/object_store.py) |
| `DB` | SQLAlchemy models, Alembic migrations | [fitscan/infrastructure/result_store.py](fitscan/fitscan/infrastructure/result_store.py), [fitscan/alembic/](fitscan/alembic/) |
| `CELERY` | Async tasks, beat schedule, retention, quota reset | [fitscan/tasks.py](fitscan/fitscan/tasks.py) |
| `CLI` | Local fitscan CLI, smoke tests | [fitscan/cli/fitscan_cli.py](fitscan/cli/fitscan_cli.py), [scripts/smoke_test.py](fitscan/scripts/smoke_test.py) |
| `OPS` | Sentry, GPU planning, Reverb webhook auth, deployment | [deploy/install.sh](fitscan/deploy/install.sh), [Makefile](fitscan/Makefile) |
| `TEST` | pytest suites, regression fixtures, benchmark | [tests/](fitscan/tests/) |

---

## Phase 0 — Foundations (largely closed by A1–A7)

### 0.1 — Auth & key management

- [✅] **FS-NIDLO-AUTH-01** — `verify_api_key` FastAPI dependency on every protected route. Hashes `X-API-Key` (sha256), looks up in `api_keys`, checks `is_active`, enforces `monthly_limit`, increments `usage_this_month` + `last_used_at`. Implemented in [fitscan/api/dependencies.py:43](fitscan/fitscan/api/dependencies.py#L43); applied via `_auth = [Depends(verify_api_key)]` at [fitscan/api/routes.py:19](fitscan/fitscan/api/routes.py#L19). _(✅ closed A1, FS-C1)_
- [✅] **FS-NIDLO-AUTH-02** — `usage_this_month` increment + `is_active` check inside the auth dep ([fitscan/api/dependencies.py:75-83](fitscan/fitscan/api/dependencies.py#L75)). Returns 429 on monthly cap. _(✅ closed A1, partial FS-C5)_
- [⬜] **FS-NIDLO-AUTH-03** — Lock down `POST /keys/register` ([fitscan/api/routes.py:226](fitscan/fitscan/api/routes.py#L226)). Currently public — anyone can self-register a free-tier key. Options: (a) admin-only (require an `X-Admin-Key` from `Settings.fitscan_admin_key`), (b) trusted-network-only (Nginx allow-list + `127.0.0.1` bind), (c) move out of HTTP entirely into a CLI-only command. Document the trust assumption. _Cross-link → BE-NIDLO-OPS-XX (Laravel keyrotation procedure), AD-NIDLO-OPS-XX (admin "issue Fitscan key" UI)._ 📐 Snad-decision pending.
- [✅] **FS-NIDLO-AUTH-04** — `FITSCAN_REQUIRE_API_KEY` env-gate. Falsy in `.env.test` so the test suite doesn't have to mint keys; production must leave it unset (defaults to true). [fitscan/api/dependencies.py:29](fitscan/fitscan/api/dependencies.py#L29). _(✅ closed A1)_
- [⬜] **FS-NIDLO-AUTH-05** — Read `monthly_limit` for tier from `Settings.free_tier_monthly_limit` instead of the hardcoded `{"free": 50, "pro": 1000, "enterprise": 999999}` at [fitscan/api/routes.py:237](fitscan/fitscan/api/routes.py#L237). Audit M7. Trivial fix; do alongside AUTH-03.
- [⬜] **FS-NIDLO-AUTH-06** — Per-key rate limit on `/measure*` routes (e.g. 10 req/hr/key — each call is 20–60s of CPU). `slowapi` + Redis-backed limiter. Audit H10. _Cross-link → OPS-07 (Redis as Celery broker)._

### 0.2 — Privacy + retention

- [✅] **FS-NIDLO-AVATAR-01** — `AvatarStore.delete(avatar_id)` removes DB row + on-disk parent dir. Guards against shared-parent traversal by checking `avatar_id in parent.parts`. [fitscan/infrastructure/result_store.py:117](fitscan/fitscan/infrastructure/result_store.py#L117). _(✅ closed A2, FS-C3)_
- [✅] **FS-NIDLO-AVATAR-02** — `AvatarStore.delete_older_than(days)` for retention. [fitscan/infrastructure/result_store.py:145](fitscan/fitscan/infrastructure/result_store.py#L145). _(✅ closed A2, FS-C3)_
- [✅] **FS-NIDLO-AVATAR-03** — `DELETE /api/v1/avatar/{id}` route (204 on success, 404 on miss). [fitscan/api/routes.py:161](fitscan/fitscan/api/routes.py#L161). The right-to-be-forgotten path Laravel calls. _Cross-link → BE-NIDLO-USER-XX (account-deletion job dispatches DELETE), → AD-NIDLO-USER-07 (admin "purge body scan" action)._ _(✅ closed A2)_
- [✅] **FS-NIDLO-CELERY-01** — `cleanup_old_avatars` daily 04:00 UTC task ([fitscan/tasks.py:37](fitscan/fitscan/tasks.py#L37) + beat schedule [tasks.py:79](fitscan/fitscan/tasks.py#L79)). Configurable via `FITSCAN_RETENTION_DAYS` (default 30). _(✅ closed A4)_
- [✅] **FS-NIDLO-CELERY-02** — `reset_api_key_quotas` monthly 1st 00:05 UTC task ([fitscan/tasks.py:52](fitscan/fitscan/tasks.py#L52)). Without it every key locks once it crosses `monthly_limit` and never resets. _(✅ closed A7, FS-C5)_
- [✅] **FS-NIDLO-STORAGE-01** — MinIO `secure=True` default ([fitscan/infrastructure/object_store.py:15](fitscan/fitscan/infrastructure/object_store.py#L15)) + S3 puts include `ServerSideEncryption='AES256'` ([object_store.py:130](fitscan/fitscan/infrastructure/object_store.py#L130)). Covers PII-at-rest for body scans. _(✅ closed A2, FS-C3)_
- [✅] **FS-NIDLO-VALID-01** — Anthropic Claude validation requires explicit `subject_id` consent token. Without it `ClaudeValidator.validate` short-circuits to `_default_result()` and the photo is NOT shipped to Anthropic. `metadata={"user_id": subject_id}` passed to `client.messages.create` so Anthropic retention controls can target the row. [fitscan/validation/claude_validator.py:55](fitscan/fitscan/validation/claude_validator.py#L55). _(✅ closed A2, FS-C3)_
- [⬜] **FS-NIDLO-VALID-02** — Document the consent contract: who supplies `subject_id`, where it comes from, retention semantics on the Laravel side. _Cross-link → BE-NIDLO-MEAS-04 (Laravel sends `user_uuid`), → FE-NIDLO-MEAS-03 (consent toggle in measurement flow)._ 📐 Snad-decision pending — privacy copy.

### 0.3 — Numerical & input safety

- [✅] **FS-NIDLO-MEAS-01** — `_stream_to_tempfile` streams uploads in 64KB chunks, aborts at `FITSCAN_MAX_UPLOAD_MB` (default 10MB) with HTTP 413. [fitscan/api/routes.py:53](fitscan/fitscan/api/routes.py#L53). Replaces the old `await image.read()` that buffered the entire body before any size check. _(✅ closed A4, FS-C4)_
- [✅] **FS-NIDLO-MEAS-02** — `Image.MAX_IMAGE_PIXELS` decompression-bomb guard at app startup (default 60M pixels). [fitscan/api/main.py:13](fitscan/fitscan/api/main.py#L13). _(✅ closed A4, FS-C4)_
- [✅] **FS-NIDLO-AVATAR-04** — `_resolve_inside_avatars_root` sandboxes every `FileResponse(...)` path. DB-stored absolute paths can no longer escape `data/avatars/`. Returns 404 (not 403) so failure mode doesn't leak which case triggered. [fitscan/api/routes.py:34](fitscan/fitscan/api/routes.py#L34). _(✅ closed A7, FS-C9)_
- [⬜] **FS-NIDLO-MEAS-03** — Tighten CORS. `allow_origins=["*"]` at [fitscan/api/main.py:30](fitscan/fitscan/api/main.py#L30) is permissive. Drive from `Settings.cors_origins`; default to the Laravel backend host only; for "internal-only" deployments remove the middleware. Audit C2.
- [⬜] **FS-NIDLO-MEAS-04** — Replace `raise HTTPException(500, str(e))` at [routes.py:127](fitscan/fitscan/api/routes.py#L127), [routes.py:291](fitscan/fitscan/api/routes.py#L291), [routes.py:366](fitscan/fitscan/api/routes.py#L366) with a `safe_500` helper that returns `{"detail": "Internal error", "trace_id": <uuid>}` and logs the full traceback against `trace_id`. Verbose responses gated on `Settings.is_production`. Audit C10. _Cross-link → BE-NIDLO-OPS-XX (correlate Laravel error → Fitscan trace_id)._

---

## Phase 1 — Measurement endpoints

### 1.1 — Synchronous routes (shipped)

- [✅] **FS-NIDLO-MEAS-10** — `POST /api/v1/measure` — measurements + optional avatar. Accepts `image`, `height_cm` OR `reference_object_cm`, `gender_profile`, `enable_smpl`, `enable_claude`, `subject_id`. [fitscan/api/routes.py:90](fitscan/fitscan/api/routes.py#L90). _Cross-link → BE-NIDLO-MEAS-01 (Laravel `MeasurementService` calls this)._
- [✅] **FS-NIDLO-MEAS-11** — `POST /api/v1/measure/measurements` — measurements only, skips avatar generation (faster). [fitscan/api/routes.py:266](fitscan/fitscan/api/routes.py#L266).
- [✅] **FS-NIDLO-MEAS-12** — `POST /api/v1/measure/avatar` — returns GLB binary directly (`Content-Type: model/gltf-binary`, `Content-Disposition: attachment`). Calls `MeasurementService.measure_with_mesh` which now exists. [fitscan/api/routes.py:303](fitscan/fitscan/api/routes.py#L303). _(✅ closed A4, FS-C6 — was 500-ing on every call)_
- [✅] **FS-NIDLO-MEAS-13** — `subject_id` consent gate on every measurement route. Photo only ships to Anthropic when supplied. [routes.py:98](fitscan/fitscan/api/routes.py#L98). _(✅ closed A2, FS-C3)_
- [⬜] **FS-NIDLO-MEAS-05** — `enable_smpl`/`enable_claude` form fields on `/measure/measurements` are **ignored** ([routes.py:271-272](fitscan/fitscan/api/routes.py#L271)) — the call to `service.measure(...)` doesn't forward them. Wire through `MeasurementService.measure` so the toggles actually take effect. Trivial fix; UI exposes both.

### 1.2 — Async pipeline (planned)

- [⬜] **FS-NIDLO-MEAS-14** — Async pipeline. Today every route is `async def` but does 20–60s of synchronous PyTorch/MediaPipe work — one in-flight request stalls one of the four uvicorn workers (audit H1). Two-step fix:
  1. Add `def measure_async(image_bytes_b64, params, idempotency_key)` Celery task in [fitscan/tasks.py](fitscan/fitscan/tasks.py).
  2. Add `POST /api/v1/measure/async` that enqueues + returns `202 {job_id}`; `GET /api/v1/jobs/{id}` returns status + result.
  _Cross-link → BE-NIDLO-MEAS-05 (Laravel polls `/jobs/{id}` OR receives webhook), → FE-NIDLO-MEAS-04 (Reverb subscription on measurement progress)._
- [⬜] **FS-NIDLO-MEAS-15** — Webhook callback to Laravel on async measurement complete. Replaces polling. POST `{job_id, status, measurements?, error?}` to a configurable `LARAVEL_WEBHOOK_URL`. Auth: HMAC-SHA256 over body using shared secret. Depends on FS-NIDLO-MEAS-14. _Cross-link → BE-NIDLO-MEAS-06 (Laravel `/webhooks/fitscan` route + HMAC verify), → OPS-08 (shared-secret rotation)._
- [⬜] **FS-NIDLO-MEAS-16** — Idempotency-key on async measure task: `idempotency_key = sha256(image_bytes + params)`. If a task with the same key exists in `result_backend`, return the cached `AsyncResult` instead of re-running. Audit H3. Depends on FS-NIDLO-MEAS-14.
- [⬜] **FS-NIDLO-MEAS-17** — Retry policy on async measure task: `autoretry_for=(httpx.HTTPError,)`, `retry_backoff=True`, `retry_kwargs={"max_retries": 3}` for the Anthropic call inside the task. DLQ on permanent failure (`task_reject_on_worker_lost=True` is already on at [tasks.py:30](fitscan/fitscan/tasks.py#L30)). Audit H3. Depends on FS-NIDLO-MEAS-14.
- [⬜] **FS-NIDLO-MEAS-18** — Threadpool offload as the pre-Celery quick-win. `def measure_body(...)` (drop the `async`) — FastAPI auto-runs sync routes in a threadpool, immediately fixing the event-loop stall. Audit H1. Land alongside MEAS-14 or as an interim step.

---

## Phase 2 — SMPL fitting & numerical correctness

### 2.1 — Fitting paths

- [✅] **FS-NIDLO-SMPL-01** — HMR2 fitting path (`SMPLBodyFitter._hmr2_fit`). [fitscan/smpl/body_fitter.py:66](fitscan/fitscan/smpl/body_fitter.py#L66). Loads checkpoint from `HMR2_CHECKPOINT_PATH`, runs single forward pass, returns `trimesh.Trimesh`.
- [🟡] **FS-NIDLO-SMPL-02** — SMPLify fallback (`_smplify_fallback`). [fitscan/smpl/body_fitter.py:76](fitscan/fitscan/smpl/body_fitter.py#L76). 100-iteration Adam loop on `betas` only (no thetas, no PCA prior). Works but slow; produces lower-quality fits than HMR2. Decision pending — see SMPL-05.
- [✅] **FS-NIDLO-SMPL-03** — Mesh rescaled to user's `height_cm` via `scale_mesh_to_height`. `PipelineOrchestrator._rescale_mesh_to_height` ([pipeline_orchestrator.py:160](fitscan/fitscan/application/pipeline_orchestrator.py#L160)) calls into [fitscan/smpl/smpl_utils.py:8](fitscan/fitscan/smpl/smpl_utils.py#L8). Was returning template ~1.7m for everyone before A4 — the headline measurement is now correct. _(✅ closed A4, FS-H13)_

### 2.2 — Open SMPL/CV correctness work

- [⬜] **FS-NIDLO-SMPL-04** — Profile + cache `rembg` background-removed images. rembg lazy-loads U2Net ONNX (~150MB) on first call; subsequent calls hit a per-process cache, but model is re-imported inside the function ([cv_pipeline/preprocessor.py:19](fitscan/fitscan/cv_pipeline/preprocessor.py#L19)). Move `from rembg import new_session, remove; _SESSION = new_session("u2net")` to module top-level and pass `session=_SESSION`. Audit H4.
- [⬜] **FS-NIDLO-SMPL-05** — Decide future of `_smplify_fallback`. Either drop it (HMR2 is the supported path; falling back without thetas produces measurably worse fits) or speed it up (replace 100 Adam iterations with closed-form initialization + 30 iterations + LBFGS). Audit H2 (companion). 📐 Snad-decision pending.
- [⬜] **FS-NIDLO-SMPL-06** — Model quantization for ~3× CPU perf. PyTorch dynamic int8 quantization on HMR2 weights, or fp16 if MPS-available. Land alongside OPS-02 (GPU planning). _Cross-link → OPS-02._
- [⬜] **FS-NIDLO-CV-01** — `ScaleCalibrator` numerical guards. `pixel_height` can go negative on inverted poses ([scale_calibrator.py:41](fitscan/fitscan/cv_pipeline/scale_calibrator.py#L41)) — the `< 50` check catches some but not large-negative cases. Add `if pixel_height < 0: raise ScaleCalibrationError(...)`. Bound `px_per_cm` to `[0.5, 60]`. Document the magic `1.07` `HEAD_CORRECTION_FACTOR` ([scale_calibrator.py:27](fitscan/fitscan/cv_pipeline/scale_calibrator.py#L27)) with a citation. Audit H12.
- [⬜] **FS-NIDLO-SMPL-07** — `MeshAnalyzer` per-gender height ratios. `bust` and `chest` both use `0.78` ([smpl/mesh_analyzer.py:23](fitscan/fitscan/smpl/mesh_analyzer.py#L23)) — anatomically male chest line (~0.80) is higher than female bust line (~0.76). Use `gender_profile` to pick the right ratio. Audit H14.
- [⬜] **FS-NIDLO-SMPL-08** — Drop or compute the dummy `slit_length_cm = 40.0` and `cape_cm = 60.0` constants returned with `confidence=0.40, source="default"` ([mesh_analyzer.py:181-189](fitscan/fitscan/smpl/mesh_analyzer.py#L181)). Frontend currently surfaces them as if measured. Audit H14. _Cross-link → FE-NIDLO-MEAS-05 (UI: hide fields where source=default)._
- [⬜] **FS-NIDLO-CV-02** — Smooth silhouette widths. Single-row mask scan ([cv_pipeline/silhouette.py:67](fitscan/fitscan/cv_pipeline/silhouette.py#L67)) overshoots on hair/cloth-fold noise by 5–20px. Median across ±5 rows. Audit H15.
- [⬜] **FS-NIDLO-CV-03** — Replace fixed depth-width ratios `{shoulder:0.42, chest:0.60, ...}` ([silhouette.py:97-106](fitscan/fitscan/cv_pipeline/silhouette.py#L97)) with gender + BMI-conditional lookup. Audit H16. Lower-priority than SMPL-07; flag as defense in depth.

### 2.3 — Regression coverage (PRE-BETA BLOCKER)

- [⬜] **FS-NIDLO-TEST-01** — Numerical regression tests. Pick 5–10 fixture photos with hand-measured ground truth (synthetic SMPL renders with known parameters, OR open-license photos with explicit consent). Commit to `tests/fixtures/`. Add `tests/integration/test_pipeline_accuracy.py` that runs the full pipeline and asserts each field within ±5cm of ground truth. **Without this, every refactor risks silently regressing the headline metric.** Audit T1. _Cross-link → BE-NIDLO-TEST-XX (backend can mock the `/measure` response in unit tests)._

---

## Phase 3 — Avatar generation + 3D

### 3.1 — Generation & export

- [✅] **FS-NIDLO-AVATAR-05** — `AvatarGenerator.generate()` produces GLB + OBJ + thumbnail + meta.json under `data/avatars/{uuid}/`. [fitscan/avatar/generator.py:30](fitscan/fitscan/avatar/generator.py#L30).
- [🟡] **FS-NIDLO-AVATAR-06** — Thumbnail rendering. `_render_thumbnail` writes a blank gray 512×512 PNG ([avatar/generator.py:106](fitscan/fitscan/avatar/generator.py#L106)) — every avatar has the same placeholder. Replace with `trimesh.Scene.save_image` or pyrender front-view. Audit M11. _Cross-link → FE-NIDLO-AVATAR-01 (avatar gallery shows real preview)._
- [🟡] **FS-NIDLO-AVATAR-07** — `_apply_texture` paints all vertices a single hardcoded skin tone `[210, 180, 140]` ([generator.py:91](fitscan/fitscan/avatar/generator.py#L91)). Sample skin tone from the source image (median pixel inside torso silhouette) instead.
- [⬜] **FS-NIDLO-AVATAR-08** — `AvatarGenerator.OUTPUT_DIR = Path("data/avatars")` is relative to cwd ([generator.py:24](fitscan/fitscan/avatar/generator.py#L24)). Anchor to `Settings.data_dir`. Same fix applies to `ObjectStore._put_local` ([object_store.py:59](fitscan/fitscan/infrastructure/object_store.py#L59)) and `DATABASE_URL` default ([result_store.py:27](fitscan/fitscan/infrastructure/result_store.py#L27)). Audit H18.

### 3.2 — Avatar artifact routes

- [✅] **FS-NIDLO-AVATAR-09** — `GET /api/v1/avatar/{id}/glb` ([routes.py:137](fitscan/fitscan/api/routes.py#L137)). Resolves through `_resolve_inside_avatars_root` sandbox.
- [✅] **FS-NIDLO-AVATAR-10** — `GET /api/v1/avatar/{id}/thumbnail` ([routes.py:149](fitscan/fitscan/api/routes.py#L149)).
- [✅] **FS-NIDLO-AVATAR-11** — `POST /api/v1/avatar/{id}/fit-garment` ([routes.py:175](fitscan/fitscan/api/routes.py#L175)). Drapes garment onto stored avatar; returns combined draped GLB. Garment-type map at [clothing_fitter.py:15](fitscan/fitscan/avatar/clothing_fitter.py#L15) supports `dress|blouse|kaba|skirt|trousers|t-shirt`.
- [🟡] **FS-NIDLO-AVATAR-12** — `ClothingFitter.fit_garment` reloads the GLB twice per call ([clothing_fitter.py:37](fitscan/fitscan/avatar/clothing_fitter.py#L37) and one inside `_assess_fit`). Cache once. Audit M10.
- [⬜] **FS-NIDLO-AVATAR-13** — `ClothingFitter.fit_garment` writes draped GLB straight to `data/avatars/{id}` instead of going through `ObjectStore.put`. Bypasses the storage abstraction — won't push to S3 in production. Refactor when next touched.
- [🟡] **FS-NIDLO-AVATAR-14** — `POST /api/v1/avatar/{id}/override` ([routes.py:201](fitscan/fitscan/api/routes.py#L201)) returns `recalculated_measurements: {}` and `convergence_note: "Override not yet implemented in this build."` — the route is honest but the contract is unfulfilled. `InverseSMPLOptimizer.optimise` is a stub at [avatar/inverse_smpl.py:43](fitscan/fitscan/avatar/inverse_smpl.py#L43). Wire the inverse-SMPL Adam loop end-to-end so client overrides actually re-shape the mesh. Audit M12. _Cross-link → FE-NIDLO-MEAS-06 (manual-override slider UI)._
- [⬜] **FS-NIDLO-AVATAR-15** — Validate override field names against the 24-field measurement schema before accepting `dict[str, float]` ([routes.py:203](fitscan/fitscan/api/routes.py#L203)). Today any key passes. Use a Pydantic `OverrideRequest` model.

---

## Phase 4 — Style classification + garment recommendation (Snad's near-term ask)

The Anthropic SDK is already pinned (`anthropic>=0.25.0`). `ClaudeValidator` is the validation-only path; this phase adds a separate classifier surface.

- [⬜] **FS-NIDLO-STYLE-01** — `POST /api/v1/measure/with-recommendations` route. Returns measurements + Claude-generated `style_classification` (`hourglass | rectangle | pear | apple | inverted_triangle`) + `garment_fit_suggestions` (e.g. "A-line skirts emphasise...", "Avoid drop-waist..."). Same input shape as `/measure`. Gated on `subject_id` consent like other Claude paths. _Cross-link → BE-NIDLO-MEAS-07 (Laravel exposes `recommendedStyles` field on `MeasurementType`), → FE-NIDLO-MEAS-07 (designer onboarding UI surfaces fit suggestions)._
- [⬜] **FS-NIDLO-STYLE-02** — `ClaudeStyleClassifier` class (separate from `ClaudeValidator`; same `metadata.user_id`, same skip-on-no-consent semantics, but a distinct prompt + JSON schema). Live at `fitscan/validation/claude_style_classifier.py`. Memoized Anthropic client (see OPS-05).
- [⬜] **FS-NIDLO-STYLE-03** — Prompt design + iteration with sample bodies. Body-shape taxonomy referenced to a fashion-industry source (cite in module docstring). Iterate on a 20-body fixture set; target ≥80% agreement with hand-labelled ground truth. 📐 Snad-decision pending — body-shape categories.
- [⬜] **FS-NIDLO-STYLE-04** — Cache classifier output per `avatar_id` (style is stable for a given avatar). New `style_classifications` table keyed by `avatar_id`; TTL or invalidate on `update_avatar`. Avoids re-charging Anthropic on repeat reads.
- [⬜] **FS-NIDLO-STYLE-05** — Cross-cut: surface results in Laravel + frontend. _Cross-link → BE-NIDLO-MEAS-07, → FE-NIDLO-MEAS-07._

---

## Phase 5 — Production hardening

- [⬜] **FS-NIDLO-OPS-01** — Sentry SDK wired. Currently Loguru-only — Anthropic API errors, SMPL fit failures, measurement value clamps, `safe_500` traces all go nowhere observable. Add `sentry-sdk` dep, init in `api/main.py` lifespan with `traces_sample_rate=0.1`, integrate Loguru sink. _Cross-link → BE-NIDLO-OPS-XX (Laravel Sentry already wired; correlate via `trace_id`). Pre-beta blocker._
- [⬜] **FS-NIDLO-OPS-02** — GPU planning decision before public launch. Contabo VPS is CPU-only — `/measure` runs 20–60s. For private beta with curated designers it's tolerable; public launch needs (a) GPU node (RunPod / Vast / Lambda) keeping Contabo for FastAPI/orchestration, (b) model quantization (FS-NIDLO-SMPL-06), or (c) on-device inference for pose at minimum (Phase 3 deferred). 📐 Snad-decision pending. _Pre-beta open question._
- [⬜] **FS-NIDLO-OPS-03** — Tempfile cleanup on uvicorn hard-kill. Best-effort fix today: `tempfile.NamedTemporaryFile(delete=False)` at [routes.py:63](fitscan/fitscan/api/routes.py#L63) only `unlink`s in the `finally` block — if the worker is SIGKILLed mid-request the JPEG lingers in `/tmp`. Mitigation: dedicated tempdir `/tmp/fitscan_uploads/` + sweep on app startup (lifespan). Audit FS-C3 sub-item e.
- [⬜] **FS-NIDLO-OPS-04** — Anthropic API key rotation procedure. Today `ANTHROPIC_API_KEY` is read once at `MeasurementService` instantiation ([measurement_service.py:47](fitscan/fitscan/application/measurement_service.py#L47)). Document the cold-restart-required model OR move to per-request env read with key versioning. _Cross-link → AD-NIDLO-OPS-XX (admin "rotate Fitscan secrets" runbook)._
- [⬜] **FS-NIDLO-OPS-05** — Memoize Anthropic client. `ClaudeValidator.validate` constructs `Anthropic(api_key=…)` per call ([claude_validator.py:72](fitscan/fitscan/validation/claude_validator.py#L72)) — bad for connection pooling. Build a module-level singleton with `client = Anthropic(timeout=httpx.Timeout(connect=5.0, read=30.0), max_retries=2)`. Audit H5. Add `cache_control={"type":"ephemeral"}` markers on the static `VALIDATION_PROMPT` (~700 tokens) — ~90% cost reduction at scale. _Pair with [Skill: claude-api]._
- [⬜] **FS-NIDLO-OPS-06** — Production DB swap. Verify before launch: `DATABASE_URL` actually points at Postgres + pgvector in prod, not SQLite. The default at [result_store.py:27](fitscan/fitscan/infrastructure/result_store.py#L27) is SQLite `sqlite:///./fitscan.db`. Also add `?sslmode=require` to the Postgres URL (audit H17). _Cross-link → DB-04 (pgvector for embeddings)._
- [⬜] **FS-NIDLO-OPS-07** — Celery beat process management. `make worker` starts a worker; nothing supervises beat. Add Supervisor / systemd units for `celery -A fitscan.tasks worker` + `celery -A fitscan.tasks beat` on Contabo. `deploy/install.sh` provisions worker but not beat — without beat the retention + quota-reset tasks (FS-NIDLO-CELERY-01/02) never fire.
- [⬜] **FS-NIDLO-OPS-08** — Reverb ↔ Fitscan webhook authentication. When async measure completes (FS-NIDLO-MEAS-15), Fitscan POSTs back to Laravel; needs HMAC SHA256 over body using `FITSCAN_WEBHOOK_SECRET` env (paired with Laravel's `SignatureMiddleware`). Both sides rotate together. Depends on FS-NIDLO-MEAS-15. _Cross-link → BE-NIDLO-OPS-XX (Laravel rotation runbook)._
- [⬜] **FS-NIDLO-OPS-09** — Reduce inline `import` statements. 5 ruff warnings persist in [routes.py](fitscan/fitscan/api/routes.py) (e.g. `from fitscan.infrastructure.result_store import AvatarStore` inside route bodies at [routes.py:140](fitscan/fitscan/api/routes.py#L140), [routes.py:152](fitscan/fitscan/api/routes.py#L152), etc.). The original purpose was to avoid circular imports, but the circles can be untied with a small refactor. Pre-existing; not blocking.
- [⬜] **FS-NIDLO-OPS-10** — Make-launched dev port mismatch. [Makefile:19](fitscan/Makefile#L19) sets `PORT := 8008` but README, install.sh, smoke_test.py, sdk/client.py all default to 8000. Smoke test against a make-launched dev server fails to connect. Audit M6. Trivial.
- [⬜] **FS-NIDLO-OPS-11** — Remove committed test/build artifacts from git: `fitscan_test.db` (20KB SQLite), `fitscan_build.log`, `.coverage`. `git rm --cached` + add to `.gitignore`. Audit M15.

---

## Phase 6 — Database schema correctness

- [⬜] **FS-NIDLO-DB-01** — Scaffold Alembic. `alembic init alembic` (file present at [fitscan/alembic/](fitscan/alembic/) but not configured), generate initial revision from existing `AvatarRecord` + `APIKeyRecord` schema, commit. `deploy/install.sh:89` runs `alembic upgrade head` on every deploy and CI runs it too — both currently break with `Path doesn't exist: alembic.ini`. Audit H7.
- [⬜] **FS-NIDLO-DB-02** — Remove `Base.metadata.create_all(engine)` import side-effect at [result_store.py:72](fitscan/fitscan/infrastructure/result_store.py#L72). Replace with Alembic migrations. Anyone who imports `fitscan` mutates the DB today. Audit H6/H7.
- [⬜] **FS-NIDLO-DB-03** — Switch JSON columns to `JSONB`. `measurements`, `betas`, `thetas` are `Text` and round-trip through `json.dumps`/`json.loads` ([result_store.py:44-46](fitscan/fitscan/infrastructure/result_store.py#L44)). Native JSONB enables indexed queries ("find avatars with waist > X") + fixes the lossy `_record_to_domain` ([line 168](fitscan/fitscan/infrastructure/result_store.py#L168)) that returns `BodyMeasurements(source="stored")` discarding all stored values. Audit H8. _Cross-link → BE-NIDLO-MEAS-XX (Laravel cached measurement read uses real values)._
- [⬜] **FS-NIDLO-DB-04** — Indexes + FKs. Today only `avatars.avatar_id` (PK) and `api_keys.key_hash` (unique) are indexed. Add `owner_id` / `api_key_id` FK on `avatars` so "list this user's avatars" is possible; add `created_at` index for the retention query. Add `pgvector` `embedding` column for "similar profiles" use-case advertised in README. Audit H9.
- [⬜] **FS-NIDLO-DB-05** — TZ-aware datetimes. `created_at`/`updated_at`/`last_used_at` use `DateTime` (TZ-naive) but produce TZ-aware values via `datetime.now(timezone.utc)`. Switch to `DateTime(timezone=True)`. Audit H9.
- [⬜] **FS-NIDLO-DB-06** — `AvatarStore.update_avatar` raises `KeyError` ([result_store.py:106](fitscan/fitscan/infrastructure/result_store.py#L106)). Inconsistent with rest of module — replace with a domain exception (`AvatarNotFoundError`).
- [⬜] **FS-NIDLO-DB-07** — Route `DATABASE_URL` through `get_settings()` instead of `os.getenv` at module import time ([result_store.py:27](fitscan/fitscan/infrastructure/result_store.py#L27)). Currently any test that imports `result_store` before `pytest_configure` loads `.env.test` silently binds to the dev DB. Audit H6.

---

## Phase 7 — Tests

- [✅] **FS-NIDLO-TEST-00** — 68 existing pytest tests, 55%+ coverage above the gate in [pyproject.toml](fitscan/pyproject.toml).
- [⬜] **FS-NIDLO-TEST-01** — Numerical regression tests against fixture images. **Pre-beta blocker**, see [Phase 2.3](#23--regression-coverage-pre-beta-blocker). _(duplicate ID kept intentional — same task surfaces in both phases for navigability)_
- [⬜] **FS-NIDLO-TEST-02** — Integration tests for `/measure/avatar`, `/avatar/{id}/glb`, `/avatar/{id}/thumbnail`, `/avatar/{id}/fit-garment`, `/avatar/{id}/override`. **Currently zero coverage** — a single test would have caught FS-C6 (`measure_with_mesh` AttributeError) on day one. Audit T2.
- [⬜] **FS-NIDLO-TEST-03** — Celery task tests: `cleanup_old_avatars` and `reset_api_key_quotas` with a fake clock (`freezegun`). Today both are wired but no test exercises them end-to-end. _Cross-link → CELERY-01, CELERY-02._
- [⬜] **FS-NIDLO-TEST-04** — `mesh_analyzer.py` SMPL-path tests. 14% coverage — none of the existing tests exercise the SMPL path with a real mesh. The mesh-slicing math (~70% of the 24 output fields) is untested. Audit T1 + T9.
- [⬜] **FS-NIDLO-TEST-05** — Full-pipeline benchmark with timing assertions in `tests/benchmark/`. Assert `total_time < 60s` on CPU and `< 5s` on GPU (when GPU mode lands per OPS-02). Catches perf regressions early. Audit T1 (companion).
- [⬜] **FS-NIDLO-TEST-06** — `RuleValidator` boundary tests. Each rule in `RELATIONAL_RULES` ([validation/rule_validator.py:44-72](fitscan/fitscan/validation/rule_validator.py#L44)) needs at-bound + just-out-of-bound cases. Audit T6.
- [⬜] **FS-NIDLO-TEST-07** — `MeasurementFuser` 3-source weighted-average test. The agreement-bonus path in `ConfidenceScorer.score` ([fusion/confidence.py:46-51](fitscan/fitscan/fusion/confidence.py#L46)) is only exercised at 0/1 source today. Audit T7.
- [⬜] **FS-NIDLO-TEST-08** — `ScaleCalibrator.calibrate_from_reference_object` synthetic A4-card fixture. The Canny+approxPolyDP+minAreaRect chain at [scale_calibrator.py:61-83](fitscan/fitscan/cv_pipeline/scale_calibrator.py#L61) is brittle — needs a known-good fixture and ±5% tolerance test. Audit T8.
- [⬜] **FS-NIDLO-TEST-09** — `ImageLoader.MAX_FILE_SIZE_MB` branch test. The size-rejection path at [image_loader.py:21](fitscan/fitscan/infrastructure/image_loader.py#L21) is uncovered. Audit T5.
- [⬜] **FS-NIDLO-TEST-10** — Fix `scripts/smoke_test.py:46` — asserts unauthenticated `/measure` returns 401, but auth is now enforced at app level (and the previous shape was inverted from reality). Audit T4. Land alongside fixing the port mismatch (OPS-10).
- [⬜] **FS-NIDLO-TEST-11** — Raise `--cov-fail-under` from 55 to 80 in [pyproject.toml](fitscan/pyproject.toml). 100% on `MeasurementFuser`, `ScaleCalibrator`, `RuleValidator` (pure-Python math, no excuses). Audit T9. Land last — depends on TEST-01..09.

---

## Phase 8 — CLI

- [⬜] **FS-NIDLO-CLI-01** — `subprocess.run` cwd hygiene. [cli/fitscan_cli.py:109](fitscan/cli/fitscan_cli.py#L109) and [scripts/generate_docs.py:174](fitscan/scripts/generate_docs.py#L174) — pass explicit `cwd=`, `check=True`, capture failures. Low practical impact (CLI only) but flag for defense in depth. Audit FS-C8.
- [⬜] **FS-NIDLO-CLI-02** — `--data` typer option at [cli/fitscan_cli.py:99](fitscan/cli/fitscan_cli.py#L99) is declared but never consumed. Either wire it through (anchor benchmark-fixture path) or remove.
- [⬜] **FS-NIDLO-CLI-03** — `sdk/client.py` connection leak. `httpx.Client` at [sdk/client.py:26](fitscan/fitscan/sdk/client.py#L26) is closed only via explicit `close()`/`__exit__`. `health()` failure path swallows `Exception` ([client.py:89](fitscan/fitscan/sdk/client.py#L89)) and leaks the connection. Audit M8.

---

## Phase 9 — Phase 2 / 3 deferred (DO NOT implement)

Per [project_phase_and_scope.md](../../.claude/projects/-Users-mac-Projects-mine-stitchub/memory/project_phase_and_scope.md). Listed for grep-ability — flag as deferred when audits surface them, do not start.

- [🔒] **FS-NIDLO-OPS-20** — GPU autoscaling. Single VPS in beta. Phase 3.
- [🔒] **FS-NIDLO-OPS-21** — Multi-region deployment. Single Contabo VPS. Phase 3.
- [🔒] **FS-NIDLO-MEAS-30** — On-device inference fallback (browser TF.js MediaPipe). Privacy + cost win, big infra work. Phase 3.
- [🔒] **FS-NIDLO-MEAS-31** — Video stream support. Image-only today. Phase 2+.
- [🔒] **FS-NIDLO-AVATAR-30** — Privacy-preserving compute (measure on device, ship only numbers, never the photo). Phase 3 alignment.

---

## Cross-cutting open backlog (audit-driven, not phased)

These are minor / defense-in-depth items from [audits/fitscan/99_FINDINGS.md](../audits/fitscan/99_FINDINGS.md) that don't justify their own task ID but should be picked up while in the area:

- [⬜] **M1** — 14 broad `except Exception` swallow stack traces. Add `exc_info=True` to logger calls; tighten exception types where safe. Locations: [pose_detector.py:81](fitscan/fitscan/cv_pipeline/pose_detector.py#L81), [sdk/client.py:89](fitscan/fitscan/sdk/client.py#L89), [generator.py:54](fitscan/fitscan/avatar/generator.py#L54), [generator.py:103](fitscan/fitscan/avatar/generator.py#L103), [clothing_fitter.py:90](fitscan/fitscan/avatar/clothing_fitter.py#L90), [routes.py:127](fitscan/fitscan/api/routes.py#L127), [routes.py:291](fitscan/fitscan/api/routes.py#L291), [routes.py:366](fitscan/fitscan/api/routes.py#L366), [pipeline_orchestrator.py:134](fitscan/fitscan/application/pipeline_orchestrator.py#L134), [pipeline_orchestrator.py:151](fitscan/fitscan/application/pipeline_orchestrator.py#L151), [body_fitter.py:55](fitscan/fitscan/smpl/body_fitter.py#L55), [mesh_analyzer.py:202](fitscan/fitscan/smpl/mesh_analyzer.py#L202), [mesh_analyzer.py:225](fitscan/fitscan/smpl/mesh_analyzer.py#L225), [claude_validator.py:114](fitscan/fitscan/validation/claude_validator.py#L114).
- [⬜] **M2** — Determinism: no `torch.manual_seed`, `np.random.seed`, or `random.seed` is set anywhere. SMPLify Adam loop is deterministic given deterministic Torch ops, but cuDNN nondeterminism + Python hash randomization can cause inter-run variance.
- [⬜] **M3** — Cite magic numbers in measurement math. `HEAD_CORRECTION_FACTOR = 1.07` ([scale_calibrator.py:27](fitscan/fitscan/cv_pipeline/scale_calibrator.py#L27)), `CIRCUMFERENCE_HEIGHTS` / `ARM_HEIGHT_RATIOS` / `LANDMARK_HEIGHTS` ([mesh_analyzer.py:20-45](fitscan/fitscan/smpl/mesh_analyzer.py#L20)), sleeve arc-length factor `1.12` ([mesh_analyzer.py:255](fitscan/fitscan/smpl/mesh_analyzer.py#L255)), `depth_width_ratios` ([silhouette.py:98-105](fitscan/fitscan/cv_pipeline/silhouette.py#L98)), `MEASUREMENT_BOUNDS_CM` ([rule_validator.py:12](fitscan/fitscan/validation/rule_validator.py#L12)).
- [⬜] **M4** — `MeshAnalyzer._geodesic_nape_to_waist` is misnamed — does straight-line distance ([mesh_analyzer.py:241-249](fitscan/fitscan/smpl/mesh_analyzer.py#L241)). Rename to `_nape_to_waist_vertical` and re-introduce true geodesic when needed.
- [⬜] **M5** — Dead config: `LOG_LEVEL`, `CONFIDENCE_THRESHOLD`, `MAX_IMAGE_SIZE_MB`, `FREE_TIER_*`, `STRIPE_*` declared in `.env.example` but never read. Wire up or delete. Note: `MAX_IMAGE_SIZE_MB` collides with hardcoded `MAX_FILE_SIZE_MB = 10` in [image_loader.py:15](fitscan/fitscan/infrastructure/image_loader.py#L15).
- [⬜] **M9** — `ProcessedImage.original` ndarray is not copied ([domain/models.py:154-160](fitscan/fitscan/domain/models.py#L154)). Frozen dataclass guards the reference but not the underlying data — mutation in any downstream stage propagates. Document or `.copy()`.
- [⬜] **M13** — `lru_cache(maxsize=1)` on `get_measurement_service` ([api/dependencies.py:13](fitscan/fitscan/api/dependencies.py#L13)) is per-process; with `--workers 4` you have 4 caches and each worker pays the lazy model load on its first request. Add an `app.lifespan` startup hook that calls `_get_service()` to warm.
- [⬜] **M14** — `__init__.py` files are mostly empty. Top-level `fitscan/__init__.py` exports `__version__` only. No re-exports of public API → consumers use deeply-qualified imports. Maintainability nit.
- [⬜] **H11** — `MeasurementFuser._weighted_average` numerical edges. `quality_multiplier = validation.photo_quality_score / 10.0` at [confidence.py:38](fitscan/fitscan/fusion/confidence.py#L38) — if Claude returns `photo_quality_score=0`, every confidence collapses to 0 silently and downstream callers may treat 0-confidence as authoritative. Clamp to `[0.1, 1.0]`.
- [⬜] **M16** — Inconsistent degraded-mode signaling. `SMPLBodyFitter` falls back to SMPLify; `PoseDetector` raises; `ImagePreprocessor` falls back to threshold removal. The API response just lowers confidence without a structured `degraded_modes: ["rembg_unavailable"]` field. Surface it.

---

## QA roadmap-gap addendum (audit gap, 2026-05-04)

Sourced from [../backend/audits/qa-roadmap-gaps-2026-05-04.md](../backend/audits/qa-roadmap-gaps-2026-05-04.md). New ⬜ entries identified during QA CSV generation that warrant roadmap tracking.

- [⬜] **FS-NIDLO-MEAS-19** — Camera permission denied UX path. Today the upload flow silently fails when Safari (or any browser) denies camera. Should fall through to file-picker mode + "use a saved photo" guidance. _(audit gap, 2026-05-04 — QA-GAP-A-16)_
- [⬜] **FS-NIDLO-MEAS-20** — Pose-quality preview / silhouette overlay on the upload preview screen so users self-correct before submitting. Reduces 422 "no person detected" rates (failure-modes §4.3). _(audit gap, 2026-05-04 — QA-GAP-A-17)_
- [⬜] **FS-NIDLO-OPS-12** — QA-test target for per-key rate limit on `/measure*` (already tracked as `FS-NIDLO-AUTH-06`). Cross-link in QA CSV. Each measurement is 20–60s of CPU; QA load test should confirm the 429 fires. _(audit gap, 2026-05-04 — QA-GAP-A-18)_
- [⬜] **FS-NIDLO-VALID-03** — Confidence-disagreement signal in the response (`degraded_modes: ["validators_disagree"]`) when rule + Claude validators diverge above a threshold. Audit M16 (cross-cutting backlog). Frontend can then render a "manual review recommended" badge. _(audit gap, 2026-05-04 — QA-GAP-A-19)_

---

## Cross-references

- **Backend tasks**: [../backend/TASKS.md](../backend/TASKS.md)
- **Frontend tasks**: [../frontend/ROADMAP.md](../frontend/ROADMAP.md)
- **Admin tasks**: [../backend/ROADMAP_ADMIN.md](../backend/ROADMAP_ADMIN.md)
- **Audit findings**: [../audits/fitscan/99_FINDINGS.md](../audits/fitscan/99_FINDINGS.md) — 10 Critical + 18 High + 16 Medium + 9 Test gaps
- **Per-module audits**: [../audits/fitscan/module_api.md](../audits/fitscan/module_api.md), [module_application.md](../audits/fitscan/module_application.md), [module_avatar.md](../audits/fitscan/module_avatar.md), [module_cv_pipeline.md](../audits/fitscan/module_cv_pipeline.md), [module_smpl.md](../audits/fitscan/module_smpl.md), [module_validation.md](../audits/fitscan/module_validation.md), [module_infrastructure.md](../audits/fitscan/module_infrastructure.md)
- **Slim ROADMAP**: [ROADMAP.md](ROADMAP.md) (exec summary; this file is granular)
- **Operations runbook**: [../docs/journeys/03-admin.md](../docs/journeys/03-admin.md) §4 Fitscan ops
- **Failure-mode map**: [../docs/journeys/05-failure-modes.md](../docs/journeys/05-failure-modes.md) §4 Fitscan failures
- **Client journey**: [../docs/journeys/01-client.md](../docs/journeys/01-client.md) §6 Body measurements
- **Memory**: `~/.claude/projects/-Users-mac-Projects-mine-stitchub/memory/project_audits.md`, `project_vendors.md`, `project_phase_and_scope.md`
- **Reference roadmap (XLent format)**: `/Users/mac/Projects/xlent/mobile/ROADMAP_MOBILE.md`
