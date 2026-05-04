# Nidlo Fitscan QA test checklist

Canonical QA test checklist for the **Fitscan** AI body-measurement service (FastAPI + PyTorch + MediaPipe + SMPL-X + Anthropic SDK).

## File

[QA_TEST_CHECKLIST.csv](QA_TEST_CHECKLIST.csv) — 115 rows, 21 columns. Mirrored to `../qa/fitscan.csv` at workspace root.

## Quick-start

1. Open the CSV in Google Sheets.
2. Filter by `feature_area` (Auth / Measurements / Avatar / CV Pipeline / SMPL / Validation / Storage / Database / Celery / Operations / Failure modes / Tests / CLI / Style / Privacy / Integration).
3. For each row, run the `test_steps` (curl commands, pytest invocations, manual checks) and confirm `expected_result`.
4. Update `status`, `tester`, `tested_on`, `notes_or_bug_link`.

Full schema reference: [../../qa/README.md](../../qa/README.md).

## What's covered

| Feature area | Rows |
|---|---|
| Auth (X-API-Key, quotas, rate limit) | 10 |
| Measurement endpoints (synchronous + planned async) | 23 |
| Avatar (generation, GLB / OBJ / thumbnail / fit-garment / override / RTBF) | 15 |
| CV pipeline (rembg, scale calibration, silhouette) | 4 |
| SMPL fitting (HMR2, SMPLify, mesh analysis) | 6 |
| Validation (rule + Claude validators, consent) | 7 |
| Storage (MinIO, S3 SSE) | 2 |
| Database (Alembic, JSONB, indexes) | 6 |
| Celery (cleanup, quota reset, beat supervision) | 4 |
| Operations (Sentry, GPU, key rotation, prod DB) | 8 |
| Failure modes from journey doc §4 | 3 |
| Tests (regression, coverage, fixtures) | 10 |
| CLI / SDK | 3 |
| Style classification (Phase 4 — Snad's planned ask) | 4 |
| Privacy (consent, RTBF, retention) | 4 |
| Integration with Backend + Admin + Frontend | 6 |

## Coverage assertions

- ✅ Every endpoint in `fitscan/api/routes.py` has at least one row.
- ✅ Every CV / SMPL / validator module has at least one row.
- ✅ Every Celery task is exercised.
- ✅ Failure modes from [docs/journeys/05-failure-modes.md §4](../../docs/journeys/05-failure-modes.md) covered.
- ✅ Every Phase from [TASKS.md](../TASKS.md) (Phases 0–8) represented.
- ✅ Roadmap gaps from the gap audit tagged `roadmap-gap`.
- ✅ Cross-system integration rows reference frontend / backend / admin task IDs.

## Pre-beta blockers explicitly tracked

| Row | What it tests | Status |
|---|---|---|
| QA-FS-TEST-001 | Numerical regression against fixture images | ⬜ |
| QA-FS-OPS-001 | Sentry SDK wired | ⬜ |
| QA-FS-OPS-002 | GPU planning decision | ⬜ |
| QA-FS-OPS-005 | Prod DATABASE_URL = Postgres + sslmode=require | ⬜ |
| QA-FS-CELERY-003 | Beat process supervised in production | ⬜ |
| QA-FS-AUTH-006 | `/keys/register` lockdown | ⬜ |

These rows will stay `Not Tested` until the underlying tasks land. They serve as a launch-day gate.

## When to update

- A new endpoint or service module ships → add rows for happy + edge + error.
- An audit finding closes → flip the row to `expected_result` reflecting the closed state, bump `last_updated`, optionally remove the `(today: ...)` parenthetical.
- A new measurement field is added → exercise it via at least one row.

## Sync command

```bash
cp fitscan/qa/QA_TEST_CHECKLIST.csv qa/fitscan.csv
```

---

**See also:** [../../qa/README.md](../../qa/README.md) for the full schema, status workflow, and master regen command.
