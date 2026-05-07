# Fitscan continuous-learning roadmap

> Step 1 ✅ — data plumbing (this commit, lives in the Laravel backend)
> Step 2 ⬜ — calibrator model trainer (not built; see [Phases](#phases))
> Step 3 ⬜ — inference hook + model registry (not built)
> Step 4 ⬜ — active-learning loop (not built)

This directory is reserved for the Phase 5 Python training pipeline.
Today it carries the README only — the actual `train_calibrator.py` /
`evaluate.py` / `serve.py` modules will land alongside the model
itself once accumulated correction data justifies the build.

---

## What we're building toward

A small **calibration model** that runs after the existing Fitscan
pipeline and corrects systematic biases learned from designer +
client edits to AI-generated measurements:

```
photo ─► [Fitscan: SMPL + Claude] ─► raw_AI_values
                                          │
                       landmarks ─────────┤
                       user_height ──────► [Calibrator] ─► corrected_values
                       gender_profile ────┘
```

Architecture: **XGBoost or a small 3-layer MLP**. Not a vision
model — the features are already extracted by the existing pipeline;
the calibrator is regression on top. ~80% of the value of self-built
ML for ~20% of the effort of fine-tuning the foundation model.

See [`fitscan/LAUNCHING.md` §8](../../LAUNCHING.md) for the wider
SaaS context, and Snad's strategic message in the Sprint F1 design
discussion for the why.

---

## Where the data comes from

The Laravel backend captures three columns side-by-side per
measurement:

| Column | What it is |
|---|---|
| `ai_baseline_mm` | Frozen Fitscan output at extraction time |
| `data_mm` | Final accepted values (after any human edits) |
| `manual_overrides_mm` | Sparse diff = exactly where humans disagreed |

**Step 1 (this commit) makes that queryable.** The Laravel side now
ships:

- A `correction_examples` Postgres view that explodes the JSONB
  columns into per-field rows
- An artisan command that exports those rows as JSONL

```bash
# In the Laravel backend repo:
php artisan fitscan:export-corrections
# → Exported 0 rows to fitscan-training/full-2026-05-07-181500.jsonl

php artisan fitscan:export-corrections --since=2026-04-01
# → Exported 142 rows to fitscan-training/since-2026-04-01--…jsonl
```

Files land in `storage/app/private/fitscan-training/` on the Laravel
host. The Python trainer (when built) will consume them via SCP /
shared volume — never via direct DB access, keeping body-PII inside
the Laravel boundary.

### JSONL line shape

Each line is one (measurement × field) row:

```json
{
  "example_id": "019e0250-aaaa-bbbb-cccc-…",
  "subject_anon": "9f86d081884c7d65",
  "section": "upper_body",
  "field": "bust",
  "ai_value_mm": 920,
  "final_value_mm": 905,
  "delta_mm": -15,
  "was_corrected": true,
  "landmarks": {
    "left_shoulder": {"x": 0.35, "y": 0.21, "visibility": 0.92},
    "…": "…"
  },
  "photo_disk": "local",
  "photo_public_id": "measurements/abc123.jpg",
  "extracted_at": "2026-05-07T10:00:00Z",
  "confirmed_at": "2026-05-07T10:42:13Z"
}
```

The `subject_anon` is `sha256(user_id)[:16]`. Same user → same hash,
so a stratified train/test split (don't put one user's measurements
in both splits) groups correctly **without** the real `user_id` ever
leaving the Laravel host. Models trained on this data carry no
mappable identifier.

### Privacy contract

- Soft-deleted measurements (RTBF path) are excluded by the view
  filter `m.deleted_at IS NULL` — never appear in the export
- Hard-purged users (post-30-day recovery cron) drop out of the next
  export naturally — no row to project
- The Python trainer **must** use only the JSONL files, never the
  raw DB. This keeps body-PII inside the Laravel boundary and gives
  Snad one place to verify the privacy story
- Models trained on a snapshot containing a now-deleted user's data
  must be retired when the user requests deletion. The `model_versions`
  table (Step 3) tracks training-set hashes for this lookup

---

## Phases

### Phase 1 — data plumbing ✅

- [x] `correction_examples` Postgres view (Laravel migration)
- [x] `php artisan fitscan:export-corrections` (Laravel command)
- [x] Anonymisation (sha256-truncated `subject_anon`)
- [x] `--since` flag for incremental exports
- [x] Soft-deleted exclusion + RTBF privacy contract
- [x] Tests pinning view shape + JSONL contract

### Phase 2 — model trainer ⬜

```
fitscan/training/
├── __init__.py
├── data_loader.py     # JSONL → PyTorch Dataset
├── features.py        # landmarks → feature vectors
├── model.py           # XGBoost or 3-layer MLP
├── train.py           # one-shot training run
├── evaluate.py        # held-out eval against ground truth
└── serve.py           # load + run a deployed checkpoint
```

Decision points before building:

1. **Minimum data threshold.** ~500 corrections per (gender × field)
   combo, per the calibrator strategy. ~3 months of beta usage at
   ~50 measurements/month per active user, ~10 active users. Don't
   start training before this; the model will overfit to noise.
2. **Architecture pick.** XGBoost is simpler to debug, ships as a
   single .ubj file, and is fast to retrain. MLP is more flexible
   (can encode landmarks as a graph, add embeddings) but heavier
   ops. Recommend XGBoost for v1; revisit if accuracy plateaus.
3. **Eval set discipline.** Hold out 20% of measurements (stratified
   by `subject_anon`) as validation; never train on them. A separate
   small audit set (50 photos with tape-measure ground truth) stays
   off-limits for any training run — that's the canonical "did the
   model improve?" check.

### Phase 3 — inference hook + model registry ⬜

- New `calibrator_models` Postgres table (Laravel migration):
  timestamp, training_set_size, eval_metric, weights_path, is_active
- New method `MeasurementService.measure()` post-processor that
  loads the active model and applies per-field deltas
- Identity fall-back when no model is `is_active=true` — keeps
  the deploy-ready boolean simple
- Shadow-mode flag: predict but don't apply for the first N calls
  per new model, log the delta vs un-calibrated for offline review

### Phase 4 — active learning ⬜

- Calibrator emits a confidence score per field
- Low-confidence predictions get flagged in the designer review UI:
  "Fitscan isn't sure about this — please scrutinise"
- Designer corrections on flagged predictions feed back into the next
  training run with extra weight (higher information value per row)
- Closes the loop: the model's own uncertainty becomes the prioritisation
  signal for human attention

---

## Open questions Snad still owns

- **When to start training.** Need ~3 months of accumulated correction
  data before Phase 2 has signal. Earlier than that = overfitting to
  noise.
- **GPU node decision** (FS-NIDLO-OPS-02). Calibrator training is CPU-
  fine even at ~10k examples; a GPU only matters for fine-tuning the
  foundation SMPL model (Strategy B, deferred).
- **Anthropic costs vs in-house validator.** Once the calibrator
  proves valuable, Strategy C (distill Claude into a small in-house
  vision model) might pay back. Decide after 6 months of paid
  Anthropic usage data.

---

## Operational notes

- The export script writes one file per run; existing files aren't
  rotated. Operators can `find storage/app/private/fitscan-training -mtime +90 -delete`
  to keep the dir tidy
- The view auto-updates as new measurements land; no migration churn
- A future Phase 2 trainer should **read** the JSONL files exclusively
  — never query the DB directly. This is the privacy boundary
- Per-call cost of running the calibrator (Phase 3): ~negligible
  (<5ms CPU on the Contabo box). No measurable user-visible latency
  hit
