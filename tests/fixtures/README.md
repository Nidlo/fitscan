# Fitscan numerical regression fixtures

**Status:** scaffold-only. The harness exists at
[`tests/integration/test_pipeline_accuracy.py`](../integration/test_pipeline_accuracy.py)
and is `pytest.skip`-gated until fixtures land here.

## What goes in this directory

8–10 photos with **hand-measured ground truth**, captured under
controlled conditions:

- Front-facing, full-body, plain background, fitted clothing
- Each photo has an accompanying `<name>.json` next to it carrying
  the ground-truth measurements **in centimetres**, plus the user's
  height (which the pipeline needs for scale calibration)

Acceptable sources:

1. **Synthetic SMPL renders.** Render a SMPL-X mesh with known beta /
   theta parameters at varied stature + body shape, save the rendered
   image + the ground-truth measurements computed from the mesh
   itself. Highest signal-to-noise; no consent issues.
2. **Open-license photos with explicit consent.** Real users who have
   signed off on their photo + measurements being committed to the
   open-source repo. Requires the user-supplied ground truth + their
   stated height_cm.

**Do NOT commit:**

- Real customer body photos from production
- Photos lifted from social media / image search
- Scraped datasets without per-image licence verification

## Filename pattern

```
fixtures/
  README.md             # this file
  fixture_001.jpg       # the photo
  fixture_001.json      # ground truth
  fixture_002.jpg
  fixture_002.json
  ...
```

## Ground-truth JSON shape

```json
{
  "name": "fixture_001",
  "source": "synthetic|consented_real",
  "height_cm": 168.0,
  "gender_profile": "ladies",
  "tolerance_cm": 5.0,
  "ground_truth": {
    "bust_cm": 91.5,
    "waist_cm": 71.2,
    "hip_cm": 96.8,
    "shoulder_cm": 41.5,
    "neck_cm": 33.2,
    "inseam_cm": 78.0,
    "shoulder_to_waist_cm": 39.5
  },
  "notes": "Synthetic SMPL render — beta=[0.5, -0.3, ...], theta=A-pose."
}
```

Required keys per fixture: `height_cm`, `gender_profile`, `tolerance_cm`,
`ground_truth` (with at least `bust_cm`, `waist_cm`, `hip_cm`).

`tolerance_cm` is the per-field accept band — defaults to ±5cm but a
fixture can tighten if its ground truth is high-confidence (e.g.
synthetic renders should run at ±1cm).

## How to add a fixture

1. Capture / render the photo under the conventions above.
2. Hand-measure (or compute from the SMPL params) the ground truth.
3. Save photo + JSON in this directory using the filename pattern.
4. Re-run `pytest tests/integration/test_pipeline_accuracy.py` —
   the harness auto-discovers every `fixture_*.jpg` next to a
   matching `.json`.

The pipeline-accuracy test will run the photo through the full
measurement pipeline and assert each named field is within
`tolerance_cm` of `ground_truth[field]`.

## Why this is a pre-beta blocker

Without these fixtures, the audit's biggest gap stays open:
**every refactor risks silently regressing the headline metric**.
SMPL + CV + scale-calibrator changes can drift bust/waist/hip by
±10cm without any test in the suite catching it. FS-NIDLO-TEST-01.

## Snad action items

- [ ] Decide source: synthetic SMPL renders vs consented real photos
      (or a mix; a small synthetic baseline is the cheapest start).
- [ ] If real photos: produce a one-page consent form covering
      open-source commit + indefinite retention.
- [ ] Capture / render the first 8–10 fixtures + ground truth.
- [ ] Drop them in this directory; the harness picks them up on
      next CI run.
