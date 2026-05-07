"""Numerical regression harness against fixture photos (FS-NIDLO-TEST-01).

**Status:** scaffold. Auto-discovers fixtures in `tests/fixtures/`
following the convention documented in
[tests/fixtures/README.md](../fixtures/README.md). When no fixtures
are present (the current state) every test is `pytest.skip`-ged at
collection time, so the harness ships green and slots in real
fixtures the moment Snad lands them.

Why this is a pre-beta blocker:
  Every refactor of CV / SMPL / scale-calibration / mesh-analyzer
  can shift bust/waist/hip by 5-15cm without any test in the
  current suite catching it. This harness is the safety net.

How fixtures land:
  Drop `fixture_*.jpg` + matching `fixture_*.json` into
  `tests/fixtures/`. See the README there for the JSON shape and
  what counts as legitimate source material (consent matters).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _discover_fixtures() -> list[tuple[Path, dict[str, Any]]]:
    """Return [(image_path, ground_truth_dict)] for every paired
    fixture under tests/fixtures/. Empty list when none exist —
    the test then `skip`s rather than failing."""
    if not FIXTURES_DIR.exists():
        return []
    out: list[tuple[Path, dict[str, Any]]] = []
    for jpg in sorted(FIXTURES_DIR.glob("fixture_*.jpg")):
        json_path = jpg.with_suffix(".json")
        if not json_path.exists():
            continue
        with json_path.open() as f:
            payload = json.load(f)
        out.append((jpg, payload))
    return out


_FIXTURES = _discover_fixtures()


@pytest.mark.skipif(
    not _FIXTURES,
    reason=(
        "No regression fixtures committed yet — see tests/fixtures/README.md "
        "for what to add. The harness auto-discovers fixtures named "
        "fixture_NNN.jpg next to fixture_NNN.json."
    ),
)
@pytest.mark.parametrize(
    "image_path,ground_truth",
    _FIXTURES,
    ids=[p.stem for p, _ in _FIXTURES] if _FIXTURES else [],
)
def test_pipeline_within_tolerance(
    image_path: Path, ground_truth: dict[str, Any]
) -> None:
    """Run the full pipeline and assert each ground-truth field is
    within `tolerance_cm` of the measured value. Per-fixture tolerance
    lets synthetic SMPL renders run tight (±1cm) while consented real
    photos stay at the audit-suggested ±5cm."""
    from fitscan.application.measurement_service import MeasurementService

    tolerance = float(ground_truth.get("tolerance_cm", 5.0))
    height_cm = float(ground_truth["height_cm"])
    gender = ground_truth.get("gender_profile", "ladies")
    expected = ground_truth["ground_truth"]

    service = MeasurementService()
    result = service.measure(
        image_path=image_path,
        height_cm=height_cm,
        gender_profile=gender,
    )

    failures: list[str] = []
    for field, expected_cm in expected.items():
        measured = result.get(field)
        if measured is None:
            failures.append(f"{field}: not measured")
            continue
        # `result` is the BodyMeasurements.to_dict() shape — each
        # field is `{"value_cm": ..., "confidence": ..., "source": ...}`.
        measured_cm = (
            measured.get("value_cm") if isinstance(measured, dict) else measured
        )
        if measured_cm is None:
            failures.append(f"{field}: null value_cm")
            continue
        diff = abs(float(measured_cm) - float(expected_cm))
        if diff > tolerance:
            failures.append(
                f"{field}: measured={measured_cm:.1f}cm "
                f"expected={expected_cm:.1f}cm Δ={diff:.1f}cm "
                f"(tolerance={tolerance:.1f})"
            )

    if failures:
        formatted = "\n".join(f"  - {line}" for line in failures)
        pytest.fail(
            f"Numerical regression in {image_path.name}:\n{formatted}"
        )
