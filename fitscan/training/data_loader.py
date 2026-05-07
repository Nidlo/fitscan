"""JSONL → in-memory training examples for the calibrator pipeline.

Reads the files the Laravel backend exports via
`php artisan fitscan:export-corrections`. Pure data loading — no
features yet, no model. Phase 2 of the continuous-learning roadmap
(see [README.md](README.md)) consumes this loader; today it ships
as the contract validator + smoke test for the JSONL shape.

Why a separate module: keeps the privacy boundary clean. The
trainer NEVER reads from the Laravel database; it reads only from
the on-disk JSONL files Laravel decided to share. Soft-deleted
users never make it into an export, and the retention sweep
bounds how long an export can survive an RTBF request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class CorrectionExample:
    """One (measurement × field) row from the export.

    Mirrors the JSON line shape documented in
    `fitscan/training/README.md` ("JSONL line shape" section). Frozen
    so a downstream feature builder can hash these for caching.
    """

    example_id: str
    subject_anon: str
    section: str
    field: str
    ai_value_mm: int
    final_value_mm: int
    delta_mm: int
    was_corrected: bool
    landmarks: dict | None
    photo_disk: str | None
    photo_public_id: str | None
    extracted_at: str | None
    confirmed_at: str | None


def load_jsonl(path: str | Path) -> list[CorrectionExample]:
    """Eagerly parse a JSONL export file into a list. Convenient for
    small files (<100k rows). Use `iter_jsonl` for streaming when
    files outgrow RAM."""
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[CorrectionExample]:
    """Stream JSONL rows. Skips blank lines silently — the export
    writer occasionally emits a trailing newline. Raises on
    malformed JSON because that's a contract violation we want to
    notice loudly, not silently drop."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {p}:{lineno}: {e.msg}"
                ) from e
            yield _from_dict(payload, source=f"{p}:{lineno}")


def _from_dict(payload: dict, *, source: str) -> CorrectionExample:
    """Project a parsed JSON dict onto the dataclass, validating the
    required fields. Validates types loosely — the backend Pint
    formatter + the CorrectionExportService's typed casts already
    guarantee shape. This catches "the export schema drifted from
    what the trainer expects" before training silently consumes
    bad rows.
    """
    required = (
        "example_id",
        "subject_anon",
        "section",
        "field",
        "ai_value_mm",
        "final_value_mm",
        "delta_mm",
        "was_corrected",
    )
    for key in required:
        if key not in payload:
            raise ValueError(
                f"Export row at {source} missing required key '{key}'. "
                "Did the Laravel export schema change?"
            )

    return CorrectionExample(
        example_id=str(payload["example_id"]),
        subject_anon=str(payload["subject_anon"]),
        section=str(payload["section"]),
        field=str(payload["field"]),
        ai_value_mm=int(payload["ai_value_mm"]),
        final_value_mm=int(payload["final_value_mm"]),
        delta_mm=int(payload["delta_mm"]),
        was_corrected=bool(payload["was_corrected"]),
        landmarks=payload.get("landmarks"),
        photo_disk=payload.get("photo_disk"),
        photo_public_id=payload.get("photo_public_id"),
        extracted_at=payload.get("extracted_at"),
        confirmed_at=payload.get("confirmed_at"),
    )


def group_by_subject(
    examples: list[CorrectionExample],
) -> dict[str, list[CorrectionExample]]:
    """Bucket examples by `subject_anon` for stratified train/test
    splits — the canonical "don't put the same person's measurements
    in both splits" pattern. Returns a stable dict (Python 3.7+
    insertion-ordered) so iteration order is deterministic across
    runs given the same input file."""
    out: dict[str, list[CorrectionExample]] = {}
    for ex in examples:
        out.setdefault(ex.subject_anon, []).append(ex)
    return out
