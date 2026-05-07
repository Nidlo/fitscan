"""Pin the JSONL contract between Laravel's exporter + Python's loader.

If either side drifts, these tests fire — that's exactly the contract
boundary we want to protect. The Laravel side has its own tests at
`backend/tests/Feature/Fitscan/CorrectionExportTest.php`; together
they bracket the file format from both producer and consumer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fitscan.training.data_loader import (
    CorrectionExample,
    group_by_subject,
    iter_jsonl,
    load_jsonl,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def sample_export(tmp_path: Path) -> Path:
    """Two rows that mirror the documented JSONL shape — one
    corrected, one un-corrected. Same `subject_anon` so the
    grouping test has something to bucket."""
    path = tmp_path / "sample.jsonl"
    _write_jsonl(
        path,
        [
            {
                "example_id": "019e0250-aaaa-bbbb-cccc-000000000001",
                "subject_anon": "9f86d081884c7d65",
                "section": "upper_body",
                "field": "bust",
                "ai_value_mm": 920,
                "final_value_mm": 905,
                "delta_mm": -15,
                "was_corrected": True,
                "landmarks": {
                    "left_shoulder": {"x": 0.35, "y": 0.21, "visibility": 0.92}
                },
                "photo_disk": "local",
                "photo_public_id": "measurements/abc.jpg",
                "extracted_at": "2026-05-07T10:00:00Z",
                "confirmed_at": "2026-05-07T10:42:13Z",
            },
            {
                "example_id": "019e0250-aaaa-bbbb-cccc-000000000001",
                "subject_anon": "9f86d081884c7d65",
                "section": "upper_body",
                "field": "waist",
                "ai_value_mm": 720,
                "final_value_mm": 720,
                "delta_mm": 0,
                "was_corrected": False,
                "landmarks": None,
                "photo_disk": "local",
                "photo_public_id": "measurements/abc.jpg",
                "extracted_at": "2026-05-07T10:00:00Z",
                "confirmed_at": "2026-05-07T10:42:13Z",
            },
        ],
    )
    return path


class TestLoadJsonl:
    def test_parses_documented_shape_into_dataclass(
        self, sample_export: Path
    ) -> None:
        rows = load_jsonl(sample_export)
        assert len(rows) == 2

        bust = next(r for r in rows if r.field == "bust")
        assert isinstance(bust, CorrectionExample)
        assert bust.ai_value_mm == 920
        assert bust.final_value_mm == 905
        assert bust.delta_mm == -15
        assert bust.was_corrected is True
        assert bust.landmarks is not None
        assert bust.landmarks["left_shoulder"]["x"] == 0.35

        waist = next(r for r in rows if r.field == "waist")
        assert waist.delta_mm == 0
        assert waist.was_corrected is False
        assert waist.landmarks is None

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        # Trailing newline is normal; mid-file blank lines shouldn't
        # break the loader either.
        path = tmp_path / "blanks.jsonl"
        path.write_text(
            json.dumps(
                {
                    "example_id": "x",
                    "subject_anon": "y",
                    "section": "upper_body",
                    "field": "bust",
                    "ai_value_mm": 900,
                    "final_value_mm": 900,
                    "delta_mm": 0,
                    "was_corrected": False,
                }
            )
            + "\n\n\n"
        )
        assert len(load_jsonl(path)) == 1

    def test_raises_on_malformed_json(self, tmp_path: Path) -> None:
        # Loud failure on contract violation — silent drop would hide
        # an export-format regression.
        path = tmp_path / "bad.jsonl"
        path.write_text("{not valid json\n")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_jsonl(path)

    def test_raises_on_missing_required_field(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.jsonl"
        _write_jsonl(
            path,
            [
                {
                    # `delta_mm` deliberately omitted — older export
                    # format that we don't support.
                    "example_id": "x",
                    "subject_anon": "y",
                    "section": "upper_body",
                    "field": "bust",
                    "ai_value_mm": 900,
                    "final_value_mm": 905,
                    "was_corrected": True,
                }
            ],
        )
        with pytest.raises(ValueError, match="missing required key 'delta_mm'"):
            load_jsonl(path)


class TestIterJsonl:
    def test_streams_lazily(self, sample_export: Path) -> None:
        # iter_jsonl should be a generator we can consume one at a
        # time — important for files larger than RAM.
        gen = iter_jsonl(sample_export)
        first = next(gen)
        assert first.section == "upper_body"
        # The second row is still un-consumed at this point.
        rest = list(gen)
        assert len(rest) == 1


class TestGroupBySubject:
    def test_buckets_examples_by_subject_anon(
        self, sample_export: Path
    ) -> None:
        # Both rows share subject_anon → both buckets land in one
        # group. This is the train/test split anchor.
        rows = load_jsonl(sample_export)
        groups = group_by_subject(rows)
        assert list(groups.keys()) == ["9f86d081884c7d65"]
        assert len(groups["9f86d081884c7d65"]) == 2

    def test_separate_subjects_into_separate_buckets(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "two-subjects.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "example_id": "1",
                    "subject_anon": "aaaa",
                    "section": "upper_body",
                    "field": "bust",
                    "ai_value_mm": 900,
                    "final_value_mm": 900,
                    "delta_mm": 0,
                    "was_corrected": False,
                },
                {
                    "example_id": "2",
                    "subject_anon": "bbbb",
                    "section": "upper_body",
                    "field": "bust",
                    "ai_value_mm": 850,
                    "final_value_mm": 860,
                    "delta_mm": 10,
                    "was_corrected": True,
                },
            ],
        )
        rows = load_jsonl(path)
        groups = group_by_subject(rows)
        assert set(groups.keys()) == {"aaaa", "bbbb"}
        assert len(groups["aaaa"]) == 1
        assert len(groups["bbbb"]) == 1
