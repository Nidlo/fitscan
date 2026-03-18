"""
Accuracy benchmark against ground truth measurements.
Run: pytest tests/benchmark/ -v --tb=short
"""
import json
from pathlib import Path

import pytest

GROUND_TRUTH_PATH = Path("data/benchmarks/synthetic/ground_truth.json")


@pytest.mark.skipif(
    not GROUND_TRUTH_PATH.exists(),
    reason="Benchmark dataset not generated yet.",
)
class TestAccuracyBenchmark:
    def test_ground_truth_file_format(self):
        cases = json.loads(GROUND_TRUTH_PATH.read_text())
        assert len(cases) > 0
        assert "height_cm" in cases[0]
        assert "chest_cm" in cases[0]

    def test_height_values_plausible(self):
        cases = json.loads(GROUND_TRUTH_PATH.read_text())
        for case in cases:
            assert 100 <= case["height_cm"] <= 220
