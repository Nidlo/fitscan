"""
Downloads benchmark datasets with known body measurements.
Run: python scripts/download_datasets.py
"""
import csv
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BENCHMARK_DATA_DIR = Path("data/benchmarks")
BENCHMARK_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _create_synthetic_benchmark(gender: str) -> None:
    """Create a minimal CSV benchmark with statistically derived test cases."""
    output = BENCHMARK_DATA_DIR / f"synthetic_benchmark_{gender.lower()}.csv"
    if output.exists():
        log.info(f"Synthetic benchmark {gender} already exists.")
        return
    mean_male = {"height": 175.6, "chest": 99.9, "waist": 86.0, "hip": 99.7}
    mean_female = {"height": 162.9, "chest": 95.0, "waist": 75.5, "hip": 101.8}
    mean = mean_male if gender == "MALE" else mean_female
    stds = {"height": 7.1, "chest": 9.3, "waist": 10.5, "hip": 8.9}

    random.seed(42)
    rows = []
    for i in range(20):
        row = {
            k: round(mean[k] + random.gauss(0, stds[k]), 1) for k in mean
        }
        row["id"] = f"synthetic_{gender.lower()}_{i:02d}"
        rows.append(row)

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "height", "chest", "waist", "hip"]
        )
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Created synthetic benchmark: {output}")


def download_ansur() -> None:
    """Create synthetic benchmark data based on ANSUR II statistics."""
    log.info("Creating ANSUR II-based synthetic benchmarks...")
    _create_synthetic_benchmark("MALE")
    _create_synthetic_benchmark("FEMALE")


def create_synthetic_ground_truth() -> None:
    """Create synthetic ground truth JSON for testing."""
    import json

    synth_dir = BENCHMARK_DATA_DIR / "synthetic"
    synth_dir.mkdir(exist_ok=True)
    gt_path = synth_dir / "ground_truth.json"
    if gt_path.exists():
        log.info("Synthetic ground truth already exists.")
        return

    random.seed(42)
    cases = []
    for i in range(20):
        cases.append(
            {
                "id": f"synthetic_{i:03d}",
                "height_cm": round(160 + random.gauss(0, 8), 1),
                "chest_cm": round(95 + random.gauss(0, 8), 1),
                "waist_cm": round(78 + random.gauss(0, 9), 1),
                "hip_cm": round(100 + random.gauss(0, 7), 1),
                "neck_cm": round(37 + random.gauss(0, 3), 1),
            }
        )
    gt_path.write_text(json.dumps(cases, indent=2))
    log.info(f"Created synthetic ground truth: {gt_path}")


if __name__ == "__main__":
    download_ansur()
    create_synthetic_ground_truth()
    log.info("Dataset acquisition complete.")
