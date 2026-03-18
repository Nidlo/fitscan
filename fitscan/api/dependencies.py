"""FastAPI dependency injection."""

from __future__ import annotations

from functools import lru_cache

from fitscan.application.measurement_service import (
    MeasurementService,
    create_pipeline,
)


@lru_cache(maxsize=1)
def _get_service() -> MeasurementService:
    return MeasurementService(pipeline=create_pipeline())


def get_measurement_service() -> MeasurementService:
    return _get_service()
