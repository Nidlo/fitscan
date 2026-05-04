"""FastAPI dependency injection."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Annotated

from fastapi import Header, HTTPException, status
from sqlalchemy.orm import Session

from fitscan.application.measurement_service import (
    MeasurementService,
    create_pipeline,
)


@lru_cache(maxsize=1)
def _get_service() -> MeasurementService:
    return MeasurementService(pipeline=create_pipeline())


def get_measurement_service() -> MeasurementService:
    return _get_service()


def _api_key_required() -> bool:
    """Treat anything other than an explicit falsy value as 'auth on'.

    Set FITSCAN_REQUIRE_API_KEY=false in the test env or for trusted-network
    bypass. Production must leave this unset (defaults to true).
    """
    return os.getenv("FITSCAN_REQUIRE_API_KEY", "true").lower() not in (
        "false",
        "0",
        "no",
        "off",
    )


def verify_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Authenticate the caller via X-API-Key. Raises 401/429 on failure."""
    if not _api_key_required():
        return

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Local import to avoid a circular import at module load: routes.py -> dependencies.py -> result_store.py
    from fitscan.infrastructure.result_store import APIKeyRecord, engine

    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()

    with Session(engine) as session:
        record = (
            session.query(APIKeyRecord)
            .filter_by(key_hash=key_hash, is_active=True)
            .first()
        )
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        if record.usage_this_month >= record.monthly_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Monthly API limit exceeded.",
            )

        record.usage_this_month += 1
        record.last_used_at = datetime.now(timezone.utc)
        session.commit()
