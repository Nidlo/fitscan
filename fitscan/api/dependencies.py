"""FastAPI dependency injection."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
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

    # Local import to avoid a circular import at module load:
    # routes.py -> dependencies.py -> result_store.py
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

        record.usage_this_month += 1  # type: ignore[assignment]
        record.last_used_at = datetime.now(timezone.utc)  # type: ignore[assignment]
        session.commit()


def verify_admin_key(
    x_admin_api_key: Annotated[str | None, Header(alias="X-Admin-API-Key")] = None,
) -> None:
    """Gate admin-only endpoints with `FITSCAN_ADMIN_API_KEY`.

    Used by both the operational endpoints under
    `/api/v1/admin/*` (Celery health, usage stats) AND by
    `/api/v1/keys/register` per the FS-NIDLO-AUTH-03 lockdown
    decision (admin-only minting; Snad issues keys via curl until
    a Filament UI exists).

    `hmac.compare_digest` for constant-time equality so a network
    attacker can't time-leak the configured key (A-04). The
    empty-string fall-through is deliberate: if the operator
    forgot to set the env, EVERY request must fail rather than
    the endpoint silently becoming public.
    """
    expected = os.getenv("FITSCAN_ADMIN_API_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FITSCAN_ADMIN_API_KEY is not configured.",
        )
    if not x_admin_api_key or not hmac.compare_digest(x_admin_api_key, expected):
        # Constant-time fallback to keep the failure path
        # indistinguishable from the success-then-mismatch path.
        secrets.compare_digest("", expected)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid admin API key.",
        )
