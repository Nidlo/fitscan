"""FastAPI dependency injection."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timezone
from functools import lru_cache
from typing import Annotated

from fastapi import Header, HTTPException, Request, status
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


def _is_localhost(host: str | None) -> bool:
    """True when the request originated from the same machine. Used by
    the FITSCAN_ALLOW_UNAUTHED_LOCALHOST dev bypass below."""
    if host is None:
        return False
    return host in ("127.0.0.1", "::1", "localhost")


def _check_admin_key(
    x_admin_api_key: str | None,
    *,
    client_host: str | None = None,
) -> None:
    """Pure check: raises HTTPException(401/503) when the key is bad
    or missing. Handles the FITSCAN_ALLOW_UNAUTHED_LOCALHOST bypass
    when `client_host` is supplied. Used by both the FastAPI
    dependency below and the route handlers' inline calls.

    Reads via pydantic `Settings` (which loads `.env` directly via
    `model_config.env_file`). We deliberately don't push `.env`
    into `os.environ` because the repo's `DATABASE_URL` placeholder
    breaks dev hosts that haven't provisioned a `fitscan` Postgres
    role yet — Settings keeps each consumer isolated from the
    others. We fall back to `os.getenv` so existing tests that
    use `monkeypatch.setenv` keep working.
    """
    from fitscan.config import get_settings  # local import — avoids cold-start cost

    settings = get_settings()

    # `monkeypatch.setenv` in tests + pydantic's `.env` reader both
    # populate `Settings`. But `os.getenv` is the only source visible
    # when env is set AFTER `Settings()` was first constructed
    # (`@lru_cache` keeps the first instance). Check os.environ first
    # so tests that monkeypatch don't have to bust the cache.
    bypass_enabled = (
        os.getenv("FITSCAN_ALLOW_UNAUTHED_LOCALHOST", "").lower()
        in ("true", "1", "yes", "on")
        or settings.fitscan_allow_unauthed_localhost
    )
    if client_host is not None and bypass_enabled and _is_localhost(client_host):
        return

    expected = os.getenv("FITSCAN_ADMIN_API_KEY", "") or settings.fitscan_admin_api_key
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


def verify_admin_key(
    request: Request,
    x_admin_api_key: Annotated[str | None, Header(alias="X-Admin-API-Key")] = None,
) -> None:
    """FastAPI dependency wrapping `_check_admin_key`.

    Used by `/api/v1/keys/register` (and any future route that wires
    `Depends(verify_admin_key)`) per the FS-NIDLO-AUTH-03 lockdown
    decision (admin-only minting).

    `hmac.compare_digest` for constant-time equality so a network
    attacker can't time-leak the configured key (A-04). The
    empty-string fall-through is deliberate: if the operator
    forgot to set the env, EVERY request must fail rather than
    the endpoint silently becoming public.

    Dev bypass: `FITSCAN_ALLOW_UNAUTHED_LOCALHOST=true` makes the
    gate trust loopback callers without the key. Useful for local
    Filament dashboards hitting `http://localhost:8008` while the
    operator is wiring up the shared-secret cross-repo flow.
    Defaults OFF — production deployments must keep the key gate.
    """
    client_host = request.client.host if request.client else None
    _check_admin_key(x_admin_api_key, client_host=client_host)
