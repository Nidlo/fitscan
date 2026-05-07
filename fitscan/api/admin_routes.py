"""Admin-only operational endpoints.

These are gated by `FITSCAN_ADMIN_API_KEY` (shared with the Nidlo backend
admin panel — separate from the per-tenant `X-API-Key` we issue to
measurement clients) so a leaked customer key can't surface
infrastructure-shaped data like worker counts and queue depth.
"""

from __future__ import annotations

import hmac
import os
import secrets
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, status

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


def _verify_admin_key(
    x_admin_api_key: Annotated[
        str | None, Header(alias="X-Admin-API-Key")
    ] = None,
) -> None:
    """Compare-then-throw with constant-time equality so a network
    attacker can't time-leak the configured key. The empty-string
    fall-through is deliberate: if the operator forgot to set the env,
    EVERY request must fail rather than the endpoint silently becoming
    public.
    """
    expected = os.getenv("FITSCAN_ADMIN_API_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FITSCAN_ADMIN_API_KEY is not configured.",
        )
    if not x_admin_api_key or not hmac.compare_digest(
        x_admin_api_key, expected
    ):
        # Constant-time fallback to keep the failure path indistinguishable
        # from the success-then-mismatch path.
        secrets.compare_digest("", expected)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid admin API key.",
        )


def _gather_celery_status() -> dict[str, Any]:
    """Probe Celery + the broker for live worker / queue stats. Each
    sub-probe is wrapped so a single dead leg (broker unreachable, no
    workers) degrades to a sentinel value instead of failing the whole
    response — the admin panel needs to know what IS broken, not just
    "everything is broken".
    """
    # Local import: tasks.py builds a Celery app at import time and pulls
    # the broker URL from env. Importing at module-level would couple this
    # router to broker availability for routing-table construction, which
    # we don't want.
    from fitscan.tasks import app as celery_app  # type: ignore[import]

    workers: list[str] = []
    active_tasks = 0
    scheduled_tasks = 0
    workers_reachable = True
    try:
        inspect = celery_app.control.inspect(timeout=2.0)
        ping_response = inspect.ping() or {}
        workers = sorted(ping_response.keys())
        active = inspect.active() or {}
        active_tasks = sum(len(v) for v in active.values())
        scheduled = inspect.scheduled() or {}
        scheduled_tasks = sum(len(v) for v in scheduled.values())
    except Exception:  # noqa: BLE001 — broker dead is the whole point of probing
        workers_reachable = False

    queue_depth: int | None = None
    broker_reachable = True
    try:
        from redis import Redis  # type: ignore[import]

        broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
        client = Redis.from_url(broker_url, socket_connect_timeout=1.5)
        queue_depth = int(client.llen("celery"))
    except Exception:  # noqa: BLE001
        broker_reachable = False
        queue_depth = None

    return {
        "workers": {
            "count": len(workers),
            "names": workers,
            "reachable": workers_reachable,
        },
        "queue": {
            "name": "celery",
            "depth": queue_depth,
        },
        "active_tasks": active_tasks,
        "scheduled_tasks": scheduled_tasks,
        "broker_reachable": broker_reachable,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/celery/status", dependencies=[])
def celery_status(
    x_admin_api_key: Annotated[
        str | None, Header(alias="X-Admin-API-Key")
    ] = None,
) -> dict[str, Any]:
    """FS-NIDLO-OPS-01 / QA-AD-OPS-010 — Returns worker + queue + beat
    health for the Nidlo admin's Operations panel. Read-only."""
    _verify_admin_key(x_admin_api_key)
    return _gather_celery_status()
