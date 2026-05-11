"""Admin-only operational endpoints.

These are gated by `FITSCAN_ADMIN_API_KEY` (shared with the Nidlo backend
admin panel — separate from the per-tenant `X-API-Key` we issue to
measurement clients) so a leaked customer key can't surface
infrastructure-shaped data like worker counts and queue depth.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, cast

from fastapi import APIRouter, Header

from .dependencies import verify_admin_key

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

# Backwards-compat alias — older imports inside this module still
# reference the underscore-prefixed name; new call-sites import
# `verify_admin_key` from `.dependencies` directly. Keeping the alias
# avoids touching every internal use in this file.
_verify_admin_key = verify_admin_key


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
        queue_depth = cast(int, client.llen("celery"))
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
    x_admin_api_key: Annotated[str | None, Header(alias="X-Admin-API-Key")] = None,
) -> dict[str, Any]:
    """FS-NIDLO-OPS-01 / QA-AD-OPS-010 — Returns worker + queue + beat
    health for the Nidlo admin's Operations panel. Read-only."""
    _verify_admin_key(x_admin_api_key)
    return _gather_celery_status()


def _gather_usage(start: datetime, end: datetime) -> dict[str, Any]:
    """Aggregate measurement_runs over [start, end). All math runs in
    SQL — the table can grow into millions of rows over a year and we
    don't want to ship them all to Python just to take a median."""
    from sqlalchemy import case, func

    # Use a session for read-only aggregation.
    from sqlalchemy.orm import Session

    from fitscan.infrastructure.result_store import (
        MeasurementRunRecord,
        engine,
    )

    with Session(engine) as session:
        base = session.query(MeasurementRunRecord).filter(
            MeasurementRunRecord.started_at >= start,
            MeasurementRunRecord.started_at < end,
        )

        total = base.count()

        if total == 0:
            return {
                "from": start.isoformat(),
                "until": end.isoformat(),
                "total_measurements": 0,
                "successful_measurements": 0,
                "error_rate": 0.0,
                "avg_latency_ms": None,
                "p95_latency_ms": None,
                "by_day": [],
                "by_error_kind": [],
            }

        successful = base.filter(MeasurementRunRecord.success.is_(True)).count()
        error_rate = round((total - successful) / total, 4)

        # Average latency: only over successful runs — failures often fail
        # fast and skew the mean down. The QA spec says "average latency"
        # which operators read as "how long does a real measurement take".
        avg_latency = (
            session.query(func.avg(MeasurementRunRecord.latency_ms))
            .filter(
                MeasurementRunRecord.started_at >= start,
                MeasurementRunRecord.started_at < end,
                MeasurementRunRecord.success.is_(True),
            )
            .scalar()
        )
        avg_latency_ms = int(avg_latency) if avg_latency is not None else None

        # Per-day count + per-day error rate. SQLite + Postgres both
        # accept date(started_at); we trust the testing path uses SQLite
        # and prod uses Postgres which both honour func.date().
        by_day_rows = (
            session.query(
                func.date(MeasurementRunRecord.started_at).label("day"),
                func.count(MeasurementRunRecord.id).label("total"),
                func.sum(
                    case((MeasurementRunRecord.success.is_(True), 1), else_=0)
                ).label("succeeded"),
            )
            .filter(
                MeasurementRunRecord.started_at >= start,
                MeasurementRunRecord.started_at < end,
            )
            .group_by("day")
            .order_by("day")
            .all()
        )
        by_day = [
            {
                "day": str(row.day),
                "total": int(row.total),
                "succeeded": int(row.succeeded or 0),
                "failed": int(row.total) - int(row.succeeded or 0),
            }
            for row in by_day_rows
        ]

        by_error_rows = (
            session.query(
                MeasurementRunRecord.error_kind,
                func.count(MeasurementRunRecord.id).label("total"),
            )
            .filter(
                MeasurementRunRecord.started_at >= start,
                MeasurementRunRecord.started_at < end,
                MeasurementRunRecord.success.is_(False),
            )
            .group_by(MeasurementRunRecord.error_kind)
            .order_by(func.count(MeasurementRunRecord.id).desc())
            .all()
        )
        by_error_kind = [
            {"error_kind": row.error_kind or "unknown", "count": int(row.total)}
            for row in by_error_rows
        ]

        # Approximate p95: ORDER BY + OFFSET. Cheap and correct enough at
        # the row counts we expect; if usage grows past ~10M runs/month,
        # swap to a percentile_cont in Postgres (won't work in SQLite).
        from sqlalchemy import desc

        p95_offset = max(0, int(successful * 0.05) - 1)
        p95_row = (
            session.query(MeasurementRunRecord.latency_ms)
            .filter(
                MeasurementRunRecord.started_at >= start,
                MeasurementRunRecord.started_at < end,
                MeasurementRunRecord.success.is_(True),
            )
            .order_by(desc(MeasurementRunRecord.latency_ms))
            .offset(p95_offset)
            .first()
        )
        p95_latency_ms = int(p95_row.latency_ms) if p95_row is not None else None

        return {
            "from": start.isoformat(),
            "until": end.isoformat(),
            "total_measurements": total,
            "successful_measurements": successful,
            "error_rate": error_rate,
            "avg_latency_ms": avg_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "by_day": by_day,
            "by_error_kind": by_error_kind,
        }


@router.get("/usage")
def usage(
    x_admin_api_key: Annotated[str | None, Header(alias="X-Admin-API-Key")] = None,
    days: int = 30,
) -> dict[str, Any]:
    """FS-NIDLO-OPS-01 / QA-AD-RPT-006 — Per-day measurement counts +
    latency + error breakdown over the trailing `days` window (default
    30). Anthropic spend is intentionally NOT included here — it lives
    in Anthropic's billing console and is linked from the admin
    report's footer."""
    _verify_admin_key(x_admin_api_key)
    days = max(1, min(days, 365))
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return _gather_usage(start, end)
