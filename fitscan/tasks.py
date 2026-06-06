"""Celery application + scheduled tasks.

Referenced by `make worker` (`celery -A fitscan.tasks worker`). Without
this module that target failed with ModuleNotFoundError (audit FS-H2).

Today the only periodic job is avatar retention; the broker/result-backend
are read from CELERY_BROKER_URL / CELERY_RESULT_BACKEND so dev and prod
can swap implementations (Redis in dev, anything-AMQP in prod) without
touching code.
"""

from __future__ import annotations

import os
import urllib.request

from celery import Celery
from celery.schedules import crontab
from loguru import logger

app = Celery(
    "fitscan",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)

app.conf.update(
    # Reliability defaults - covers the audit FS-H3 retry-policy gap for
    # any future tasks that don't override.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=15 * 60,
    task_soft_time_limit=10 * 60,
    worker_max_tasks_per_child=200,
)


@app.task(bind=True, name="fitscan.tasks.cleanup_old_avatars", ignore_result=True)
def cleanup_old_avatars(self, days: int = 30) -> int:
    """Delete avatars older than `days`. Drives the right-to-be-forgotten /
    retention policy from audit FS-C3. Idempotent: re-running deletes only
    rows still past the cutoff.
    """
    # Local import to keep this module loadable without the full ML stack -
    # `celery -A fitscan.tasks` shouldn't have to import torch.
    from fitscan.infrastructure.result_store import AvatarStore

    removed = AvatarStore().delete_older_than(days)
    logger.info(f"cleanup_old_avatars: removed={removed} cutoff_days={days}")
    return removed


@app.task(bind=True, name="fitscan.tasks.reset_api_key_quotas", ignore_result=True)
def reset_api_key_quotas(self) -> int:
    """Reset every API key's `usage_this_month` counter to 0.

    The auth dependency (audit FS-C1) increments `usage_this_month` and
    rejects callers past `monthly_limit`, but without this monthly reset
    every key would silently lock once it crossed its quota - even on
    the 1st of the next month. Audit FS-C5.
    """
    from sqlalchemy.orm import Session

    from fitscan.infrastructure.result_store import APIKeyRecord, engine

    with Session(engine) as session:
        updated = (
            session.query(APIKeyRecord)
            .filter(APIKeyRecord.usage_this_month > 0)
            .update({APIKeyRecord.usage_this_month: 0}, synchronize_session=False)
        )
        session.commit()
    logger.info(f"reset_api_key_quotas: zeroed counters for {updated} keys")
    return updated


@app.task(bind=True, name="fitscan.tasks.beat_heartbeat", ignore_result=True)
def beat_heartbeat(self) -> None:
    """Ping Healthchecks.io to confirm Celery beat is alive.

    Skips silently when HEALTHCHECKS_BEAT_URL is not set (local/CI).
    """
    url = os.getenv("HEALTHCHECKS_BEAT_URL", "")
    if not url:
        return
    try:
        urllib.request.urlopen(url, timeout=5)  # noqa: S310
    except Exception:
        pass


# Daily 04:00 UTC retention sweep + 1st-of-month 00:05 UTC quota reset.
# Beat must be running for these to fire:
#   celery -A fitscan.tasks beat --loglevel=info
app.conf.beat_schedule = {
    "cleanup-old-avatars-daily": {
        "task": "fitscan.tasks.cleanup_old_avatars",
        "schedule": crontab(hour=4, minute=0),
        "args": (int(os.getenv("FITSCAN_RETENTION_DAYS", "30")),),
    },
    "reset-api-key-quotas-monthly": {
        "task": "fitscan.tasks.reset_api_key_quotas",
        "schedule": crontab(day_of_month=1, hour=0, minute=5),
    },
    "beat-heartbeat": {
        "task": "fitscan.tasks.beat_heartbeat",
        "schedule": crontab(minute="*/5"),
    },
}
