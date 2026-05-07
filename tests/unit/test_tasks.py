"""Tests for Celery scheduled tasks (audit FS-C3, FS-C5).

Without these the daily retention sweep + monthly quota reset can be
silently broken by a refactor that renames `delete_older_than` or moves
the task module — the same way `fitscan.tasks` was missing entirely
when the audit ran.
"""

from __future__ import annotations

from unittest.mock import patch

from celery.schedules import crontab

from fitscan.tasks import app, cleanup_old_avatars, reset_api_key_quotas


class TestCleanupOldAvatarsTask:
    def test_calls_avatar_store_delete_older_than(self):
        with patch(
            "fitscan.infrastructure.result_store.AvatarStore.delete_older_than",
            return_value=3,
        ) as mock_delete:
            removed = cleanup_old_avatars(days=30)
        mock_delete.assert_called_once_with(30)
        assert removed == 3

    def test_default_retention_window_is_30_days(self):
        with patch(
            "fitscan.infrastructure.result_store.AvatarStore.delete_older_than",
            return_value=0,
        ) as mock_delete:
            cleanup_old_avatars()
        mock_delete.assert_called_once_with(30)


class TestResetApiKeyQuotasTask:
    def test_zeroes_only_keys_with_nonzero_usage(self):
        # Insert two keys: one with usage, one without.
        from fitscan.infrastructure.result_store import (
            APIKeyRecord,
            Session,
            engine,
        )

        with Session(engine) as session:
            session.add(
                APIKeyRecord(
                    key_hash="reset-test-1",
                    owner_name="Used",
                    owner_email="used@test.dev",
                    usage_this_month=42,
                )
            )
            session.add(
                APIKeyRecord(
                    key_hash="reset-test-2",
                    owner_name="Unused",
                    owner_email="unused@test.dev",
                    usage_this_month=0,
                )
            )
            session.commit()

        try:
            updated = reset_api_key_quotas()
            assert updated >= 1

            with Session(engine) as session:
                used = (
                    session.query(APIKeyRecord)
                    .filter_by(key_hash="reset-test-1")
                    .first()
                )
                assert used.usage_this_month == 0
        finally:
            with Session(engine) as session:
                session.query(APIKeyRecord).filter(
                    APIKeyRecord.key_hash.in_(["reset-test-1", "reset-test-2"])
                ).delete(synchronize_session=False)
                session.commit()


class TestBeatScheduleWiring:
    """Lock the schedule entries so a future config edit can't silently
    drop the retention sweep or the quota reset."""

    def test_cleanup_avatars_runs_daily_at_04_utc(self):
        entry = app.conf.beat_schedule["cleanup-old-avatars-daily"]
        assert entry["task"] == "fitscan.tasks.cleanup_old_avatars"
        sched = entry["schedule"]
        assert isinstance(sched, crontab)
        # crontab fields normalise to sets — 4 means hour=4, 0 means minute=0
        assert 4 in sched.hour
        assert 0 in sched.minute

    def test_quota_reset_runs_first_of_month(self):
        entry = app.conf.beat_schedule["reset-api-key-quotas-monthly"]
        assert entry["task"] == "fitscan.tasks.reset_api_key_quotas"
        sched = entry["schedule"]
        assert isinstance(sched, crontab)
        assert 1 in sched.day_of_month
