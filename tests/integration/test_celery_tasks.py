"""Celery task integration tests with frozen clock (FS-NIDLO-TEST-03).

Pins the contract on the two scheduled tasks the platform depends on:
- `cleanup_old_avatars` enforces the 30-day retention policy that
  audit FS-C3 closed.
- `reset_api_key_quotas` zeroes `usage_this_month` so monthly rate
  limits don't silently lock keys forever (audit FS-C5).

Both tasks are wired to Celery beat ([fitscan/tasks.py:79](.)) but
without these tests no run-level safety net catches a regression
that breaks the side-effect.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from freezegun import freeze_time
from sqlalchemy.orm import Session

from fitscan.infrastructure.result_store import (
    APIKeyRecord,
    AvatarRecord,
    engine,
)
from fitscan.tasks import cleanup_old_avatars, reset_api_key_quotas


@pytest.fixture(autouse=True)
def _wipe_tables():
    """Each test starts from a clean avatars + api_keys slate so
    fixtures don't pollute across runs."""
    with Session(engine) as session:
        session.query(AvatarRecord).delete()
        session.query(APIKeyRecord).delete()
        session.commit()
    yield
    with Session(engine) as session:
        session.query(AvatarRecord).delete()
        session.query(APIKeyRecord).delete()
        session.commit()


def _seed_avatar(
    session: Session, avatar_id: str, created_at: datetime, glb_dir: Path
) -> None:
    glb_path = glb_dir / f"{avatar_id}.glb"
    glb_path.write_bytes(b"glTF\x02fake")
    record = AvatarRecord(
        avatar_id=avatar_id,
        gender_profile="ladies",
        height_cm=170.0,
        measurements=json.dumps({}),
        betas=json.dumps([]),
        thetas=json.dumps([]),
        glb_path=str(glb_path.resolve()),
        obj_path=None,
        thumbnail_path=None,
        created_at=created_at,
    )
    session.add(record)


class TestCleanupOldAvatars:
    def test_removes_only_avatars_past_the_cutoff(self, tmp_path: Path):
        # Two avatars: one created 40 days ago (past 30-day default),
        # one created 5 days ago (still inside window).
        now = datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)
        with freeze_time(now):
            with Session(engine) as session:
                _seed_avatar(
                    session, "old-1", now - timedelta(days=40), tmp_path
                )
                _seed_avatar(
                    session, "fresh-1", now - timedelta(days=5), tmp_path
                )
                session.commit()

            removed = cleanup_old_avatars.run(days=30)

            assert removed == 1
            with Session(engine) as session:
                ids = {r.avatar_id for r in session.query(AvatarRecord).all()}
                assert ids == {"fresh-1"}

    def test_removes_zero_when_all_avatars_are_within_window(
        self, tmp_path: Path
    ):
        now = datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)
        with freeze_time(now):
            with Session(engine) as session:
                _seed_avatar(
                    session, "fresh-a", now - timedelta(days=1), tmp_path
                )
                _seed_avatar(
                    session, "fresh-b", now - timedelta(days=10), tmp_path
                )
                session.commit()

            removed = cleanup_old_avatars.run(days=30)

            assert removed == 0
            with Session(engine) as session:
                assert session.query(AvatarRecord).count() == 2

    def test_honours_custom_days_arg(self, tmp_path: Path):
        # 7-day retention — anything older than 7 days goes.
        now = datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)
        with freeze_time(now):
            with Session(engine) as session:
                _seed_avatar(
                    session, "8d-old", now - timedelta(days=8), tmp_path
                )
                _seed_avatar(
                    session, "3d-old", now - timedelta(days=3), tmp_path
                )
                session.commit()

            removed = cleanup_old_avatars.run(days=7)

            assert removed == 1


class TestResetApiKeyQuotas:
    def _seed_key(
        self,
        session: Session,
        key_hash: str,
        usage: int,
        is_active: bool = True,
    ) -> None:
        session.add(
            APIKeyRecord(
                key_hash=key_hash,
                owner_name="Test",
                owner_email="t@example.com",
                tier="free",
                monthly_limit=50,
                usage_this_month=usage,
                is_active=is_active,
            )
        )

    def test_zeroes_every_keys_usage_counter(self):
        with Session(engine) as session:
            self._seed_key(session, "hash-a", usage=37)
            self._seed_key(session, "hash-b", usage=12)
            self._seed_key(session, "hash-c", usage=0)  # already zero
            session.commit()

        # Only the rows with usage > 0 are touched (the WHERE clause
        # in the task), so the count should be 2.
        updated = reset_api_key_quotas.run()
        assert updated == 2

        with Session(engine) as session:
            counts = {
                r.key_hash: r.usage_this_month
                for r in session.query(APIKeyRecord).all()
            }
            assert counts == {"hash-a": 0, "hash-b": 0, "hash-c": 0}

    def test_resets_inactive_keys_too(self):
        # An inactive key that previously hit its cap should still get
        # its counter zeroed — re-activation shouldn't carry stale usage.
        with Session(engine) as session:
            self._seed_key(session, "hash-locked", usage=50, is_active=False)
            session.commit()

        updated = reset_api_key_quotas.run()
        assert updated == 1

        with Session(engine) as session:
            row = (
                session.query(APIKeyRecord)
                .filter(APIKeyRecord.key_hash == "hash-locked")
                .one()
            )
            assert row.usage_this_month == 0
            # is_active stays as set by the operator — the reset task
            # doesn't re-enable keys, only zeroes usage.
            assert row.is_active is False

    def test_returns_zero_when_no_keys_have_usage(self):
        with Session(engine) as session:
            self._seed_key(session, "hash-fresh", usage=0)
            session.commit()

        updated = reset_api_key_quotas.run()
        assert updated == 0
