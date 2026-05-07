"""Tests for `AvatarStore.delete` and `AvatarStore.delete_older_than`.

Locks in the right-to-be-forgotten + retention paths from audit FS-C3.
Without these tests a future refactor could silently revert to the
"no delete path, retention forever" state the audit flagged.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fitscan.domain.models import AvatarExport, BodyMeasurements
from fitscan.infrastructure.result_store import (
    AvatarRecord,
    AvatarStore,
    Session,
    engine,
)


def _seed_avatar(
    avatar_dir: Path,
    avatar_id: str,
    *,
    created_at: datetime | None = None,
) -> Path:
    """Insert an avatar row + a minimal artifact directory on disk so
    AvatarStore.delete has something to clean up."""
    avatar_dir.mkdir(parents=True, exist_ok=True)
    glb_path = avatar_dir / "avatar.glb"
    glb_path.write_bytes(b"glb")
    thumb_path = avatar_dir / "thumb.png"
    thumb_path.write_bytes(b"png")

    record = AvatarRecord(
        avatar_id=avatar_id,
        gender_profile="ladies",
        height_cm=170.0,
        measurements=json.dumps({"source": "test"}),
        betas=json.dumps([]),
        thetas=json.dumps([]),
        glb_path=str(glb_path),
        thumbnail_path=str(thumb_path),
        created_at=created_at,
    )
    with Session(engine) as session:
        session.merge(record)
        session.commit()
    return avatar_dir


class TestAvatarStoreDelete:
    def test_delete_removes_db_row_and_filesystem_dir(self, tmp_path: Path) -> None:
        avatar_id = "test-delete-1"
        avatar_dir = _seed_avatar(tmp_path / avatar_id, avatar_id)

        store = AvatarStore()
        assert store.get_avatar(avatar_id) is not None
        assert avatar_dir.exists()

        ok = store.delete(avatar_id)
        assert ok is True
        assert store.get_avatar(avatar_id) is None
        assert not avatar_dir.exists()

    def test_delete_is_idempotent_on_missing_avatar(self) -> None:
        store = AvatarStore()
        assert store.delete("does-not-exist") is False

    def test_delete_only_removes_directory_when_avatar_id_in_path(
        self, tmp_path: Path
    ) -> None:
        """If a bug ever stored a row with a glb_path pointing at a shared
        parent (e.g. just `data/avatars/`), `delete` must NOT recursively
        wipe the shared directory. The avatar-id-in-parts guard enforces
        this."""
        avatar_id = "test-shared-parent"
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()
        sentinel = shared_dir / "sibling.txt"
        sentinel.write_text("must survive")

        # Store glb_path AT the shared dir (not inside an avatar-id subdir)
        record = AvatarRecord(
            avatar_id=avatar_id,
            gender_profile="ladies",
            measurements=json.dumps({}),
            betas=json.dumps([]),
            thetas=json.dumps([]),
            glb_path=str(shared_dir / "rogue.glb"),
        )
        (shared_dir / "rogue.glb").write_bytes(b"rogue")
        with Session(engine) as session:
            session.merge(record)
            session.commit()

        AvatarStore().delete(avatar_id)
        # Sibling file (and the shared dir) survived the delete.
        assert sentinel.exists()
        assert sentinel.read_text() == "must survive"


class TestAvatarStoreRetention:
    def test_delete_older_than_removes_only_stale_rows(
        self, tmp_path: Path
    ) -> None:
        now = datetime.now(timezone.utc)
        old_id = "stale-avatar"
        fresh_id = "fresh-avatar"

        _seed_avatar(
            tmp_path / old_id,
            old_id,
            created_at=now - timedelta(days=45),
        )
        _seed_avatar(
            tmp_path / fresh_id,
            fresh_id,
            created_at=now - timedelta(days=5),
        )

        store = AvatarStore()
        removed = store.delete_older_than(days=30)

        assert removed >= 1
        assert store.get_avatar(old_id) is None
        assert store.get_avatar(fresh_id) is not None

        # Cleanup
        store.delete(fresh_id)

    def test_delete_older_than_is_zero_when_nothing_stale(
        self, tmp_path: Path
    ) -> None:
        avatar_id = "fresh-only"
        _seed_avatar(
            tmp_path / avatar_id,
            avatar_id,
            created_at=datetime.now(timezone.utc),
        )

        store = AvatarStore()
        # Use a huge cutoff so nothing qualifies as "stale".
        removed = store.delete_older_than(days=10_000)
        assert removed == 0
        assert store.get_avatar(avatar_id) is not None

        # Cleanup
        store.delete(avatar_id)
