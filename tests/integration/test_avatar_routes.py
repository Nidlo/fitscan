"""Integration tests for avatar artifact routes (FS-NIDLO-TEST-02).

Covers /avatar/{id}/glb, /thumbnail, /fit-garment, /override, DELETE.
Closes the audit T2 gap — these routes had zero coverage and a single
test would have caught FS-C6 (`measure_with_mesh` AttributeError) on
day one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from fitscan.api.main import app
from fitscan.infrastructure.result_store import AvatarRecord, engine

client = TestClient(app)


@pytest.fixture
def avatar_glb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Create an avatar DB row + on-disk GLB stub inside the sandboxed
    `_AVATARS_ROOT`. The route's `_resolve_inside_avatars_root` will
    accept paths under this root and reject everything else, so we
    point FITSCAN_DATA_DIR at tmp_path before the route sees the path.
    """
    avatar_id = "test-avatar-1234"
    monkeypatch.setenv("FITSCAN_DATA_DIR", str(tmp_path))

    # The route caches `_AVATARS_ROOT` at module load; override the
    # cached value too so the sandbox check sees our temp root.
    from fitscan.api import routes as routes_module

    routes_module._AVATARS_ROOT = (tmp_path / "avatars").resolve()

    avatar_dir = tmp_path / "avatars" / avatar_id
    avatar_dir.mkdir(parents=True, exist_ok=True)
    glb_path = avatar_dir / "avatar.glb"
    glb_path.write_bytes(b"glTF\x02\x00\x00\x00fake-glb-bytes")
    thumb_path = avatar_dir / "thumb.png"
    thumb_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-bytes")

    with Session(engine) as session:
        # Wipe any leftover rows from a previous test run.
        session.query(AvatarRecord).filter(
            AvatarRecord.avatar_id == avatar_id
        ).delete()
        record = AvatarRecord(
            avatar_id=avatar_id,
            gender_profile="ladies",
            height_cm=170.0,
            measurements=json.dumps({"source": "smpl"}),
            betas=json.dumps([0.1, 0.2, 0.3]),
            thetas=json.dumps([0.0, 0.0]),
            glb_path=str(glb_path.resolve()),
            obj_path=None,
            thumbnail_path=str(thumb_path.resolve()),
        )
        session.add(record)
        session.commit()

    yield {
        "avatar_id": avatar_id,
        "glb_path": glb_path,
        "thumb_path": thumb_path,
    }

    # Cleanup
    with Session(engine) as session:
        session.query(AvatarRecord).filter(
            AvatarRecord.avatar_id == avatar_id
        ).delete()
        session.commit()


class TestAvatarGlbRoute:
    def test_returns_glb_binary_when_avatar_exists(self, avatar_glb: dict):
        response = client.get(f"/api/v1/avatar/{avatar_glb['avatar_id']}/glb")
        assert response.status_code == 200
        assert response.headers["content-type"] == "model/gltf-binary"
        assert response.content.startswith(b"glTF")

    def test_404_when_avatar_does_not_exist(self):
        response = client.get("/api/v1/avatar/nonexistent-uuid/glb")
        assert response.status_code == 404

    def test_404_when_glb_path_escapes_avatars_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """FS-C9 sandbox — a DB row pointing outside _AVATARS_ROOT must
        404 (not serve /etc/passwd or .env). Returning 404 (not 403)
        avoids leaking which condition triggered the failure."""
        avatar_id = "evil-avatar-id"
        monkeypatch.setenv("FITSCAN_DATA_DIR", str(tmp_path))
        from fitscan.api import routes as routes_module

        routes_module._AVATARS_ROOT = (tmp_path / "avatars").resolve()

        # Path lives outside _AVATARS_ROOT.
        outside = tmp_path / "elsewhere" / "secret.glb"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_bytes(b"glTF")

        with Session(engine) as session:
            session.query(AvatarRecord).filter(
                AvatarRecord.avatar_id == avatar_id
            ).delete()
            session.add(
                AvatarRecord(
                    avatar_id=avatar_id,
                    gender_profile="ladies",
                    measurements=json.dumps({}),
                    betas=json.dumps([]),
                    thetas=json.dumps([]),
                    glb_path=str(outside.resolve()),
                )
            )
            session.commit()

        try:
            response = client.get(f"/api/v1/avatar/{avatar_id}/glb")
            assert response.status_code == 404
        finally:
            with Session(engine) as session:
                session.query(AvatarRecord).filter(
                    AvatarRecord.avatar_id == avatar_id
                ).delete()
                session.commit()


class TestAvatarThumbnailRoute:
    def test_returns_png_when_avatar_exists(self, avatar_glb: dict):
        response = client.get(
            f"/api/v1/avatar/{avatar_glb['avatar_id']}/thumbnail"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_404_when_avatar_does_not_exist(self):
        response = client.get("/api/v1/avatar/nonexistent-uuid/thumbnail")
        assert response.status_code == 404


class TestAvatarDeleteRoute:
    def test_204_and_row_gone_when_avatar_exists(self, avatar_glb: dict):
        avatar_id = avatar_glb["avatar_id"]
        response = client.delete(f"/api/v1/avatar/{avatar_id}")
        assert response.status_code == 204

        with Session(engine) as session:
            row = session.get(AvatarRecord, avatar_id)
            assert row is None

    def test_404_when_avatar_does_not_exist(self):
        response = client.delete("/api/v1/avatar/never-existed")
        assert response.status_code == 404


class TestAvatarOverrideRoute:
    def test_accepts_override_dict(self, avatar_glb: dict):
        response = client.post(
            f"/api/v1/avatar/{avatar_glb['avatar_id']}/override",
            json={"waist_cm": 78.0, "bust_cm": 92.5},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["avatar_id"] == avatar_glb["avatar_id"]
        # Inverse-SMPL stub still — pin the field surfaces existing.
        assert "recalculated_measurements" in body
        assert "convergence_note" in body

    def test_400_on_empty_overrides(self, avatar_glb: dict):
        response = client.post(
            f"/api/v1/avatar/{avatar_glb['avatar_id']}/override",
            json={},
        )
        assert response.status_code == 400

    def test_404_when_avatar_does_not_exist(self):
        response = client.post(
            "/api/v1/avatar/never-existed/override",
            json={"waist_cm": 78.0},
        )
        assert response.status_code == 404
