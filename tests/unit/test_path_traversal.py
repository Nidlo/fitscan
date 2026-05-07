"""Path-traversal tests for the avatar FileResponse routes (audit FS-C9).

The DB-stored `glb_path` / `thumbnail_path` columns are absolute paths.
Without sandboxing, a row carrying `glb_path = "/etc/passwd"` (introduced
by a future SQLi or admin-panel bug) would have FastAPI happily stream
that file to the caller. `_resolve_inside_avatars_root` must reject any
path that doesn't sit inside the configured avatars root.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from fitscan.api.routes import _AVATARS_ROOT, _resolve_inside_avatars_root


class TestResolveInsideAvatarsRoot:
    def test_rejects_absolute_path_outside_root(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _resolve_inside_avatars_root("/etc/passwd")
        assert exc_info.value.status_code == 404

    def test_rejects_relative_traversal_escape(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _resolve_inside_avatars_root("data/avatars/../../etc/passwd")
        assert exc_info.value.status_code == 404

    def test_rejects_none_or_empty_path(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _resolve_inside_avatars_root(None)
        assert exc_info.value.status_code == 404

        with pytest.raises(HTTPException) as exc_info:
            _resolve_inside_avatars_root("")
        assert exc_info.value.status_code == 404

    def test_rejects_path_to_nonexistent_file_inside_root(
        self, tmp_path: Path
    ) -> None:
        """Even when the path resolves inside the root, missing files
        return 404 — never 200 with empty bytes."""
        ghost = _AVATARS_ROOT / "ghost-id" / "avatar.glb"
        with pytest.raises(HTTPException) as exc_info:
            _resolve_inside_avatars_root(str(ghost))
        assert exc_info.value.status_code == 404

    def test_accepts_path_inside_root(self, tmp_path: Path, monkeypatch) -> None:
        """A path that resolves inside `_AVATARS_ROOT` AND exists on disk
        returns successfully — proving the guard isn't a blanket reject."""
        from fitscan.api import routes as routes_module

        # Redirect the root to a temporary directory and seed a real file.
        new_root = (tmp_path / "avatars").resolve()
        new_root.mkdir(parents=True, exist_ok=True)
        target = new_root / "abc-123" / "avatar.glb"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"glb")

        monkeypatch.setattr(routes_module, "_AVATARS_ROOT", new_root)
        resolved = routes_module._resolve_inside_avatars_root(str(target))
        assert resolved == target.resolve()

    def test_does_not_leak_existence_through_status_code(self) -> None:
        """Path-outside-root and path-not-found must both return 404 so a
        caller can't probe whether a particular path *exists* on the box
        by comparing 403 vs 404."""
        with pytest.raises(HTTPException) as outside:
            _resolve_inside_avatars_root("/etc/passwd")
        with pytest.raises(HTTPException) as inside_missing:
            _resolve_inside_avatars_root(str(_AVATARS_ROOT / "missing.glb"))
        assert outside.value.status_code == inside_missing.value.status_code == 404
