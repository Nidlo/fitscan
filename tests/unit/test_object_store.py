"""Tests for object store."""
import os
from pathlib import Path

import pytest


class TestObjectStoreLocal:
    def test_put_and_get_local(self, tmp_path):
        os.environ["STORAGE_BACKEND"] = "local"
        from fitscan.infrastructure.object_store import ObjectStore

        store = ObjectStore()
        store._settings.storage_backend = "local"
        key = "test/test_file.bin"
        data = b"hello world"
        url = store._put_local(key, data)
        assert "/local/" in url
        retrieved = store._get_local(key)
        assert retrieved == data
        # Cleanup
        (Path("data/avatars") / key).unlink(missing_ok=True)
