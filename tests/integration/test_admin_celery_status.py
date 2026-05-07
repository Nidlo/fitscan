"""Integration tests for the admin-gated Celery status endpoint.

FS-NIDLO-OPS-01 / QA-AD-OPS-010. The endpoint is the Nidlo admin
Operations panel's window into the Celery + broker health — pinned here
so we can't accidentally regress the auth gate or change the response
shape without flipping the QA row red.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from fitscan.api.main import app

ADMIN_KEY = "test-admin-key-do-not-use-in-prod"


@pytest.fixture(autouse=True)
def _set_admin_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FITSCAN_ADMIN_API_KEY", ADMIN_KEY)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestAuth:
    def test_503_when_admin_key_not_configured(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An operator who forgets to set the env shouldn't get a silently
        # public endpoint — we'd rather degrade noisily.
        monkeypatch.setenv("FITSCAN_ADMIN_API_KEY", "")
        response = client.get("/api/v1/admin/celery/status")
        assert response.status_code == 503

    def test_401_when_header_missing(self, client: TestClient) -> None:
        response = client.get("/api/v1/admin/celery/status")
        assert response.status_code == 401

    def test_401_when_header_wrong(self, client: TestClient) -> None:
        response = client.get(
            "/api/v1/admin/celery/status",
            headers={"X-Admin-API-Key": "not-the-right-key"},
        )
        assert response.status_code == 401


class TestSuccessShape:
    def test_returns_expected_shape_when_celery_unreachable(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No broker / no workers in the test env — the endpoint must
        # degrade gracefully (sentinel values) rather than 500.
        monkeypatch.setenv("CELERY_BROKER_URL", "redis://127.0.0.1:1/0")

        response = client.get(
            "/api/v1/admin/celery/status",
            headers={"X-Admin-API-Key": ADMIN_KEY},
        )
        assert response.status_code == 200
        body = response.json()

        # Top-level keys must be stable — the Nidlo admin panel reads them
        # directly to render the status card.
        assert set(body.keys()) >= {
            "workers",
            "queue",
            "active_tasks",
            "scheduled_tasks",
            "broker_reachable",
            "checked_at",
        }
        assert isinstance(body["workers"]["count"], int)
        assert isinstance(body["workers"]["names"], list)
        assert body["queue"]["name"] == "celery"
        # checked_at is an ISO-8601 timestamp.
        assert "T" in body["checked_at"]

    def test_returns_zero_workers_when_inspect_returns_none(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # When no workers respond to ping(), inspect() returns None per
        # task. We assert the endpoint reports zero gracefully.
        from fitscan.api import admin_routes

        class _DeadInspect:
            def ping(self) -> None:
                return None

            def active(self) -> None:
                return None

            def scheduled(self) -> None:
                return None

        class _DeadControl:
            def inspect(self, timeout: float = 0):  # noqa: ARG002
                return _DeadInspect()

        monkeypatch.setattr(
            admin_routes,
            "_gather_celery_status",
            admin_routes._gather_celery_status,  # exercise the real function
        )

        # Patch the celery app's control proxy to return our dead inspect.
        from fitscan.tasks import app as celery_app

        monkeypatch.setattr(celery_app, "control", _DeadControl())

        response = client.get(
            "/api/v1/admin/celery/status",
            headers={"X-Admin-API-Key": ADMIN_KEY},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["workers"]["count"] == 0
        assert body["workers"]["names"] == []
        assert body["active_tasks"] == 0
        assert body["scheduled_tasks"] == 0
