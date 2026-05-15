"""Integration tests for the admin-gated Celery status endpoint.

FS-NIDLO-OPS-01 / QA-AD-OPS-010. The endpoint is the Nidlo admin
Operations panel's window into the Celery + broker health — pinned here
so we can't accidentally regress the auth gate or change the response
shape without flipping the QA row red.
"""

from __future__ import annotations

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

    def test_localhost_bypass_when_flag_enabled(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Dev-only path: FITSCAN_ALLOW_UNAUTHED_LOCALHOST=true makes
        # the admin endpoints trust any loopback caller. TestClient's
        # default client.host is "testclient" (NOT localhost) so this
        # exercises both branches — the test ALSO confirms a non-loopback
        # host with the bypass enabled still fails (no escalation).
        monkeypatch.setenv("FITSCAN_ALLOW_UNAUTHED_LOCALHOST", "true")
        response = client.get("/api/v1/admin/celery/status")
        # TestClient is not localhost — bypass should NOT apply.
        assert response.status_code == 401

    def test_localhost_bypass_with_loopback_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force TestClient to set request.client.host = "127.0.0.1" so
        # the bypass branch fires.
        monkeypatch.setenv("FITSCAN_ALLOW_UNAUTHED_LOCALHOST", "true")
        # Provide a dummy admin key so the OTHER branch (no env set)
        # doesn't return 503 if the bypass DOESN'T fire.
        monkeypatch.setenv("FITSCAN_ADMIN_API_KEY", "irrelevant-since-bypassed")

        loopback_client = TestClient(app, client=("127.0.0.1", 12345))
        response = loopback_client.get("/api/v1/admin/celery/status")
        assert response.status_code == 200


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


class TestUsageAuth:
    def test_401_when_key_missing(self, client: TestClient) -> None:
        response = client.get("/api/v1/admin/usage")
        assert response.status_code == 401

    def test_401_when_key_wrong(self, client: TestClient) -> None:
        response = client.get(
            "/api/v1/admin/usage",
            headers={"X-Admin-API-Key": "wrong"},
        )
        assert response.status_code == 401


class TestUsageShape:
    def test_returns_zero_state_when_no_runs_recorded(self, client: TestClient) -> None:
        # Ensure measurement_runs is empty in the test DB.
        from sqlalchemy.orm import Session

        from fitscan.infrastructure.result_store import (
            MeasurementRunRecord,
            engine,
        )

        with Session(engine) as session:
            session.query(MeasurementRunRecord).delete()
            session.commit()

        response = client.get(
            "/api/v1/admin/usage",
            headers={"X-Admin-API-Key": ADMIN_KEY},
        )
        assert response.status_code == 200
        body = response.json()

        assert body["total_measurements"] == 0
        assert body["successful_measurements"] == 0
        assert body["error_rate"] == 0.0
        assert body["avg_latency_ms"] is None
        assert body["p95_latency_ms"] is None
        assert body["by_day"] == []
        assert body["by_error_kind"] == []

    def test_aggregates_seeded_runs_correctly(self, client: TestClient) -> None:
        from datetime import datetime, timedelta, timezone

        from sqlalchemy.orm import Session

        from fitscan.infrastructure.result_store import (
            MeasurementRunRecord,
            engine,
        )

        with Session(engine) as session:
            session.query(MeasurementRunRecord).delete()
            session.commit()

        now = datetime.now(timezone.utc)

        with Session(engine) as session:
            # 4 successful runs with varying latency.
            for ms in (100, 200, 300, 400):
                session.add(
                    MeasurementRunRecord(
                        started_at=now - timedelta(minutes=ms),
                        completed_at=now - timedelta(minutes=ms - 1),
                        latency_ms=ms,
                        success=True,
                        error_kind=None,
                    )
                )
            # 1 failed run with kind=validation.
            session.add(
                MeasurementRunRecord(
                    started_at=now - timedelta(minutes=10),
                    completed_at=now - timedelta(minutes=9),
                    latency_ms=50,
                    success=False,
                    error_kind="validation",
                )
            )
            session.commit()

        response = client.get(
            "/api/v1/admin/usage",
            headers={"X-Admin-API-Key": ADMIN_KEY},
        )
        assert response.status_code == 200
        body = response.json()

        assert body["total_measurements"] == 5
        assert body["successful_measurements"] == 4
        assert body["error_rate"] == 0.2
        # Avg of 100/200/300/400 = 250.
        assert body["avg_latency_ms"] == 250
        # by_error_kind has the validation bucket.
        kinds = {row["error_kind"]: row["count"] for row in body["by_error_kind"]}
        assert kinds.get("validation") == 1

    def test_clamps_days_window_to_safe_range(self, client: TestClient) -> None:
        # ?days=10000 must clamp; nobody needs a 27-year history page.
        response = client.get(
            "/api/v1/admin/usage?days=10000",
            headers={"X-Admin-API-Key": ADMIN_KEY},
        )
        assert response.status_code == 200
        body = response.json()
        # 365 max => the from/until span is at most ~366 days.
        from datetime import datetime

        f = datetime.fromisoformat(body["from"])
        u = datetime.fromisoformat(body["until"])
        assert (u - f).days <= 366
