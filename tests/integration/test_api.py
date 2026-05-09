"""Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient

from fitscan.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"


class TestMeasureEndpoint:
    def test_measure_requires_image(self):
        response = client.post(
            "/api/v1/measure",
            data={"height_cm": "170"},
        )
        assert response.status_code == 422

    def test_measure_requires_height_or_reference(self):
        import io

        dummy = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        response = client.post(
            "/api/v1/measure",
            files={"image": ("test.jpg", dummy, "image/jpeg")},
        )
        assert response.status_code == 400


class TestKeyRegistration:
    """FS-NIDLO-AUTH-03 admin-only lockdown.

    Public unauthenticated calls now 401. Snad mints keys via curl
    with `X-Admin-API-Key`. Same gate as `/api/v1/admin/*`.
    """

    ADMIN_KEY = "test-admin-key-do-not-use-in-prod"
    HEADERS = {"X-Admin-API-Key": ADMIN_KEY}

    @pytest.fixture(autouse=True)
    def _set_admin_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FITSCAN_ADMIN_API_KEY", self.ADMIN_KEY)

    def test_register_returns_api_key(self):
        response = client.post(
            "/api/v1/keys/register",
            headers=self.HEADERS,
            data={
                "owner_name": "Test User",
                "owner_email": "test@example.com",
                "tier": "free",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["api_key"].startswith("fsk_")
        assert data["tier"] == "free"
        assert data["monthly_limit"] == 50

    def test_register_rejects_unknown_tier(self):
        # Tier allowlist (pre-existing): any string was stored
        # verbatim while the dict lookup silently fell back to
        # free's limit, producing rows that claimed "tier=enterprise"
        # with a free quota. Now an unknown tier is a 400.
        response = client.post(
            "/api/v1/keys/register",
            headers=self.HEADERS,
            data={
                "owner_name": "Test User",
                "owner_email": "test@example.com",
                "tier": "diamond",
            },
        )
        assert response.status_code == 400
        assert "Unknown tier" in response.json()["detail"]

    def test_register_accepts_pro_and_enterprise(self):
        for tier, limit in [("pro", 1000), ("enterprise", 999999)]:
            response = client.post(
                "/api/v1/keys/register",
                headers=self.HEADERS,
                data={
                    "owner_name": "Test User",
                    "owner_email": f"{tier}@example.com",
                    "tier": tier,
                },
            )
            assert response.status_code == 200, f"tier={tier} rejected"
            data = response.json()
            assert data["tier"] == tier
            assert data["monthly_limit"] == limit

    def test_register_401_without_admin_header(self):
        # FS-NIDLO-AUTH-03 — public unauthenticated calls must fail
        # at the gate, before tier validation, before any DB write.
        response = client.post(
            "/api/v1/keys/register",
            data={
                "owner_name": "Anyone",
                "owner_email": "anyone@example.com",
                "tier": "free",
            },
        )
        assert response.status_code == 401

    def test_register_401_with_wrong_admin_key(self):
        response = client.post(
            "/api/v1/keys/register",
            headers={"X-Admin-API-Key": "not-the-right-key"},
            data={
                "owner_name": "Anyone",
                "owner_email": "anyone@example.com",
                "tier": "free",
            },
        )
        assert response.status_code == 401

    def test_register_503_when_admin_env_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # An operator who forgets to set FITSCAN_ADMIN_API_KEY in
        # production must NOT get a silently-public endpoint —
        # we'd rather fail noisily.
        monkeypatch.setenv("FITSCAN_ADMIN_API_KEY", "")
        response = client.post(
            "/api/v1/keys/register",
            headers=self.HEADERS,
            data={
                "owner_name": "Test",
                "owner_email": "t@example.com",
                "tier": "free",
            },
        )
        assert response.status_code == 503
