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
    def test_register_returns_api_key(self):
        response = client.post(
            "/api/v1/keys/register",
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
        # Audit fix — previously any string passed for `tier` was stored
        # verbatim while the dict lookup silently fell back to free's
        # limit. The DB row then claimed "tier=enterprise" with a free
        # quota. Now an unknown tier is a 400.
        response = client.post(
            "/api/v1/keys/register",
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
