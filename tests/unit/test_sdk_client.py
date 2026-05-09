"""Tests for SDK client."""

from fitscan.sdk.client import FitScanClient


class TestFitScanClient:
    def test_init_default(self):
        client = FitScanClient()
        assert client.base == "http://localhost:8000"
        client.close()

    def test_init_custom_url(self):
        client = FitScanClient(base_url="http://custom:9000/")
        assert client.base == "http://custom:9000"
        client.close()

    def test_context_manager(self):
        with FitScanClient() as client:
            assert client.base == "http://localhost:8000"

    def test_health_returns_false_when_unreachable(self):
        client = FitScanClient(base_url="http://localhost:19999", timeout=1.0)
        assert client.health() is False
        client.close()
