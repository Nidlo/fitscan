"""ResendClient integration tests (FS-NIDLO-NOTIF-01).

Uses httpx.MockTransport so we never touch the real Resend API in CI.
Covers the four interesting paths: no-op when key missing, success,
HTTP error, transport exception.
"""

from __future__ import annotations

import json

import httpx
import pytest

from fitscan.infrastructure.resend_client import ResendClient


def _ok_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["from"].startswith("Nidlo Fitscan <")
    assert body["to"] == ["dest@example.com"]
    return httpx.Response(200, json={"id": "resend-msg-123"})


def _fail_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(401, json={"message": "unauthorized"})


def _boom_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused")


class TestResendClientNoOp:
    def test_returns_none_and_logs_when_api_key_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RESEND_API_KEY", raising=False)
        client = ResendClient()
        result = client.send("dest@example.com", "Subject", "<p>hi</p>")
        assert result is None


class TestResendClientSuccess:
    def test_returns_message_id_on_200(self) -> None:
        transport = httpx.MockTransport(_ok_handler)
        client = ResendClient(
            api_key="test-key",
            from_address="noreply@nidlo.test",
            from_name="Nidlo Fitscan",
            transport=transport,
        )
        msg_id = client.send("dest@example.com", "Subject", "<p>hi</p>")
        assert msg_id == "resend-msg-123"

    def test_attaches_list_unsubscribe_headers_when_url_supplied(self) -> None:
        captured: dict[str, dict[str, str] | None] = {"headers": None}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = json.loads(request.content).get("headers")
            return httpx.Response(200, json={"id": "x"})

        client = ResendClient(
            api_key="k",
            from_address="from@x.test",
            from_name="X",
            transport=httpx.MockTransport(_handler),
        )
        client.send(
            "dest@example.com",
            "Subject",
            "<p>hi</p>",
            unsubscribe_url="https://nidlo.test/unsub/abc",
        )
        assert captured["headers"] is not None
        assert "List-Unsubscribe" in captured["headers"]
        assert (
            captured["headers"]["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"
        )


class TestResendClientFailure:
    def test_returns_none_on_4xx_and_does_not_throw(self) -> None:
        transport = httpx.MockTransport(_fail_handler)
        client = ResendClient(
            api_key="bad-key",
            from_address="noreply@nidlo.test",
            from_name="Nidlo Fitscan",
            transport=transport,
        )
        result = client.send("dest@example.com", "Subject", "<p>hi</p>")
        assert result is None

    def test_returns_none_on_transport_error_and_does_not_throw(self) -> None:
        transport = httpx.MockTransport(_boom_handler)
        client = ResendClient(
            api_key="k",
            from_address="noreply@nidlo.test",
            from_name="Nidlo Fitscan",
            transport=transport,
        )
        result = client.send("dest@example.com", "Subject", "<p>hi</p>")
        assert result is None
