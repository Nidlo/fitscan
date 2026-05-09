"""Resend transactional email client (FS-NIDLO-NOTIF-01).

Tiny HTTP wrapper over Resend's REST API. The fitscan service has only
a handful of email touchpoints (measurement-complete confirmation,
retention-purge notice) — wiring up a Mail framework would be overkill.
This file is the whole email surface.

The client is a no-op when `RESEND_API_KEY` is empty so a fresh
checkout / CI run doesn't need a Resend account. Production deploys
must set the key + a verified sender domain.

Usage::

    from fitscan.infrastructure.resend_client import ResendClient

    ResendClient().send(
        to="user@example.com",
        subject="Your measurement is ready",
        html="<p>...</p>",
    )

Synchronous on purpose — `httpx.Client` is what the rest of the
infrastructure layer uses, and adding async would force every caller
to be inside a coroutine. The send is fast and rare; sync is fine.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

_RESEND_API_BASE = "https://api.resend.com"


class ResendClient:
    """Thin wrapper. Reads `RESEND_API_KEY` + `RESEND_FROM_ADDRESS` +
    `RESEND_FROM_NAME` from env so dev/test/prod can swap without
    touching code. Constructor params are an escape hatch for tests
    that want to inject a stub key + a custom transport."""

    def __init__(
        self,
        api_key: str | None = None,
        from_address: str | None = None,
        from_name: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._api_key = (
            api_key if api_key is not None else os.getenv("RESEND_API_KEY", "")
        )
        self._from_address = (
            from_address
            if from_address is not None
            else os.getenv("RESEND_FROM_ADDRESS", "noreply@nidlo.test")
        )
        self._from_name = (
            from_name
            if from_name is not None
            else os.getenv("RESEND_FROM_NAME", "Nidlo Fitscan")
        )
        self._transport = transport
        self._timeout_seconds = timeout_seconds

    def send(
        self,
        to: str,
        subject: str,
        html: str,
        unsubscribe_url: str | None = None,
    ) -> str | None:
        """Send a single transactional email. Returns the Resend
        message id on success, None on no-op (no API key) or a recorded
        failure. Failures are logged at WARNING — fitscan emails are
        nice-to-have, not part of the measurement contract."""
        if not self._api_key:
            logger.info(
                "ResendClient skipped: no RESEND_API_KEY (to={to} subject={subj})",
                to=to,
                subj=subject,
            )
            return None

        from_field = f"{self._from_name} <{self._from_address}>"
        headers: dict[str, str] = {}
        if unsubscribe_url:
            headers["List-Unsubscribe"] = f"<{unsubscribe_url}>"
            headers["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"

        payload: dict[str, Any] = {
            "from": from_field,
            "to": [to],
            "subject": subject,
            "html": html,
        }
        if headers:
            payload["headers"] = headers

        client_kwargs: dict[str, Any] = {"timeout": self._timeout_seconds}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        try:
            with httpx.Client(**client_kwargs) as client:
                response = client.post(
                    f"{_RESEND_API_BASE}/emails",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

            if response.status_code >= 400:
                logger.warning(
                    "ResendClient send failed: http={status} to={to} body={body}",
                    status=response.status_code,
                    to=to,
                    body=response.text[:300],
                )
                return None

            data = response.json()
            return str(data.get("id")) if isinstance(data, dict) else None
        except httpx.HTTPError as exc:
            logger.warning(
                "ResendClient transport error: to={to} exc={exc}",
                to=to,
                exc=str(exc),
            )
            return None
