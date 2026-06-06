"""`AnthropicApiProvider`, the direct Anthropic Messages API backend.

This is the default provider. It passes the request through to
`client.messages.create(...)` unchanged, so the temperature, prompt-cache
`cache_control`, and `metadata.user_id` controls the measurement pipeline
relies on all keep working exactly as before.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fitscan.llm.provider import Messages


@lru_cache(maxsize=4)
def _anthropic_client(api_key: str) -> Any:
    """Memoize one Anthropic client per API key (FS-NIDLO-OPS-05).

    Constructing `Anthropic(...)` per call tears down + rebuilds the
    underlying httpx connection pool every time, measurably increasing
    tail latency under load. A module-level `lru_cache` keyed on the API
    key keeps a stable client per key (we only ever have one in practice;
    the cache size is room for test fixtures that mint per-test keys).
    """
    from anthropic import Anthropic

    return Anthropic(api_key=api_key)


class AnthropicApiProvider:
    """Calls Claude through the Anthropic Messages API.

    `is_enabled` is false for an empty key or the well-known
    `test_key_placeholder`, which keeps the unit-test gate behaviour the
    validators relied on before the provider seam existed.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or ""

    @property
    def name(self) -> str:
        return "api"

    @property
    def is_enabled(self) -> bool:
        return bool(self._api_key) and self._api_key != "test_key_placeholder"

    def complete(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: Messages,
        system: str | list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        client = _anthropic_client(self._api_key)
        # Only forward the optional fields the caller actually set, so the
        # call is identical to the pre-seam `messages.create(...)` (passing
        # `system=None` / `metadata=None` would change SDK behaviour).
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        if metadata is not None:
            kwargs["metadata"] = metadata

        response = self._create(client, kwargs)
        return response.content[0].text

    @staticmethod
    def _create(client: Any, kwargs: dict[str, Any]) -> Any:
        """Call `messages.create`, retrying once without `temperature` if the
        model deprecates it.

        Adaptive-thinking models (Opus 4.7+/4.8) reject `temperature` with a
        400 `invalid_request_error`. The callers hardcode `temperature=0` for
        determinism on Sonnet; rather than hardcode a model capability list,
        we drop the param only when the API tells us it isn't supported.
        """
        try:
            return client.messages.create(**kwargs)
        except Exception as e:  # noqa: BLE001 (narrow re-raise below)
            msg = str(e).lower()
            if "temperature" in kwargs and "temperature" in msg and "deprecat" in msg:
                kwargs.pop("temperature")
                return client.messages.create(**kwargs)
            raise
