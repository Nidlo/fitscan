"""The `LLMProvider` seam: a thin, faithful wrapper over the single Claude
call the measurement + style code needs.

The method mirrors the subset of the Anthropic Messages API the callers
actually use (`model`, `max_tokens`, `messages`, `system`, `temperature`,
`metadata`) and returns the assistant's text directly, so a call site goes
from:

    response = client.messages.create(...); raw = response.content[0].text

to:

    raw = self._llm.complete(...)

Providers translate the request to their own transport (direct API vs
Claude Code) and normalise the result back to that text.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# Anthropic-style content-block messages: `[{"role": "user", "content": [...]}]`.
# Heterogeneous by nature (image blocks + text blocks), so the element type
# stays `dict[str, Any]` to match the shape the Anthropic SDK consumes.
Messages = list[dict[str, Any]]


@runtime_checkable
class LLMProvider(Protocol):
    """A backend that turns one Claude request into assistant text."""

    @property
    def name(self) -> str:
        """Short identifier for logs/telemetry, e.g. `"api"` / `"claude_code"`."""
        ...

    @property
    def is_enabled(self) -> bool:
        """Whether this provider is configured to make real calls.

        Callers gate on this (instead of inspecting an API key directly)
        so subscription-authenticated Claude Code counts as enabled even
        with no `ANTHROPIC_API_KEY` present.
        """
        ...

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
        """Run one request and return the assistant's text.

        :param model: Claude model id (e.g. `claude-sonnet-4-6`).
        :param max_tokens: response token cap.
        :param messages: Anthropic-style content-block messages, including
            base64 image blocks for the vision paths.
        :param system: optional system prompt, a plain string or Anthropic
            system blocks (which may carry `cache_control`).
        :param temperature: optional sampling temperature; honoured by the
            direct API, ignored by Claude Code (the SDK does not expose it).
        :param metadata: optional Anthropic request metadata (e.g.
            `{"user_id": subject_id}`); honoured by the direct API, ignored
            by Claude Code.
        :returns: the assistant text. Raises on transport/parse failure so
            the caller's existing try/except degrades to its fallback.
        """
        ...
