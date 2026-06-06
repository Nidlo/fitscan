"""Swappable LLM-provider layer for Fitscan.

Every Claude call in the service (vision measurement, validation,
correction, height estimation, landmark verification, and style
classification) goes through the `LLMProvider` seam. Two concrete
backends exist:

  - `AnthropicApiProvider`, the direct Anthropic Messages API (default).
    Preserves temperature, prompt caching, and `metadata.user_id`.
  - `ClaudeCodeProvider`, routes the same calls through Claude Code via
    the `claude-agent-sdk` package, so the worker can run off a Claude
    subscription (`claude login` / `setup-token`) instead of metered API
    billing.

Selection is env-driven (`FITSCAN_LLM_PROVIDER`); see `factory.create_llm_provider`.
"""

from __future__ import annotations

from fitscan.llm.factory import create_llm_provider
from fitscan.llm.provider import LLMProvider

__all__ = ["LLMProvider", "create_llm_provider"]
