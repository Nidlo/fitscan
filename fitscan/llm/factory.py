"""Factory: select the configured `LLMProvider` from settings.

Keeps provider construction in one place so both wiring sites
(`measurement_service.create_pipeline` for the validator, and the style
route for `StyleAnalyzer`) resolve the same way.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from fitscan.llm.anthropic_api_provider import AnthropicApiProvider
from fitscan.llm.claude_code_provider import ClaudeCodeProvider
from fitscan.llm.provider import LLMProvider

PROVIDER_API = "api"
PROVIDER_CLAUDE_CODE = "claude_code"


def _load_mcp_servers(raw: str) -> dict[str, Any]:
    """Parse the optional MCP-servers JSON config.

    Accepts either the full `{"mcpServers": {...}}` envelope or a bare
    `{...}` map of server-name → config. Returns an empty map (MCP
    disabled) on empty/invalid input rather than failing the worker.
    """
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "claude_code: FITSCAN_CLAUDE_CODE_MCP_SERVERS is not valid JSON; ignoring"
        )
        return {}
    if isinstance(parsed, dict) and isinstance(parsed.get("mcpServers"), dict):
        return parsed["mcpServers"]
    if isinstance(parsed, dict):
        return parsed
    logger.warning("claude_code: MCP config must be a JSON object; ignoring")
    return {}


def create_llm_provider(
    *,
    provider: str = PROVIDER_API,
    api_key: str = "",
    claude_code_auth: str = "api_key",
    claude_code_oauth_token: str = "",
    claude_code_model: str = "",
    claude_code_mcp_servers: str = "",
) -> LLMProvider:
    """Build the provider named by `provider`.

    :param provider: `api` (default) or `claude_code`.
    :param api_key: Anthropic API key, used by the API provider, and by the
        Claude Code provider when its auth mode is `api_key`.
    :param claude_code_auth: `api_key` / `oauth_token` / `cli_session`.
    :param claude_code_oauth_token: token from `claude setup-token`.
    :param claude_code_model: model override for the Claude Code path
        (defaults to Opus); empty keeps the caller's per-call model.
    :param claude_code_mcp_servers: optional MCP-servers JSON.
    :returns: the configured provider. Unknown `provider` values fall back
        to the API provider with a warning (fail safe, not closed).
    """
    if provider == PROVIDER_CLAUDE_CODE:
        return ClaudeCodeProvider(
            auth=claude_code_auth,
            api_key=api_key,
            oauth_token=claude_code_oauth_token,
            model=claude_code_model,
            mcp_servers=_load_mcp_servers(claude_code_mcp_servers),
        )
    if provider != PROVIDER_API:
        logger.warning(
            f"unknown FITSCAN_LLM_PROVIDER={provider!r}; using the direct API"
        )
    return AnthropicApiProvider(api_key=api_key)
