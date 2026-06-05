"""`ClaudeCodeProvider`, routes Claude calls through Claude Code.

Instead of the metered Anthropic API, this backend drives Claude Code via
the `claude-agent-sdk` package, so the worker can run off a Claude
subscription (`claude login` / `claude setup-token`). Every `complete()`
call is an isolated, fresh Claude Code session.

Trade-offs versus the direct API (all by design of the Agent SDK, not this
code):
  - No `temperature` control, the SDK does not expose it, so measurement
    determinism is weaker than the API path.
  - No `metadata.user_id` forwarding, the consent gate in the callers
    still applies (a photo is only ever sent when a `subject_id` is
    present), but Anthropic-side retention can't be targeted per user.
  - No prompt-cache `cache_control`.
  - Images are delivered to the model by writing them to temp files and
    letting Claude Code read them with the `Read` tool (the SDK has no
    inline base64 image input); multi-image reliability is weaker.

Auth is env-selected. The provider builds the subprocess environment so the
chosen credential actually wins, in particular it SCRUBS `ANTHROPIC_API_KEY`
for the subscription modes, because the API key otherwise silently shadows a
`CLAUDE_CODE_OAUTH_TOKEN` / logged-in session (the precedence trap Fitscan
already hit in Sprint 34).
"""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from loguru import logger

from fitscan.llm.provider import Messages

# Auth modes (value of FITSCAN_CLAUDE_CODE_AUTH).
AUTH_API_KEY = "api_key"
AUTH_OAUTH_TOKEN = "oauth_token"
AUTH_CLI_SESSION = "cli_session"


class ClaudeCodeProvider:
    """Calls Claude through Claude Code (the `claude-agent-sdk` package)."""

    def __init__(
        self,
        *,
        auth: str = AUTH_API_KEY,
        api_key: str = "",
        oauth_token: str = "",
        model: str = "",
        mcp_servers: dict[str, Any] | None = None,
    ) -> None:
        """:param auth: one of `api_key` / `oauth_token` / `cli_session`.
        :param api_key: `ANTHROPIC_API_KEY` value, used when `auth=api_key`.
        :param oauth_token: `CLAUDE_CODE_OAUTH_TOKEN` value (from
            `claude setup-token`), used when `auth=oauth_token`.
        :param model: optional model override applied to every call. When
            set, it replaces the per-call model the caller passes (the
            callers hardcode Sonnet for the API path; Claude Code defaults
            to Opus). Empty keeps the caller's per-call model.
        :param mcp_servers: optional Agent-SDK `mcp_servers` mapping to
            attach to every request.
        """
        self._auth = auth
        self._api_key = api_key or ""
        self._oauth_token = oauth_token or ""
        self._model = model or ""
        self._mcp_servers = mcp_servers or {}

    @property
    def name(self) -> str:
        return "claude_code"

    @property
    def is_enabled(self) -> bool:
        if self._auth == AUTH_API_KEY:
            return bool(self._api_key) and self._api_key != "test_key_placeholder"
        if self._auth == AUTH_OAUTH_TOKEN:
            return bool(self._oauth_token)
        if self._auth == AUTH_CLI_SESSION:
            # Relies on a prior `claude login` credential on disk; we can't
            # cheaply verify it here, so treat the mode as enabled and let a
            # missing credential surface as a call failure (caller degrades).
            return True
        return False

    def build_env(self) -> dict[str, str]:
        """Build the subprocess environment for the chosen auth mode.

        The scrub is the security-critical bit: precedence is
        `ANTHROPIC_API_KEY` > `CLAUDE_CODE_OAUTH_TOKEN` > logged-in session,
        so the subscription modes must remove a stray API key (and bearer
        token) from the environment or it silently overrides the
        subscription credential.
        """
        env = dict(os.environ)

        def _scrub_api_key() -> None:
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("ANTHROPIC_AUTH_TOKEN", None)

        if self._auth == AUTH_API_KEY:
            if self._api_key:
                env["ANTHROPIC_API_KEY"] = self._api_key
            env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
        elif self._auth == AUTH_OAUTH_TOKEN:
            _scrub_api_key()
            if self._oauth_token:
                env["CLAUDE_CODE_OAUTH_TOKEN"] = self._oauth_token
        elif self._auth == AUTH_CLI_SESSION:
            _scrub_api_key()
            env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)

        return env

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
        # temperature + metadata are intentionally unused: the Agent SDK
        # exposes neither. Logged once at debug so the absence is traceable.
        if temperature is not None:
            logger.debug("claude_code: temperature ignored (SDK does not expose it)")

        system_text = self._system_to_text(system)
        with tempfile.TemporaryDirectory(prefix="fitscan_cc_") as workdir:
            prompt = self._messages_to_prompt(messages, Path(workdir))
            return self._strip_fences(
                self._invoke(
                    system_text=system_text,
                    prompt=prompt,
                    model=self._model or model,
                    max_tokens=max_tokens,
                    workdir=workdir,
                )
            )

    @staticmethod
    def _system_to_text(system: str | list[dict[str, Any]] | None) -> str | None:
        """Flatten Anthropic system blocks (which may carry `cache_control`)
        into the plain string the Agent SDK's `system_prompt` expects."""
        if system is None:
            return None
        if isinstance(system, str):
            return system
        parts = [
            str(block.get("text", ""))
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p) or None

    @staticmethod
    def _messages_to_prompt(messages: Messages, workdir: Path) -> str:
        """Turn Anthropic content-block messages into a single text prompt.

        Image blocks have no inline equivalent in Claude Code, so each is
        written to a JPEG under `workdir` and referenced by path; Claude
        Code reads them with the `Read` tool. Text blocks are appended as-is.
        """
        text_parts: list[str] = []
        image_refs: list[str] = []

        for message in messages:
            content = message.get("content", [])
            if isinstance(content, str):
                text_parts.append(content)
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(str(block.get("text", "")))
                elif btype == "image":
                    source = block.get("source", {})
                    if source.get("type") == "base64" and source.get("data"):
                        idx = len(image_refs) + 1
                        path = workdir / f"image_{idx}.jpg"
                        path.write_bytes(base64.b64decode(source["data"]))
                        image_refs.append(str(path))

        prompt_parts: list[str] = []
        if image_refs:
            listed = "\n".join(f"- {ref}" for ref in image_refs)
            prompt_parts.append(
                "Read the following image file(s) with the Read tool and analyse "
                f"them to answer the request below:\n{listed}"
            )
        prompt_parts.extend(text_parts)
        return "\n\n".join(p for p in prompt_parts if p)

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Strip a leading/trailing markdown code fence if present.

        Claude Code is conversational and tends to wrap JSON in ```json
        fences; the callers' `json.loads` would choke on those. The direct
        API provider leaves output untouched, so this normalisation is
        scoped to the Claude Code path only.
        """
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped
        inner = stripped.split("```", 2)
        if len(inner) < 2:
            return stripped
        body = inner[1]
        for lang in ("json\n", "json ", "json"):
            if body.startswith(lang):
                body = body[len(lang) :]
                break
        body = body.strip()
        if body.endswith("```"):
            body = body[:-3].strip()
        return body

    def _invoke(
        self,
        *,
        system_text: str | None,
        prompt: str,
        model: str,
        max_tokens: int,
        workdir: str,
    ) -> str:
        """Run one fresh Claude Code session and return the assistant text.

        Isolated behind this method so tests can patch it without a real
        `claude-agent-sdk` install. Fields follow the documented Agent SDK
        API; the env scrub from `build_env()` selects the auth credential.
        """
        from claude_agent_sdk import ClaudeAgentOptions, query  # noqa: PLC0415

        options = self._build_options(
            ClaudeAgentOptions,
            system_text=system_text,
            model=model,
            max_tokens=max_tokens,
            workdir=workdir,
        )

        async def _run() -> str:
            chunks: list[str] = []
            async for message in query(prompt=prompt, options=options):
                result = getattr(message, "result", None)
                if isinstance(result, str):
                    chunks.append(result)
                    continue
                content = getattr(message, "content", None)
                if isinstance(content, list):
                    for block in content:
                        block_text = getattr(block, "text", None)
                        if isinstance(block_text, str):
                            chunks.append(block_text)
            return "".join(chunks)

        return self._run_coro(_run())

    def _build_options(
        self,
        options_cls: Any,
        *,
        system_text: str | None,
        model: str,
        max_tokens: int,
        workdir: str,
    ) -> Any:
        """Construct `ClaudeAgentOptions`, including only fields the installed
        SDK version supports (it varies across releases). `Read` is the only
        tool granted, and permissions are pre-approved so the headless worker
        never blocks on a prompt."""
        import dataclasses

        candidate: dict[str, Any] = {
            "model": model,
            "system_prompt": system_text,
            "allowed_tools": ["Read"],
            "permission_mode": "bypassPermissions",
            "max_turns": 4,
            "cwd": workdir,
            "env": self.build_env(),
            "max_tokens": max_tokens,
        }
        if self._mcp_servers:
            candidate["mcp_servers"] = self._mcp_servers

        try:
            supported = {f.name for f in dataclasses.fields(options_cls)}
        except TypeError:
            supported = set(candidate)
        kwargs = {
            k: v for k, v in candidate.items() if k in supported and v is not None
        }
        return options_cls(**kwargs)

    @staticmethod
    def _run_coro(coro: Any) -> str:
        """Drive an async coroutine from sync code.

        The validators run in FastAPI's sync threadpool, but if a running
        event loop is ever present, `asyncio.run` would raise, so the
        coroutine always runs on a fresh loop in a worker thread.
        """
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
