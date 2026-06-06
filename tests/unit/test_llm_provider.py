"""Tests for the swappable LLM-provider layer (fitscan/llm/).

Covers provider selection, the Claude Code auth-env scrub (the
security-critical bit), image-to-temp-file translation, and that the
validators drive whichever provider is injected.
"""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from fitscan.domain.models import BodyMeasurements
from fitscan.llm.anthropic_api_provider import AnthropicApiProvider
from fitscan.llm.claude_code_provider import (
    AUTH_API_KEY,
    AUTH_CLI_SESSION,
    AUTH_OAUTH_TOKEN,
    ClaudeCodeProvider,
)
from fitscan.llm.factory import create_llm_provider
from fitscan.llm.provider import LLMProvider
from fitscan.validation.claude_validator import ClaudeValidator


class RecordingProvider:
    """Minimal LLMProvider that records the last request and returns a
    canned text. Lets us assert the validators call the seam correctly."""

    def __init__(self, response: str, enabled: bool = True) -> None:
        self._response = response
        self._enabled = enabled
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "recording"

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def complete(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return self._response


class TestProviderProtocol:
    def test_recording_provider_satisfies_protocol(self):
        assert isinstance(RecordingProvider("{}"), LLMProvider)

    def test_api_provider_satisfies_protocol(self):
        assert isinstance(AnthropicApiProvider("k"), LLMProvider)

    def test_claude_code_provider_satisfies_protocol(self):
        assert isinstance(ClaudeCodeProvider(), LLMProvider)


class TestAnthropicApiProvider:
    def test_disabled_for_empty_key(self):
        assert AnthropicApiProvider("").is_enabled is False

    def test_disabled_for_placeholder_key(self):
        assert AnthropicApiProvider("test_key_placeholder").is_enabled is False

    def test_enabled_for_real_key(self):
        assert AnthropicApiProvider("sk-ant-real").is_enabled is True

    def test_complete_passes_through_optional_fields(self, monkeypatch):
        captured: dict = {}

        class _FakeContent:
            text = '{"ok": true}'

        class _FakeResponse:
            content = [_FakeContent()]

        class _FakeMessages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return _FakeResponse()

        class _FakeAnthropic:
            def __init__(self, api_key=None):
                self.messages = _FakeMessages()

        import anthropic

        monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)

        provider = AnthropicApiProvider("sk-ant-passthrough-test")
        out = provider.complete(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            metadata={"user_id": "u1"},
        )
        assert out == '{"ok": true}'
        assert captured["model"] == "claude-sonnet-4-6"
        assert captured["temperature"] == 0.0
        assert captured["metadata"] == {"user_id": "u1"}
        # system was not provided → must not be forwarded
        assert "system" not in captured

    def test_drops_temperature_when_model_deprecates_it(self, monkeypatch):
        """Opus 4.8 rejects `temperature`; the provider retries without it."""
        calls: list = []

        class _FakeContent:
            text = "{}"

        class _FakeResponse:
            content = [_FakeContent()]

        class _FakeMessages:
            def create(self, **kwargs):
                calls.append(kwargs)
                if "temperature" in kwargs:
                    raise RuntimeError(
                        "Error code: 400 - `temperature` is deprecated for this model."
                    )
                return _FakeResponse()

        class _FakeAnthropic:
            def __init__(self, api_key=None):
                self.messages = _FakeMessages()

        import anthropic

        monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)

        provider = AnthropicApiProvider("sk-ant-temp-deprecation-test")
        out = provider.complete(
            model="claude-opus-4-8",
            max_tokens=128,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
        )
        assert out == "{}"
        assert len(calls) == 2  # first with temperature, retry without
        assert "temperature" in calls[0]
        assert "temperature" not in calls[1]


class TestClaudeCodeAuthEnv:
    """The env scrub is the security-critical bit: ANTHROPIC_API_KEY beats
    a subscription token, so the subscription modes must remove it."""

    def test_api_key_mode_sets_key_and_drops_token(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "stale-token")
        provider = ClaudeCodeProvider(auth=AUTH_API_KEY, api_key="sk-ant-x")
        env = provider.build_env()
        assert env["ANTHROPIC_API_KEY"] == "sk-ant-x"
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in env

    def test_oauth_mode_scrubs_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-shadow")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-shadow")
        provider = ClaudeCodeProvider(auth=AUTH_OAUTH_TOKEN, oauth_token="oauth-123")
        env = provider.build_env()
        assert "ANTHROPIC_API_KEY" not in env
        assert "ANTHROPIC_AUTH_TOKEN" not in env
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-123"

    def test_cli_session_mode_scrubs_all_inline_creds(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-shadow")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-shadow")
        provider = ClaudeCodeProvider(auth=AUTH_CLI_SESSION)
        env = provider.build_env()
        assert "ANTHROPIC_API_KEY" not in env
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in env

    def test_is_enabled_per_mode(self):
        assert ClaudeCodeProvider(auth=AUTH_API_KEY, api_key="k").is_enabled is True
        assert ClaudeCodeProvider(auth=AUTH_API_KEY, api_key="").is_enabled is False
        oauth_set = ClaudeCodeProvider(auth=AUTH_OAUTH_TOKEN, oauth_token="t")
        assert oauth_set.is_enabled is True
        oauth_empty = ClaudeCodeProvider(auth=AUTH_OAUTH_TOKEN, oauth_token="")
        assert oauth_empty.is_enabled is False
        assert ClaudeCodeProvider(auth=AUTH_CLI_SESSION).is_enabled is True


class TestClaudeCodeTranslation:
    def test_system_blocks_flattened_to_text(self):
        system = [{"type": "text", "text": "rules", "cache_control": {"type": "e"}}]
        assert ClaudeCodeProvider._system_to_text(system) == "rules"
        assert ClaudeCodeProvider._system_to_text("plain") == "plain"
        assert ClaudeCodeProvider._system_to_text(None) is None

    def test_strip_fences(self):
        fenced = '```json\n{"a": 1}\n```'
        assert ClaudeCodeProvider._strip_fences(fenced) == '{"a": 1}'
        assert ClaudeCodeProvider._strip_fences('{"a": 1}') == '{"a": 1}'

    def test_images_written_to_temp_and_referenced(self, tmp_path):
        img_b64 = base64.b64encode(b"\xff\xd8\xff\xfake-jpeg").decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "data": img_b64},
                    },
                    {"type": "text", "text": "measure this"},
                ],
            }
        ]
        prompt = ClaudeCodeProvider._messages_to_prompt(messages, tmp_path)
        written = list(tmp_path.glob("image_*.jpg"))
        assert len(written) == 1
        assert str(written[0]) in prompt
        assert "measure this" in prompt
        assert "Read tool" in prompt

    def test_model_override_replaces_caller_model(self, monkeypatch):
        seen: dict = {}

        def _fake_invoke(self, *, system_text, prompt, model, max_tokens, workdir):
            seen["model"] = model
            return "{}"

        monkeypatch.setattr(ClaudeCodeProvider, "_invoke", _fake_invoke)
        provider = ClaudeCodeProvider(
            auth=AUTH_API_KEY, api_key="k", model="claude-opus-4-8"
        )
        provider.complete(
            model="claude-sonnet-4-6",
            max_tokens=128,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert seen["model"] == "claude-opus-4-8"

    def test_no_model_override_keeps_caller_model(self, monkeypatch):
        seen: dict = {}

        def _fake_invoke(self, *, system_text, prompt, model, max_tokens, workdir):
            seen["model"] = model
            return "{}"

        monkeypatch.setattr(ClaudeCodeProvider, "_invoke", _fake_invoke)
        provider = ClaudeCodeProvider(auth=AUTH_API_KEY, api_key="k")
        provider.complete(
            model="claude-sonnet-4-6",
            max_tokens=128,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert seen["model"] == "claude-sonnet-4-6"

    def test_complete_drives_invoke_with_image_on_disk(self, monkeypatch):
        seen: dict = {}

        def _fake_invoke(self, *, system_text, prompt, model, max_tokens, workdir):
            # The referenced image must exist on disk at call time.
            for line in prompt.splitlines():
                line = line.strip()
                if line.startswith("- ") and line.endswith(".jpg"):
                    assert Path(line[2:]).exists()
            seen["system_text"] = system_text
            seen["model"] = model
            return '```json\n{"is_valid": true}\n```'

        monkeypatch.setattr(ClaudeCodeProvider, "_invoke", _fake_invoke)

        img_b64 = base64.b64encode(b"jpegbytes").decode()
        provider = ClaudeCodeProvider(auth=AUTH_API_KEY, api_key="k")
        out = provider.complete(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=[{"type": "text", "text": "technique"}],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "data": img_b64},
                        },
                        {"type": "text", "text": "go"},
                    ],
                }
            ],
        )
        # Fences stripped by the provider.
        assert out == '{"is_valid": true}'
        assert seen["system_text"] == "technique"
        assert seen["model"] == "claude-sonnet-4-6"


class TestFactory:
    def test_default_is_api_provider(self):
        provider = create_llm_provider(provider="api", api_key="k")
        assert isinstance(provider, AnthropicApiProvider)

    def test_claude_code_selected(self):
        provider = create_llm_provider(
            provider="claude_code",
            api_key="k",
            claude_code_auth="api_key",
        )
        assert isinstance(provider, ClaudeCodeProvider)

    def test_unknown_provider_falls_back_to_api(self):
        provider = create_llm_provider(provider="nonsense", api_key="k")
        assert isinstance(provider, AnthropicApiProvider)

    def test_mcp_envelope_parsed(self):
        cfg = json.dumps({"mcpServers": {"db": {"command": "x"}}})
        provider = create_llm_provider(
            provider="claude_code",
            claude_code_auth="cli_session",
            claude_code_mcp_servers=cfg,
        )
        assert provider._mcp_servers == {"db": {"command": "x"}}

    def test_invalid_mcp_json_ignored(self):
        provider = create_llm_provider(
            provider="claude_code",
            claude_code_auth="cli_session",
            claude_code_mcp_servers="{not json",
        )
        assert provider._mcp_servers == {}


class TestValidatorUsesInjectedProvider:
    def test_validate_uses_provider_and_forwards_metadata(self):
        provider = RecordingProvider(
            '{"is_valid": true, "flagged_measurements": [],'
            ' "photo_quality_score": 9.0, "quality_issues": [],'
            ' "recommendation": "proceed", "notes": "ok"}'
        )
        validator = ClaudeValidator(llm=provider)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        m = BodyMeasurements(source="test")
        result = validator.validate(image, m, subject_id="user-xyz")

        assert result.photo_quality_score == 9.0
        assert len(provider.calls) == 1
        assert provider.calls[0]["metadata"] == {"user_id": "user-xyz"}

    def test_disabled_provider_skips_call(self):
        provider = RecordingProvider("{}", enabled=False)
        validator = ClaudeValidator(llm=provider)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        m = BodyMeasurements(source="test")
        result = validator.validate(image, m, subject_id="user-xyz")

        assert result.is_valid  # default-pass
        assert provider.calls == []

    def test_consent_gate_still_blocks_without_subject_id(self):
        provider = RecordingProvider("{}")
        validator = ClaudeValidator(llm=provider)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        m = BodyMeasurements(source="test")
        validator.validate(image, m)  # no subject_id

        assert provider.calls == []  # no photo shipped

    def test_measure_forwards_system_block(self):
        provider = RecordingProvider('{"measurements_cm": {}}')
        validator = ClaudeValidator(llm=provider)
        front = np.zeros((100, 100, 3), dtype=np.uint8)
        validator.measure(front, None, height_cm=170.0, subject_id="u1")

        assert len(provider.calls) == 1
        system = provider.calls[0]["system"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}


class _SequenceProvider:
    """Returns each canned response once. Thread-safe: the ensemble runs the
    K calls concurrently, so hand out responses under a lock (order-independent
    since the fusion takes a median)."""

    def __init__(self, responses: list[str]) -> None:
        self._remaining = list(responses)
        self._lock = threading.Lock()
        self.calls = 0

    @property
    def name(self) -> str:
        return "sequence"

    @property
    def is_enabled(self) -> bool:
        return True

    def complete(self, **kwargs: Any) -> str:
        with self._lock:
            self.calls += 1
            return self._remaining.pop()


def _waist(value: float) -> str:
    return (
        '{"measurements_cm": {"waist": {"value": '
        + str(value)
        + ', "confidence": 0.7, "note": ""}}}'
    )


class TestMeasureEnsemble:
    def test_ensemble_runs_k_times_and_medians(self):
        provider = _SequenceProvider([_waist(90.0), _waist(95.0), _waist(100.0)])
        validator = ClaudeValidator(llm=provider, ensemble_size=3)
        front = np.zeros((100, 100, 3), dtype=np.uint8)
        m = validator.measure(front, None, height_cm=170.0, subject_id="u1")

        assert provider.calls == 3
        assert m is not None
        assert m.waist_cm.value_cm == 95.0  # median of 90/95/100

    def test_ensemble_rejects_single_spike(self):
        # Two consistent reads + one wild spike -> median keeps the consistent.
        provider = _SequenceProvider([_waist(95.0), _waist(95.0), _waist(130.0)])
        validator = ClaudeValidator(llm=provider, ensemble_size=3)
        front = np.zeros((100, 100, 3), dtype=np.uint8)
        m = validator.measure(front, None, height_cm=170.0, subject_id="u1")

        assert m is not None
        assert m.waist_cm.value_cm == 95.0  # 130 spike discarded by median

    def test_ensemble_size_one_is_single_call(self):
        provider = _SequenceProvider([_waist(95.0)])
        validator = ClaudeValidator(llm=provider, ensemble_size=1)
        front = np.zeros((100, 100, 3), dtype=np.uint8)
        validator.measure(front, None, height_cm=170.0, subject_id="u1")
        assert provider.calls == 1
