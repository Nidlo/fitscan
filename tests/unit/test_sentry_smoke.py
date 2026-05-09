"""Tests for scripts/sentry_smoke.py.

The smoke script is what an operator runs to verify SENTRY_DSN wiring.
Its happy path (DSN set → event lands in Sentry UI) is exactly what the
operator runs the script for in the first place — that IS the smoke
test, and unit-testing real network calls to Sentry is fragile and
unnecessary.

What we DO test: the skip path. If SENTRY_DSN is unset the script must
exit 0 with a friendly message and must NOT import sentry_sdk (so CI /
fresh installs without the SDK don't crash). A future refactor that
moves the import to the top of the file would silently break this — a
test pins the contract.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "sentry_smoke.py"


@pytest.fixture(autouse=True)
def _no_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTRY_DSN", raising=False)


def _load() -> object:
    spec = importlib.util.spec_from_file_location("sentry_smoke", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skips_with_exit_zero_when_dsn_unset(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["sentry_smoke.py"])
    module = _load()
    assert module.main() == 0

    captured = capsys.readouterr()
    assert "SENTRY_DSN is not set" in captured.out
    assert "python scripts/sentry_smoke.py" in captured.out


def test_skips_on_empty_string_dsn(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SENTRY_DSN", "   ")
    monkeypatch.setattr(sys, "argv", ["sentry_smoke.py"])
    module = _load()
    assert module.main() == 0

    assert "SENTRY_DSN is not set" in capsys.readouterr().out


def test_skip_path_does_not_import_sentry_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("sentry_sdk", None)
    monkeypatch.setattr(sys, "argv", ["sentry_smoke.py"])
    module = _load()
    assert module.main() == 0

    assert "sentry_sdk" not in sys.modules, (
        "skip path must not import sentry_sdk — keeps the script usable on "
        "fresh installs and in CI where the SDK may not be present"
    )


def test_message_override_accepted_on_skip_path(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["sentry_smoke.py", "--message", "custom probe"])
    module = _load()
    assert module.main() == 0
    assert "SENTRY_DSN is not set" in capsys.readouterr().out
