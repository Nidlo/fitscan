"""Unit tests for the `python -m fitscan.cli set-admin-key` flow.

Direct calls into the command function (no subprocess) — Typer's
runner is fine but adds noise; calling the function with kwargs is
the cleanest way to assert file mutations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from fitscan.cli.admin_key import set_admin_key


def test_creates_env_file_when_missing(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    assert not env.exists()

    set_admin_key("my-secret-key", env_file=env)

    assert env.exists()
    assert "FITSCAN_ADMIN_API_KEY=my-secret-key" in env.read_text()


def test_updates_existing_key_in_place(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "OTHER=value\n" "FITSCAN_ADMIN_API_KEY=old-key\n" "TAIL_LINE=present\n",
        encoding="utf-8",
    )

    set_admin_key("new-key", env_file=env)

    text = env.read_text()
    assert "FITSCAN_ADMIN_API_KEY=new-key" in text
    assert "old-key" not in text
    # Surrounding lines untouched.
    assert "OTHER=value" in text
    assert "TAIL_LINE=present" in text


def test_appends_when_key_not_present(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("OTHER=value\n", encoding="utf-8")

    set_admin_key("appended-key", env_file=env)

    text = env.read_text()
    assert "OTHER=value" in text
    assert "FITSCAN_ADMIN_API_KEY=appended-key" in text


def test_ignores_commented_lookalike_lines(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# FITSCAN_ADMIN_API_KEY=legacy-commented-out\n" "OTHER=value\n",
        encoding="utf-8",
    )

    set_admin_key("real-key", env_file=env)

    text = env.read_text()
    # The commented line must survive — operators sometimes keep it as a note.
    assert "# FITSCAN_ADMIN_API_KEY=legacy-commented-out" in text
    # AND the real key was appended (since no uncommented match was found).
    assert "FITSCAN_ADMIN_API_KEY=real-key" in text


def test_refuses_empty_key(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("OTHER=value\n", encoding="utf-8")

    with pytest.raises(typer.Exit) as exc_info:
        set_admin_key("   ", env_file=env)

    assert exc_info.value.exit_code == 2
    # File untouched.
    assert env.read_text() == "OTHER=value\n"
