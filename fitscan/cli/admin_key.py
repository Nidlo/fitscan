"""`fitscan.cli` set-admin-key — bootstrap or rotate `FITSCAN_ADMIN_API_KEY`.

Reads / writes the local `.env` so the operator doesn't have to edit
it by hand after generating a key on the Nidlo admin dashboard. We
intentionally don't auto-restart Fitscan — that's a deploy concern
(systemd, supervisor, docker, dev tmux session, etc.) and Fitscan
operators reasonably want explicit control.

Usage:
    python -m fitscan.cli set-admin-key <KEY>
    python -m fitscan.cli set-admin-key <KEY> --env-file /path/to/.env

Idempotent: re-running with the same key is a no-op.
"""

from __future__ import annotations

from pathlib import Path

import typer

ENV_KEY = "FITSCAN_ADMIN_API_KEY"


def set_admin_key(
    key: str = typer.Argument(..., help="The shared secret to install."),
    env_file: Path = typer.Option(
        Path(".env"),
        "--env-file",
        "-e",
        help="Path to the env file to update (defaults to ./.env).",
    ),
) -> None:
    """Write or update `FITSCAN_ADMIN_API_KEY` in `env_file`."""
    key = key.strip()
    if not key:
        typer.echo("Refusing to set an empty key.", err=True)
        raise typer.Exit(code=2)

    if not env_file.exists():
        env_file.write_text(f"{ENV_KEY}={key}\n", encoding="utf-8")
        typer.echo(f"Created {env_file} with {ENV_KEY}.")
        typer.echo("Restart the Fitscan service for the new key to take effect.")
        return

    contents = env_file.read_text(encoding="utf-8")
    lines = contents.splitlines()

    found = False
    out_lines: list[str] = []
    for line in lines:
        # Match the exact key with optional leading whitespace; ignore
        # commented lines (legacy notes the operator may have left).
        stripped = line.lstrip()
        if stripped.startswith(f"{ENV_KEY}=") and not stripped.startswith("#"):
            out_lines.append(f"{ENV_KEY}={key}")
            found = True
        else:
            out_lines.append(line)

    if not found:
        out_lines.append(f"{ENV_KEY}={key}")

    # Preserve trailing newline shape of the original file when possible.
    trailing = "\n" if contents.endswith("\n") else ""
    env_file.write_text("\n".join(out_lines) + trailing, encoding="utf-8")

    action = "Updated" if found else "Appended"
    typer.echo(f"{action} {ENV_KEY} in {env_file}.")
    typer.echo("Restart the Fitscan service for the new key to take effect.")
