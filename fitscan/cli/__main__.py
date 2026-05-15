"""Entry point for ``python -m fitscan.cli``.

Each subcommand is a small Typer command exported from a sibling
module. Keep this module thin — the actual work lives in the
sibling files so tests can import them directly without going
through Typer's runner.
"""

from __future__ import annotations

import typer

from .admin_key import set_admin_key

app = typer.Typer(
    name="fitscan-cli",
    help="FitScan operator CLI (rotate admin API key, etc.).",
    no_args_is_help=True,
)

# `python -m fitscan.cli set-admin-key <key>`
app.command(name="set-admin-key")(set_admin_key)


if __name__ == "__main__":
    app()
