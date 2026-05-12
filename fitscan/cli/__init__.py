"""FitScan admin CLI module — invoked via ``python -m fitscan.cli``.

The legacy `cli/fitscan_cli.py` script (sibling of the package) still
hosts the measurement subcommands. This package adds operational
commands used to bootstrap a fresh deployment — currently just
`set-admin-key`, which writes / rotates `FITSCAN_ADMIN_API_KEY` in
the local `.env` so the operator doesn't have to edit it by hand.
"""
