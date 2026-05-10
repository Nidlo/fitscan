"""Alembic environment.

DATABASE_URL is sourced from the environment (matching how
`fitscan.infrastructure.result_store` resolves it at import time). In CI
the workflow writes a `.env.test` file before invoking alembic — we load
that explicitly here because alembic doesn't go through Pydantic Settings.

We set `FITSCAN_AUTO_CREATE_TABLES=false` before importing Base so
result_store's import-time `Base.metadata.create_all(engine)` is skipped.
That keeps alembic as the source of truth for schema in environments that
opt in (CI integration tests, prod). Local sqlite dev keeps the old
implicit-create behavior because the env var defaults to true.
"""

import os
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Load .env.test (CI) or .env (dev) before importing anything that reads
# DATABASE_URL. python-dotenv ships transitively via pydantic-settings.
try:
    from dotenv import load_dotenv

    for candidate in (".env.test", ".env"):
        if Path(candidate).exists():
            load_dotenv(candidate, override=False)
            break
except ImportError:
    pass

# Suppress result_store's import-time create_all — alembic owns schema here.
os.environ.setdefault("FITSCAN_AUTO_CREATE_TABLES", "false")

# Import after env loading so the engine binds to the right URL.
from fitscan.infrastructure.result_store import Base  # noqa: E402

config = context.config

# Override the placeholder URL in alembic.ini with the live one.
db_url = os.environ.get("DATABASE_URL", "sqlite:///./fitscan.db")
config.set_main_option("sqlalchemy.url", db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
