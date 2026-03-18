"""
Root conftest.py. Loaded before any test module.
Forces all tests to use .env.test — never .env or production values.
"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


def pytest_configure(config):
    """Load .env.test before anything else runs."""
    env_test = Path(__file__).parent.parent / ".env.test"
    if env_test.exists():
        load_dotenv(env_test, override=True)
    else:
        raise FileNotFoundError(
            ".env.test not found. Create it from .env.example."
        )
    os.environ["APP_ENV"] = "test"


@pytest.fixture(scope="session", autouse=True)
def enforce_test_env():
    """Fail loudly if any test accidentally targets the wrong environment."""
    assert os.getenv("APP_ENV") == "test", (
        "APP_ENV must be 'test' during tests. Check .env.test."
    )
    db_url = os.getenv("DATABASE_URL", "")
    assert "fitscan_test" in db_url or "sqlite" in db_url, (
        f"Tests must use fitscan_test database. Got: {db_url}"
    )
