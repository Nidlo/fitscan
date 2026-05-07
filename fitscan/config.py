"""
Single source of truth for all configuration.
Loaded once at startup. All modules import from here — never from os.environ directly.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Reads from environment variables. .env file loaded externally."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API
    anthropic_api_key: str = ""
    app_env: str = "development"
    log_level: str = "INFO"
    api_port: int = 8000

    # Database
    database_url: str = "sqlite:///./fitscan.db"

    # Storage
    storage_backend: str = "local"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "fitscan-avatars"
    s3_bucket: str = "fitscan-avatars"
    s3_region: str = "us-east-1"

    # Queue
    celery_broker_url: str = "redis://localhost:6379/0"

    # Models
    smpl_model_path: Path = Path("data/models/smpl/SMPL_NEUTRAL.pkl")
    hmr2_checkpoint_path: Path = Path("data/models/hmr2/epoch=35-step=1000000.ckpt")

    # Feature flags
    enable_claude_validation: bool = True
    enable_smpl_fitting: bool = True
    enable_avatar_generation: bool = True
    confidence_threshold: float = 0.65
    max_image_size_mb: int = 10

    # SaaS
    free_tier_monthly_limit: int = 50
    pro_tier_monthly_limit: int = 1000
    enterprise_tier_monthly_limit: int = 999_999

    # CORS — comma-separated list of allowed origins for the API.
    # Default to the Laravel backend host only. Wildcard `*` is opt-in
    # for development; production must keep a finite allowlist.
    # (FS-NIDLO-MEAS-03)
    cors_origins: str = "http://localhost:8000"

    # Observability — Sentry DSN. When unset, the SDK initialises as a
    # no-op so dev/test environments don't accidentally publish errors.
    # `traces_sample_rate` controls performance-trace sampling.
    # (FS-NIDLO-OPS-01)
    sentry_dsn: str = ""
    sentry_traces_sample_rate: float = 0.1

    # Dedicated tempdir for streamed measurement uploads. Living under
    # its own root (rather than the system /tmp) lets us sweep stale
    # files on app startup without risking deleting unrelated tempfiles.
    # Files older than `upload_tempdir_max_age_seconds` get cleaned at
    # boot — covers the "uvicorn was hard-killed mid-request" leak
    # path. (FS-NIDLO-OPS-03)
    upload_tempdir: Path = Path("/tmp/fitscan_uploads")
    upload_tempdir_max_age_seconds: int = 60 * 60  # 1 hour

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse `cors_origins` into the list FastAPI's CORSMiddleware
        wants. Empty values stripped; `*` left intact for dev."""
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton. Import and call this everywhere. DRY."""
    return Settings()
