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
