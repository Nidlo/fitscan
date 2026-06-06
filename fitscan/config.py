"""
Single source of truth for all configuration.
Loaded once at startup. All modules import from here, never from os.environ directly.
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
    # Sprint 36 Stage 1, SMPL fitting is now off by default. The
    # /measure/measurements path uses Claude as the primary measurer;
    # SMPL only fits when /measure/avatar explicitly opts in (the
    # mesh is needed for the 3D try-on render). Flipping the default
    # avoids loading the ~750MB HMR2 checkpoint and saves 30-60s per
    # scan on the typical measurement path.
    enable_smpl_fitting: bool = False
    enable_avatar_generation: bool = True
    confidence_threshold: float = 0.65
    max_image_size_mb: int = 10

    # SaaS
    free_tier_monthly_limit: int = 50
    pro_tier_monthly_limit: int = 1000
    enterprise_tier_monthly_limit: int = 999_999

    # CORS, comma-separated list of allowed origins for the API.
    # Default to the Laravel backend host only. Wildcard `*` is opt-in
    # for development; production must keep a finite allowlist.
    # (FS-NIDLO-MEAS-03)
    cors_origins: str = "http://localhost:8000"

    # Observability, Sentry DSN. When unset, the SDK initialises as a
    # no-op so dev/test environments don't accidentally publish errors.
    # `traces_sample_rate` controls performance-trace sampling.
    # (FS-NIDLO-OPS-01)
    sentry_dsn: str = ""
    sentry_traces_sample_rate: float = 0.1

    # FS-NIDLO-OPS-12, admin shared secret for /api/v1/admin/* and
    # /api/v1/keys/register. Set to a long random hex via
    # `python -m fitscan.cli set-admin-key <KEY>` or by hand.
    fitscan_admin_api_key: str = ""

    # FS-NIDLO-OPS-12, DEV-ONLY loopback bypass. When true, requests
    # from 127.0.0.1 / localhost / ::1 don't need X-Admin-API-Key.
    # Production MUST keep this false.
    fitscan_allow_unauthed_localhost: bool = False

    # Anthropic API key for Claude Vision validation. Read here (rather
    # than os.getenv-only in create_pipeline) because the worker loads
    # `.env` exclusively through pydantic Settings, we don't push values
    # into os.environ. See the matching pattern in `_check_admin_key` and
    # `_api_key_required`. Empty string disables Claude validation (the
    # rule validator still runs).
    anthropic_api_key: str = ""

    # LLM provider backend for every Claude call (measurement, validation,
    # correction, height, landmarks, style). `api` (default) uses the direct
    # Anthropic Messages API; `claude_code` routes through Claude Code via the
    # `claude-agent-sdk` package so the worker can run off a Claude
    # subscription instead of metered API billing. Default `api` keeps the
    # production path unchanged. Env: FITSCAN_LLM_PROVIDER. See fitscan/llm/
    # and docs/claude_code.md.
    fitscan_llm_provider: str = "api"

    # Auth mode for the `claude_code` provider on a headless box:
    #   - `api_key`     → uses ANTHROPIC_API_KEY (API billing)
    #   - `oauth_token` → uses the subscription token below (from
    #                     `claude setup-token`); the provider scrubs
    #                     ANTHROPIC_API_KEY so it can't shadow the token
    #   - `cli_session` → uses an on-disk `claude login` credential
    # Ignored when the provider is not claude_code. Env: FITSCAN_CLAUDE_CODE_AUTH.
    fitscan_claude_code_auth: str = "api_key"

    # Long-lived subscription token for auth mode `oauth_token`. Mint once on
    # a machine with a browser via `claude setup-token`, then set it in .env
    # on the headless worker. Empty otherwise. The field name is prefixed so
    # it does NOT collide with Claude Code's own CLAUDE_CODE_OAUTH_TOKEN env
    # var (the provider sets that for the subprocess itself).
    # Env: FITSCAN_CLAUDE_CODE_OAUTH_TOKEN.
    fitscan_claude_code_oauth_token: str = ""

    # Model for the Claude Code path. Defaults to Opus 4.8 (the callers
    # hardcode Sonnet for the direct-API path; on Claude Code we run Opus for
    # higher measurement quality). Overrides every Claude Code call's model.
    # Env: FITSCAN_CLAUDE_CODE_MODEL.
    fitscan_claude_code_model: str = "claude-opus-4-8"

    # How many times the Claude measurement runs per scan, fused by per-field
    # median. 1 (default) = single call. >1 rejects single-run spikes on
    # sampling models (Opus can't pin temperature=0) at K times the cost. 3 is
    # the recommended stability setting for the Claude Code / Opus path.
    # Env: FITSCAN_MEASURE_ENSEMBLE_SIZE.
    fitscan_measure_ensemble_size: int = 1

    # Optional MCP-servers JSON attached to every Claude Code request. Either
    # the `{"mcpServers": {...}}` envelope or a bare server-name → config map.
    # Empty disables MCP. No MCP improves anthropometric measurement today;
    # this is plumbing for tool/data servers you choose to add.
    # Env: FITSCAN_CLAUDE_CODE_MCP_SERVERS.
    fitscan_claude_code_mcp_servers: str = ""

    # DEV/TEST, turn off the per-tenant X-API-Key gate on /api/v1/*
    # endpoints (measurement, style classification). Production MUST
    # leave this true so callers must present a valid registered key.
    # Read via this Settings field rather than os.getenv-only because
    # the worker loads .env exclusively through pydantic Settings (we
    # don't push values into os.environ, see _check_admin_key for
    # the same rationale).
    fitscan_require_api_key: bool = True

    # Dedicated tempdir for streamed measurement uploads. Living under
    # its own root (rather than the system /tmp) lets us sweep stale
    # files on app startup without risking deleting unrelated tempfiles.
    # Files older than `upload_tempdir_max_age_seconds` get cleaned at
    # boot, covers the "uvicorn was hard-killed mid-request" leak
    # path. (FS-NIDLO-OPS-03)
    upload_tempdir: Path = Path("/tmp/fitscan_uploads")
    upload_tempdir_max_age_seconds: int = 60 * 60  # 1 hour

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse `cors_origins` into the list FastAPI's CORSMiddleware
        wants. Empty values stripped; `*` left intact for dev."""
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
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
