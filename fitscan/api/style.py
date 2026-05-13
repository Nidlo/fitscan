"""Style classification router (POST /api/v1/style/classify).

Mounted by `fitscan.api.main`. Authenticates via the same `verify_api_key`
gate as the measurement endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from fitscan.application.measurement_service import _resolve_anthropic_key
from fitscan.application.style_analyzer import StyleAnalyzer
from fitscan.domain.style_models import StyleProfileRequest, StyleProfileResponse

from .dependencies import verify_api_key

router = APIRouter(prefix="/api/v1/style", tags=["style"])


def _get_style_analyzer() -> StyleAnalyzer:
    """Per-call StyleAnalyzer bound to the .env-resolved Anthropic key.

    The analyzer is cheap to construct (it lazily caches an Anthropic
    client via lru_cache keyed on the API key), so we don't bother
    memoising the analyzer itself.

    Uses `_resolve_anthropic_key()` (same helper as measurement_service)
    so `.env` takes precedence over `os.environ`. Without it, a stale
    `ANTHROPIC_API_KEY=sk-ant-sid…` export in the shell (e.g. from
    Claude Code) would shadow the real deploy key from .env and the
    Public API returns 401 invalid-x-api-key — silently caught at
    style_analyzer.py:134 and turned into a deterministic fallback.
    """
    key = _resolve_anthropic_key()
    return StyleAnalyzer(api_key=key or None)


@router.post(
    "/classify",
    response_model=StyleProfileResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Classify a client's style profile",
)
def classify_style(
    request: StyleProfileRequest,
    analyzer: StyleAnalyzer = Depends(_get_style_analyzer),
) -> StyleProfileResponse:
    return analyzer.analyze(request)
