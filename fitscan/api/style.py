"""Style classification router (POST /api/v1/style/classify).

Mounted by `fitscan.api.main`. Authenticates via the same `verify_api_key`
gate as the measurement endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from fitscan.application.style_analyzer import StyleAnalyzer
from fitscan.config import get_settings
from fitscan.domain.style_models import StyleProfileRequest, StyleProfileResponse

from .dependencies import verify_api_key

router = APIRouter(prefix="/api/v1/style", tags=["style"])


def _get_style_analyzer() -> StyleAnalyzer:
    """Per-call StyleAnalyzer bound to the configured Anthropic key.

    The analyzer is cheap to construct (it lazily caches an Anthropic
    client via lru_cache keyed on the API key), so we don't bother
    memoising the analyzer itself.
    """
    settings = get_settings()
    return StyleAnalyzer(api_key=settings.anthropic_api_key or None)


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
