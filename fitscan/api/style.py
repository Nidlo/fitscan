"""Style classification router (POST /api/v1/style/classify).

Mounted by `fitscan.api.main`. Authenticates via the same `verify_api_key`
gate as the measurement endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from fitscan.application.measurement_service import _build_llm_provider
from fitscan.application.style_analyzer import StyleAnalyzer
from fitscan.domain.style_models import StyleProfileRequest, StyleProfileResponse

from .dependencies import verify_api_key

router = APIRouter(prefix="/api/v1/style", tags=["style"])


def _get_style_analyzer() -> StyleAnalyzer:
    """Per-call StyleAnalyzer bound to the configured LLM provider.

    The analyzer is cheap to construct, so we don't memoise it. Provider
    selection (direct API vs Claude Code) and Anthropic-key resolution both
    live in `_build_llm_provider()` (shared with the measurement pipeline),
    so the style path honours `FITSCAN_LLM_PROVIDER` and the same `.env`
    key-precedence rule that stops a stray Claude Code session token from
    shadowing the deploy key.
    """
    return StyleAnalyzer(llm=_build_llm_provider())


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
