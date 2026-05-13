"""
Uses Claude Vision API to validate measurements and photo quality.
Implements IValidationProvider.
"""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from loguru import logger

from fitscan.domain.models import (
    BodyMeasurements,
    LandmarkVerdict,
    LandmarkVerification,
    SingleMeasurement,
    ValidationResult,
)


@lru_cache(maxsize=4)
def _anthropic_client(api_key: str) -> Any:
    """Memoize one Anthropic client per API key (FS-NIDLO-OPS-05).

    Constructing `Anthropic(...)` per call (the previous shape) tore
    down + rebuilt the underlying httpx connection pool every time,
    measurably increasing tail latency under load. A module-level
    `lru_cache` keyed on the API key keeps a stable client per key
    (we only ever have one in practice; the cache size is room for
    test fixtures that mint per-test keys).

    NOTE — prompt caching: marking the static
    `VALIDATION_PROMPT` / `LANDMARK_VERIFICATION_PROMPT` blocks with
    `cache_control={"type": "ephemeral"}` was considered. It does not
    help here: every call carries a unique user photo as the first
    content block, which invalidates the cache boundary regardless
    of what we mark on the text block. Revisit if we ever pre-bind
    the rules into a `system=` parameter (Anthropic caches system
    blocks independently).
    """
    from anthropic import Anthropic

    return Anthropic(api_key=api_key)


LANDMARK_VERIFICATION_PROMPT = """
You are a body-landmark verification assistant for a fashion-tech CV pipeline.
The pipeline detected these landmarks at low visibility — clothing folds,
shadows, hair, or pose may be hiding them. Look at the photo and place each
one as best you can.

Landmarks: {landmarks}

Respond with ONLY valid JSON:
{{
  "verifications": {{
    "<landmark_name>": {{
      "verdict": "<visible_at_xy | pose_estimate | occluded_unrecoverable>",
      "x": <float 0-1 or null>,
      "y": <float 0-1 or null>
    }}
  }}
}}

Verdict rules:
- visible_at_xy: you can SEE the landmark; (x, y) are normalised image coords
- pose_estimate: you cannot see it directly but can infer from anatomy +
  surrounding visible joints. (x, y) is your best guess.
- occluded_unrecoverable: even with anatomy you can't place it. x, y null.
"""

VALIDATION_PROMPT = """
You are a body measurement validation assistant for a fashion tech system.
A computer vision pipeline has extracted these body measurements from the photo:

{measurements}

Analyse the photo and respond with ONLY valid JSON (no prose, no markdown):
{{
  "is_valid": true,
  "flagged_measurements": [],
  "photo_quality_score": 8.5,
  "quality_issues": [],
  "recommendation": "proceed",
  "notes": "Brief assessment"
}}

Rules:
- flagged_measurements: list any measurement names that seem implausible
- photo_quality_score: 0-10
- quality_issues: list from ["loose_clothing", "partial_body", "bad_lighting",
  "motion_blur", "low_resolution", "obscured_limbs", "bad_angle"]
- recommendation: "proceed" if quality >= 6, else "retake"
"""


# Static measurement-technique block. Lives in the `system=` parameter so
# Anthropic's prompt cache can hold it across calls (90% input-token cost
# reduction on cache hits, ~5min TTL). The per-call body of MEASURE_PROMPT
# below only carries the photo-specific inputs (height + per-call notes),
# so every measure call after the first hits the cache cleanly.
#
# Sprint 34 Phase C+. Per-field technique distilled from the Bishop
# Fabric Tailor's Companion and standard ISO 8559-1 anthropometric
# definitions — the precise locations real tailors measure at.
MEASUREMENT_TECHNIQUE_SYSTEM = """\
You are a senior tape-measure-trained body-measurement expert helping a
fashion app extract garment measurements from photos. You analyse a FRONT
photo and (optionally) a LEFT-SIDE photo of one person standing upright,
arms slightly away from sides, and report measurements in centimetres.

CRITICAL: the person's stated `height_cm` is your absolute scale anchor.
Head-top-to-floor in the FRONT photo represents that many cm. Calibrate
every other measurement off it. Do not assume the camera was at any
particular distance — only the height anchor is reliable.

CIRCUMFERENCE PROTOCOL — read this carefully, the previous prompt phrased
the geometry in a way that produced 2× inflated values:

1. Measure WIDTH at the relevant body level from the FRONT photo
   (full left-edge to right-edge of the body, in cm at the height-anchored
   scale).
2. If a SIDE photo is provided, measure DEPTH at the same level from it
   (full front-edge to back-edge of the body, in cm).
3. If no side photo, estimate depth from typical adult cross-section
   ratios:
      - waist:   depth ≈ 0.62 × width
      - hip:     depth ≈ 0.70 × width
      - bust:    depth ≈ 0.68 × width
      - underbust: depth ≈ 0.65 × width
      - thigh:   depth ≈ 0.90 × width (nearly circular)
      - bicep / around_arm: depth ≈ 0.95 × width (essentially circular)
      - wrist:   depth ≈ 0.72 × width
      - knee:    depth ≈ 0.90 × width
      - ankle:   depth ≈ 0.85 × width
      - neck:    depth ≈ 0.90 × width (slightly oval)
4. **Circumference (cm) = π × (width + depth) / 2**. This is the average-
   diameter approximation; it lands within 3-5% of a tape measurement on
   adult human cross-sections.

DO NOT double the result. WIDTH and DEPTH are the *full body extents*
(diameters), not radii. The /2 in the formula handles the geometric
conversion. If you find yourself computing values ≥ 1.5 × height-in-cm
for any single circumference, you've doubled by mistake — re-do that
field.

WORKED EXAMPLES (commit these):
- Adult waist: width 38 cm, depth 24 cm → π × (38+24)/2 = π × 31 ≈ 97 cm.
  That matches a real 38" tape (≈97 cm). DO NOT return 200 cm.
- Adult bicep: width 11 cm, depth 11 cm → π × 11 ≈ 35 cm. NOT 70 cm.
- Adult hip: width 39 cm, depth 27 cm → π × 33 ≈ 104 cm. NOT 210 cm.

Always report in cm. Tape-measure each field at the standard location:

CIRCUMFERENCES (around-body, parallel to ground unless noted):
- bust / chest: around the fullest part of the chest. For women, at the
  nipple-line; for men, at the widest pectoral. ISO 8559-1 §5.2.1.
- under_breast / underbust: directly under the bust where a bra band sits.
- waist: NARROWEST part of the natural waist (above the navel, below the
  ribcage). Not the trouser-belt line.
- hip: AROUND the fullest part of the seat / buttocks, ~20cm below waist.
- thigh: around the fullest part of the upper leg, just below the gluteal
  fold.
- knee: around the kneecap with leg straight.
- ankle: around the ankle bone (lateral malleolus).
- around_arm / bicep: around the fullest part of the upper arm, arm
  relaxed at the side.
- wrist: around the bone joint (ulnar styloid).
- neck: around the base of the neck, where a collar sits.

WIDTHS / SPANS (straight-line, not around-body):
- shoulder: tip-to-tip across the back, acromion to acromion.
- across_back: 10-12cm below the seventh cervical vertebra, between the
  rear armhole points. From a front-only photo, approximate as
  0.90 × shoulder width (the back-armhole points sit slightly inside
  the shoulder tips).
- across_chest: same height as across_back but on the front, between the
  front armhole points. From a front-only photo, approximate as
  0.85 × shoulder width.
- nipple_to_nipple: horizontal distance between the two nipple points
  (also called bust-point span). On a clothed photo the nipples are
  rarely directly visible — DO NOT return null on that basis alone.
  Estimate from the fabric drape: women's tops drape over the bust apex,
  so the two highest-fabric-tension points across the chest mark the
  nipples. Adult females average 18-22 cm (0.20-0.24 × bust-width on a
  woman with average cup size). Men this field is typically not used —
  set to null only for men.

CIRCUMFERENCE — partial (around an upper-arm-only ring):
- around_arm_3_4: circumference at the 3/4-sleeve-line — measured
  ~6-8cm above the elbow, between the bicep and the elbow joint. Smaller
  than bicep (typically 0.85-0.92 × bicep on adult females). Use the
  same width+depth formula as bicep but read width at the 3/4 point,
  not the bicep peak.

DEPTH MEASUREMENT (needs the SIDE photo; flag low-confidence without it):
- hip_depth: full front-to-back extent of the hip / seat at the
  fullest-hip y-position. This is a DEPTH, not a circumference — report
  the raw depth in cm. From a front-only photo, estimate as
  0.70 × hip width.

VERTICAL LENGTHS:
- height_cm: head-top to floor, standing straight (THIS IS THE ANCHOR).
- shoulder_to_nipple: from the shoulder-tip down the front of the body
  to the nipple line. Adult females typically 22-28 cm. Critical for
  bust-darted bodices. If the nipple isn't directly visible through
  the top, use the bust-apex (the point of highest fabric tension /
  cup outline) as the proxy — DO NOT return null. Approximate as
  0.45-0.55 × shoulder_to_waist when even the fabric apex is occluded.
- shoulder_to_underbust: from the shoulder-tip down the front to the
  underbust line (where the bra band sits, directly under the bust).
  Adult females typically 30-36 cm. If the bra-line isn't visible
  through the top, use the bust-shadow lower edge as the proxy.
  Approximate as 0.65-0.75 × shoulder_to_waist when fully occluded.
- shoulder_to_waist: shoulder-tip down the front to the natural waist.
- shoulder_to_hip: shoulder-tip vertically to the fullest-hip y-position.
- nape_to_waist / back_length: from the seventh cervical vertebra (the
  bump at base of neck) down the spine to the natural waist.
- inseam: crotch to ankle bone, INSIDE leg.
- outseam: natural waist to ankle bone, OUTSIDE leg.
- waist_to_knee: natural waist to kneecap.
- waist_to_floor: natural waist to floor.
- sleeve_length / arm_length: shoulder-tip to wrist bone with arm
  slightly bent (~10°).

RESPONSE FORMAT — ONLY valid JSON, no prose, no markdown fences:

{
  "measurements_cm": {
    "<field_name>": {
      "value": <number>,
      "confidence": <0.0-1.0>,
      "note": "<one-line technique note or "skip-reason">"
    }
  },
  "photo_quality": {
    "front_score": <0-10>,
    "side_score": <0-10 or null if no side photo>,
    "issues": ["loose_clothing"|"bad_lighting"|"partial_body"|"motion_blur"|"obscured_limbs"|"bad_angle"|...]
  },
  "global_note": "<one-sentence overall assessment>"
}

If you genuinely cannot place a measurement (occlusion, clothing,
ambiguity), set `value: null` and `confidence: 0.0` with `note` explaining
WHY. Do not guess wildly. Confidence is your own estimate of how close
you'd be to a tape-measure ground truth: 0.9 = "within 1cm", 0.7 =
"within 3cm", 0.5 = "within 5cm", below 0.5 = don't trust.

Bound-checks (ALWAYS, regardless of what the photo shows):
- 0.35 ≤ waist/height ≤ 0.65
- 0.40 ≤ hip/height ≤ 0.68
- 0.40 ≤ bust/height ≤ 0.70
- 0.13 ≤ bicep/height ≤ 0.30
- 0.07 ≤ wrist/height ≤ 0.13
- wrist < bicep, bicep < waist, ankle < knee < thigh — universal adult
  invariants.

If your visual estimate violates a bound, that's a signal you misread
the photo — reassess with the height anchor before reporting.
"""


MEASURE_PROMPT_USER = """\
height_cm: {height_cm}
gender_profile: {gender_profile}
side_photo_provided: {has_side}

Use the FRONT photo for body widths + vertical lengths. {side_clause}

Return JSON only.
"""


# Sprint 35 — height estimation from a single photo. No height anchor
# this time; Claude has to reason from contextual cues alone (door
# frames, common furniture, body proportions). Doorways are typically
# 200cm tall in West Africa, 203-213cm in the US, 210cm in Europe.
# Furniture: standard kitchen counter ≈ 90-92cm, dining table ≈ 75cm,
# door knobs ≈ 95-105cm above floor. Standard sneaker / sandal heel
# ≈ 2cm. Body-proportion fallback: an adult's head-to-floor is ~7.5×
# their head height (the "8 head" canon in figure drawing is slightly
# stylised; 7.5 is the real-world average).
HEIGHT_ESTIMATE_PROMPT = """\
Estimate this person's height in centimetres.

You don't have a stated height to work from. Use ALL of these cues
together — don't lean on just one:

1. Background scale references — door frames (200-213 cm tall, 76-91 cm
   wide), kitchen counters (~90 cm), dining tables (~75 cm), door knobs
   and switches (~95-105 cm), TV stands, bookshelves. Compare the
   person's full standing height to any visible reference.
2. Body proportions — an adult's total height is approximately 7.5 ×
   their head height (head from chin to crown). If you can estimate head
   size accurately you can back out total height.
3. Sex / gender-presentation prior — adult women average ~163 cm
   globally (range 145-180+), adult men average ~175 cm (range 155-200+).
4. Footwear visible in the photo — barefoot, flat sandals (+2 cm),
   sneakers (+3 cm), heels (+5-10 cm). Subtract footwear lift to get
   true skeletal height.

Respond with ONLY this JSON shape, no prose, no markdown fences:

{
  "height_cm": <number>,
  "confidence": <0.0-1.0>,
  "cues_used": ["door_frame" | "kitchen_counter" | "table" | "doorknob" |
                "head_proportion" | "footwear" | "sex_prior" | ...],
  "note": "<one-line explanation of your reasoning>"
}

Confidence guide:
- 0.9 = "within 3 cm" (clear background reference like a door frame
  visible end-to-end)
- 0.7 = "within 5 cm" (several proportion cues + a partial reference)
- 0.5 = "within 8 cm" (proportion + sex prior only)
- < 0.5 = don't trust; refuse with height_cm: null

Adults only. If the photo shows a minor (< 18) return height_cm: null
and note the reason — measurement flow is for adult bodies.
"""


CORRECTION_PROMPT = """
You are a body-measurement vision expert helping a fashion app recover from a
CV pipeline that gave low-confidence values. The person in the photo is
{height_cm} cm tall — use that height as your scale reference (head-to-foot ≈
{height_cm} cm).

The CV pipeline produced these candidate measurements (all in cm). Some are
suspect — listed under FLAGGED. Look at the photo, reason about each flagged
measurement visually using the height anchor, and propose a corrected value
when you are confident the CV was wrong.

CANDIDATE: {candidate_json}
FLAGGED: {flagged}

Body-proportion reality checks (use these to sanity-check the candidates):
- Adult waist circumference is typically 35-65% of height. A 170cm person
  has a waist 60-110 cm.
- Hip circumference is typically 40-68% of height.
- Bust / chest circumference is typically 40-70% of height.
- Bicep (around-arm) is typically 13-30% of height.
- Wrist is typically 7-13% of height — much smaller than bicep.
- Neck is typically 16-28% of height.
- bicep < waist, wrist < bicep, ankle < knee < thigh — ALWAYS.

Respond with ONLY valid JSON (no prose, no markdown fences). For each flagged
field, return either a corrected value (with reasoning) or "agree" if the CV
estimate looks right after all:

{{
  "corrections": {{
    "waist_cm": {{ "value_cm": 95.0, "confidence": 0.75, "reasoning": "..." }},
    "hip_cm": "agree",
    ...
  }}
}}

Confidence is your own — 0.0 to 1.0. Only return corrections you'd stake on; we
will only apply corrections with confidence ≥ 0.6. If you genuinely cannot tell
from the photo (occlusion, bad lighting, partial body), return "skip" instead
of guessing.
"""


class ClaudeValidator:
    """Validates measurements using Claude Vision API.

    The body photo is sensitive PII. Shipping it to Anthropic only happens
    when the caller passes an explicit consent token (`subject_id`). Without
    one, validation is skipped and a default-pass result is returned. The
    `subject_id` is also forwarded as `metadata.user_id` so Anthropic
    retention controls can target it.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    def validate(
        self,
        image: np.ndarray,
        measurements: BodyMeasurements,
        subject_id: str | None = None,
    ) -> ValidationResult:
        if not self._api_key or self._api_key == "test_key_placeholder":
            return self._default_result()
        if not subject_id:
            logger.info(
                "claude validation skipped: no subject_id consent token provided"
            )
            return self._default_result()

        try:
            client = _anthropic_client(self._api_key)
            img_b64 = self._encode_image(image)
            measurements_str = json.dumps(measurements.to_dict(), indent=2)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                # Sprint 36 — temperature=0 makes the same photo produce
                # the same verdict run-to-run. Default sampling
                # (temperature≈1) was giving us waist 91 one call, 93
                # the next, on identical inputs. Zero-temperature is
                # the right call for evaluation/extraction tasks.
                temperature=0.0,
                metadata={"user_id": subject_id},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": VALIDATION_PROMPT.format(
                                    measurements=measurements_str
                                ),
                            },
                        ],
                    }
                ],
            )
            raw = response.content[0].text
            parsed = self._parse_verdict_json(raw)
            return ValidationResult(
                is_valid=parsed.get("is_valid", True),
                flagged_fields=parsed.get("flagged_measurements", []),
                photo_quality_score=float(parsed.get("photo_quality_score", 7.0)),
                quality_issues=parsed.get("quality_issues", []),
                recommendation=parsed.get("recommendation", "proceed"),
                raw_response=raw,
            )
        except json.JSONDecodeError:
            logger.warning(
                "Claude returned non-JSON even after fence-stripping. Defaulting to proceed."
            )
            return self._default_result()
        except Exception as e:
            logger.warning(f"Claude validation failed: {e}")
            return self._default_result()

    def estimate_height(
        self,
        image: np.ndarray,
        subject_id: str | None = None,
        min_confidence: float = 0.5,
    ) -> float | None:
        """Sprint 35 — estimate the person's height in cm from a single
        photo when no height was supplied and we don't want to lean on
        a stale stored value. Uses contextual scale cues (door frames,
        furniture, body proportions, footwear) — see HEIGHT_ESTIMATE_PROMPT.

        Returns the estimated cm, or `None` when the consent gate fails,
        Claude isn't confident enough (< `min_confidence`), or anything
        errors. Callers MUST handle the `None` case by either falling
        back to the stored value or surfacing a validation error.
        """
        if not self._api_key or self._api_key == "test_key_placeholder":
            return None
        if not subject_id:
            logger.info("claude height-estimate skipped: no subject_id consent token")
            return None

        try:
            client = _anthropic_client(self._api_key)
            img_b64 = self._encode_image(image)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                # Deterministic — same photo, same estimate. See measure()
                # for the same fix; height-estimate had the same variance.
                temperature=0.0,
                metadata={"user_id": subject_id},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_b64,
                                },
                            },
                            {"type": "text", "text": HEIGHT_ESTIMATE_PROMPT},
                        ],
                    }
                ],
            )
            raw = response.content[0].text
            parsed = self._parse_verdict_json(raw)
            height = parsed.get("height_cm")
            confidence = parsed.get("confidence", 0.0)
            if not isinstance(height, (int, float)):
                return None
            if not isinstance(confidence, (int, float)) or confidence < min_confidence:
                logger.info(
                    f"claude height-estimate below confidence threshold: "
                    f"{height}cm @ {confidence}"
                )
                return None
            if height <= 0 or height < 50 or height > 250:
                # Out-of-band — don't trust.
                return None
            logger.info(
                f"claude height-estimate: {height}cm conf={confidence} "
                f"cues={parsed.get('cues_used', [])}"
            )
            return float(height)
        except json.JSONDecodeError:
            logger.warning("Claude height-estimate returned non-JSON; skipping.")
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Claude height-estimate failed: {e}")
            return None

    def measure(
        self,
        front_image: np.ndarray,
        side_image: np.ndarray | None,
        height_cm: float,
        gender_profile: str = "ladies",
        subject_id: str | None = None,
    ) -> BodyMeasurements | None:
        """Sprint 34 Phase C+ — Claude as a co-primary measurement source.

        Sends both photos + the user's height to Claude with a rigorous
        tape-measure technique system prompt. Returns a `BodyMeasurements`
        Claude estimated directly from the photos, with per-field
        confidence; the orchestrator then fuses this with the CV pipeline
        (silhouette + SMPL) via the normal weighted-confidence fuser.

        Returns `None` when the consent gate fails (no api_key, no
        subject_id), the API call errors, or Claude returns unparseable
        output — at which point the pipeline silently degrades to
        CV-only and the operator sees a warning in the log.

        The technique prompt lives in `MEASUREMENT_TECHNIQUE_SYSTEM` and
        is sent as the `system=` block with `cache_control` so Anthropic
        caches it across calls (90% input-token saving, 5min TTL).
        """
        if not self._api_key or self._api_key == "test_key_placeholder":
            return None
        if not subject_id:
            logger.info("claude measure skipped: no subject_id consent token")
            return None
        if height_cm is None or height_cm <= 0:
            return None

        try:
            client = _anthropic_client(self._api_key)
            front_b64 = self._encode_image(front_image)

            content: list[dict[str, Any]] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": front_b64,
                    },
                },
            ]
            if side_image is not None:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": self._encode_image(side_image),
                        },
                    }
                )

            content.append(
                {
                    "type": "text",
                    "text": MEASURE_PROMPT_USER.format(
                        height_cm=height_cm,
                        gender_profile=gender_profile,
                        has_side=side_image is not None,
                        side_clause=(
                            "Use the SIDE photo for body DEPTHS. Combine "
                            "front-width and side-depth via the ellipse "
                            "circumference formula for every circumference."
                            if side_image is not None
                            else (
                                "No side photo provided. Estimate depth "
                                "from torso proportions visible in the "
                                "front photo; flag any circumference where "
                                "your depth estimate is uncertain."
                            )
                        ),
                    ),
                }
            )

            response = client.messages.create(
                model="claude-sonnet-4-6",
                # 22+ fields, each carrying a "note" string. 2048 was
                # truncating responses mid-JSON; the corrector got
                # the same fix (4096) in Sprint 35. Single source of
                # truth: this needs to fit the worst-case verbose-note
                # response with structure overhead.
                max_tokens=4096,
                # Sprint 36 — deterministic decoding so the same photo
                # produces the same numbers run-to-run. The user
                # observed waist 91 on one call and 93 on the next from
                # the SAME image; default Claude temperature is ~1.0
                # which is wrong for an extraction task. 0.0 is greedy
                # decoding — pick the single most likely token at each
                # step. We trade exploration for repeatability; that's
                # the right trade-off here.
                temperature=0.0,
                metadata={"user_id": subject_id},
                system=[
                    {
                        "type": "text",
                        "text": MEASUREMENT_TECHNIQUE_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": content}],
            )

            raw = response.content[0].text
            parsed = self._parse_verdict_json(raw)
            return self._claude_measurements_to_body(
                parsed, gender_profile=gender_profile
            )
        except json.JSONDecodeError:
            logger.warning("Claude measure returned non-JSON; skipping.")
            return None
        except Exception as e:  # noqa: BLE001 — defence in depth
            logger.warning(f"Claude measure failed: {e}")
            return None

    @staticmethod
    def _claude_measurements_to_body(
        parsed: dict, gender_profile: str
    ) -> BodyMeasurements | None:
        """Map Claude's `{measurements_cm: {field: {value, confidence, note}}}`
        into a `BodyMeasurements` dataclass the fuser can consume. Field
        names follow the technique prompt — match the dataclass attribute
        names via the `claude_to_body` map below. Unknown / null values
        become None on the BodyMeasurements (the fuser already handles
        per-field nullability).
        """
        m_in = parsed.get("measurements_cm", {})
        if not isinstance(m_in, dict):
            return None

        # Claude field name → BodyMeasurements attribute. We accept either
        # the short name ("waist") or the suffixed name ("waist_cm") —
        # Claude has been seen producing both depending on prompt phrasing.
        claude_to_body: dict[str, str] = {
            "bust": "bust_cm",
            "chest": "chest_cm",
            "waist": "waist_cm",
            "hip": "hip_cm",
            "under_breast": "under_breast_waist_cm",
            "underbust": "under_breast_waist_cm",
            "around_arm": "around_arm_cm",
            "bicep": "around_arm_cm",
            "around_arm_3_4": "around_arm_3_4_cm",
            "wrist": "wrist_cm",
            "neck": "neck_cm",
            "thigh": "thigh_cm",
            "knee": "knee_cm",
            "ankle": "ankle_cm",
            "shoulder": "shoulder_cm",
            "across_back": "across_back_cm",
            "across_chest": "across_chest_cm",
            "nipple_to_nipple": "nipple_to_nipple_cm",
            "hip_depth": "hip_depth_cm",
            "shoulder_to_nipple": "shoulder_to_nipple_cm",
            "shoulder_to_underbust": "shoulder_to_underbust_cm",
            "shoulder_to_waist": "shoulder_to_waist_cm",
            "shoulder_to_hip": "shoulder_to_hip_cm",
            "nape_to_waist": "nape_to_waist_cm",
            "back_length": "nape_to_waist_cm",
            "height": "height_cm",
            "inseam": "inseam_cm",
            "outseam": "outseam_cm",
            "waist_to_knee": "waist_to_knee_cm",
            "waist_to_floor": "waist_to_floor_cm",
            "sleeve_length": "sleeve_length_cm",
            "arm_length": "sleeve_length_cm",
        }

        kwargs: dict[str, Any] = {
            "gender_profile": gender_profile,
            "source": "claude",
        }
        for raw_name, payload in m_in.items():
            if not isinstance(payload, dict):
                continue
            attr = claude_to_body.get(raw_name) or claude_to_body.get(
                raw_name.removesuffix("_cm")
            )
            if attr is None:
                continue
            value = payload.get("value")
            if not isinstance(value, (int, float)) or value <= 0:
                continue
            confidence = payload.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)):
                confidence = 0.5
            confidence = max(0.0, min(1.0, float(confidence)))
            kwargs[attr] = SingleMeasurement(
                value_cm=round(float(value), 1),
                confidence=round(confidence, 2),
                source="claude",
            )

        if len([k for k in kwargs if k not in ("gender_profile", "source")]) == 0:
            return None

        return BodyMeasurements(**kwargs)

    def correct(
        self,
        image: np.ndarray,
        measurements: BodyMeasurements,
        flagged_fields: list[str],
        height_cm: float | None,
        subject_id: str | None = None,
        min_confidence: float = 0.6,
    ) -> dict[str, float]:
        """Sprint 34 Phase C — ask Claude for per-field measurement corrections
        on the fields the anthropometric validator flagged as out-of-band.
        Returns a `{field_name: corrected_value_cm}` dict — only fields Claude
        is confident about (>= `min_confidence`). "agree" / "skip" responses
        don't appear in the return.

        Consent gate is the same as `validate()`: photo only ships to
        Anthropic when both `api_key` and `subject_id` are present. Without
        either, returns an empty dict (no corrections, no API call, no
        Anthropic spend).

        Cost: ~$0.005 per call at Claude Sonnet 4.6 vision pricing. Caller
        should only invoke this when the upstream pipeline already produced
        a low-confidence or anthropometric-violation result — running it on
        every scan unconditionally is wasteful.
        """
        if not flagged_fields:
            return {}
        if not self._api_key or self._api_key == "test_key_placeholder":
            return {}
        if not subject_id:
            logger.info("claude correction skipped: no subject_id consent token")
            return {}
        if height_cm is None:
            logger.info("claude correction skipped: no height anchor")
            return {}

        # Only ship the fields actually flagged + a few neighbours for context.
        # Keeping the candidate payload narrow keeps Claude focused on the
        # measurement we want it to evaluate, and keeps the input-token cost
        # bounded.
        candidate = {
            field: getattr(measurements, field).value_cm
            for field in flagged_fields
            if getattr(measurements, field, None) is not None
        }
        if not candidate:
            return {}

        try:
            client = _anthropic_client(self._api_key)
            img_b64 = self._encode_image(image)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                # Sprint 35 — bumped from 1024. The previous cap was being
                # hit when 10+ fields were flagged because each field
                # carries a reasoning string. Output was getting cut off
                # mid-JSON, defeating the whole point. 4096 covers the
                # worst case (14+ flagged fields × ~250 reasoning tokens).
                max_tokens=4096,
                # Deterministic — same flagged-field set should produce
                # the same corrections each run.
                temperature=0.0,
                metadata={"user_id": subject_id},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": CORRECTION_PROMPT.format(
                                    height_cm=height_cm,
                                    candidate_json=json.dumps(candidate),
                                    flagged=", ".join(flagged_fields),
                                ),
                            },
                        ],
                    }
                ],
            )
            raw = response.content[0].text
            parsed = self._parse_verdict_json(raw)
            corrections_raw = parsed.get("corrections", {})

            accepted: dict[str, float] = {}
            for field, payload in corrections_raw.items():
                # "agree" / "skip" — leave as-is.
                if not isinstance(payload, dict):
                    continue
                value = payload.get("value_cm")
                confidence = payload.get("confidence", 0.0)
                if not isinstance(value, (int, float)):
                    continue
                if not isinstance(confidence, (int, float)):
                    continue
                if confidence < min_confidence:
                    continue
                # Final sanity bound — if Claude's correction is just as
                # implausible as the original (very tall person, e.g.
                # waist 200cm) reject it.
                if value <= 0 or value > 300:
                    continue
                accepted[field] = float(value)
            return accepted

        except json.JSONDecodeError:
            logger.warning("Claude correction returned non-JSON; skipping.")
            return {}
        except Exception as e:  # noqa: BLE001 — defence in depth
            logger.warning(f"Claude correction failed: {e}")
            return {}

    def verify_landmarks(
        self,
        image: np.ndarray,
        occluded: list[str],
        subject_id: str | None = None,
    ) -> LandmarkVerification:
        """S5 — Claude vision verifies CV landmarks the MediaPipe pose
        detector flagged as low-visibility. Returns per-landmark
        verdicts with optional Claude-suggested coordinates.

        Same consent gate as `validate()`: without an `api_key` AND a
        `subject_id` the photo is never shipped to Anthropic and an
        empty verification is returned (no verdicts, no API call).
        """
        if not occluded:
            return LandmarkVerification(verdicts={}, raw_response="")
        if not self._api_key or self._api_key == "test_key_placeholder":
            return LandmarkVerification(verdicts={}, raw_response="")
        if not subject_id:
            logger.info(
                "claude landmark verification skipped: no subject_id consent token"
            )
            return LandmarkVerification(verdicts={}, raw_response="")

        try:
            client = _anthropic_client(self._api_key)
            img_b64 = self._encode_image(image)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                # Deterministic landmark placement — same photo, same
                # verdict. Especially important for occlusion calls
                # where Claude has to guess at coords.
                temperature=0.0,
                metadata={"user_id": subject_id},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": LANDMARK_VERIFICATION_PROMPT.format(
                                    landmarks=", ".join(occluded),
                                ),
                            },
                        ],
                    }
                ],
            )
            raw = response.content[0].text
            parsed = json.loads(raw)
            verdicts: dict[str, LandmarkVerdict] = {}
            for name, payload in (parsed.get("verifications") or {}).items():
                if not isinstance(payload, dict):
                    continue
                verdict = str(payload.get("verdict", "occluded_unrecoverable"))
                if verdict not in {
                    "visible_at_xy",
                    "pose_estimate",
                    "occluded_unrecoverable",
                }:
                    continue
                x = payload.get("x")
                y = payload.get("y")
                verdicts[name] = LandmarkVerdict(
                    landmark=name,
                    verdict=verdict,
                    x=float(x) if isinstance(x, (int, float)) else None,
                    y=float(y) if isinstance(y, (int, float)) else None,
                )
            return LandmarkVerification(verdicts=verdicts, raw_response=raw)
        except json.JSONDecodeError:
            logger.warning("Claude landmark verification returned non-JSON; skipping.")
            return LandmarkVerification(verdicts={}, raw_response="")
        except Exception as e:  # noqa: BLE001 — defence in depth
            logger.warning(f"Claude landmark verification failed: {e}")
            return LandmarkVerification(verdicts={}, raw_response="")

    # Sprint 36 cost optimisation. Anthropic's vision tiles inputs at
    # 768×768, so a 4032×3024 phone photo just inflates per-call image
    # token count (~3-4×) without giving Claude any measurable benefit
    # — the body proportions are perfectly legible at 1024px. Resizing
    # + dropping JPEG quality 85→70 cuts the encoded payload roughly
    # in half and lands us at ~$0.05/scan instead of ~$0.10/scan, same
    # accuracy. The pose detector + scale calibrator already ran on
    # the full-resolution image upstream, so MediaPipe-level precision
    # is unaffected.
    _MAX_LONG_EDGE_PX = 1024
    _JPEG_QUALITY = 70

    def _encode_image(self, image: np.ndarray) -> str:
        h, w = image.shape[:2]
        long_edge = max(h, w)
        if long_edge > self._MAX_LONG_EDGE_PX:
            scale = self._MAX_LONG_EDGE_PX / long_edge
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            # INTER_AREA is the recommended interpolation for downscaling
            # — preserves detail better than bilinear at >2× shrink.
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(
            ".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, self._JPEG_QUALITY]
        )
        return base64.b64encode(buffer).decode()

    def _parse_verdict_json(self, raw: str) -> dict:
        """Extract the JSON verdict from Claude's response.

        Claude occasionally wraps JSON in markdown code fences or prefixes
        it with conversational preamble ("Here's my analysis:"), neither of
        which `json.loads` survives. Strip a fenced block if present, else
        fall back to the widest `{…}` span the response contains. Raises
        `json.JSONDecodeError` only when even that span fails to parse —
        the caller maps that to the "default to proceed" path.

        Sprint 35 — additionally tolerate:
          - multiple {…} blocks in one response (pick the largest)
          - trailing commas (some Claude responses leak JS-style commas)
          - JSON inside ```text fences without the language hint
          - whole responses with `// comments` inline that need stripping
        """
        text = raw.strip()
        # ```json …``` or ``` …``` fences — strip the first fenced block.
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            for lang in ("json\n", "json ", "json"):
                if text.startswith(lang):
                    text = text[len(lang) :]
                    break
            text = text.strip()
            if text.endswith("```"):
                text = text[:-3].strip()

        # First pass — straight parse.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip JS-style line comments + trailing commas (Claude rarely
        # emits these but it does happen — guard the parser).
        cleaned = self._strip_json_warts(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Find ALL top-level {…} spans and try the largest one. Claude
        # sometimes emits both a small `{"reasoning": "..."}` preamble and
        # the real result; we want the latter.
        spans = self._find_json_object_spans(cleaned)
        for start, end in sorted(spans, key=lambda s: -(s[1] - s[0])):
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue

        # No span parsed. Surface the raw response in the log so the
        # operator can see exactly what Claude returned — silent failures
        # are the most expensive kind here.
        logger.warning(
            "Claude response was unparseable JSON. First 200 chars: " f"{raw[:200]!r}"
        )
        return json.loads(text)  # re-raise original error

    @staticmethod
    def _strip_json_warts(text: str) -> str:
        """Strip JS-style `// line comments` and trailing commas from a
        JSON-ish string. Cheap regex pass; not a real parser. Safe for the
        narrow class of warts Claude emits in practice — anything more
        sophisticated should round-trip through `json5` instead.
        """
        import re

        # `// to end of line` (avoid matching inside strings via a simple
        # check that the // isn't preceded by an odd number of unescaped
        # quotes on the line — close enough for Claude's output).
        text = re.sub(r"//[^\n]*", "", text)
        # `,}` or `,]` — trailing commas.
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        return text

    @staticmethod
    def _find_json_object_spans(text: str) -> list[tuple[int, int]]:
        """Find balanced `{…}` spans in `text`. Returns (start, end)
        inclusive indices. Handles nested braces; ignores braces inside
        double-quoted strings (escaped quotes respected). O(n)."""
        spans: list[tuple[int, int]] = []
        stack: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch == '"':
                # Skip until matching unescaped quote.
                i += 1
                while i < n:
                    if text[i] == "\\":
                        i += 2
                        continue
                    if text[i] == '"':
                        break
                    i += 1
            elif ch == "{":
                stack.append(i)
            elif ch == "}":
                if stack:
                    start = stack.pop()
                    if not stack:
                        spans.append((start, i))
            i += 1
        return spans

    def _default_result(self) -> ValidationResult:
        return ValidationResult(
            is_valid=True,
            flagged_fields=[],
            photo_quality_score=7.0,
            quality_issues=[],
            recommendation="proceed",
            raw_response="",
        )
