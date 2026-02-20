"""JSON-schema enforcement for the Mediator response.

Defines the canonical response schema and provides helpers to inject
response_format into a LiteLLM call and validate responses post-hoc.

The ``validate_response`` function supports an optional *confidence_threshold*
parameter (default **0.85**).  When the LLM's ``confidence_interval`` falls
below this threshold the validated payload's ``decision`` is automatically
upgraded to ``"NEEDS_REVIEW"`` so downstream consumers can route the
settlement to a human reviewer or secondary validation agent.
"""

from __future__ import annotations

import json
from typing import Any

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.85

MEDIATOR_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["APPROVED", "REJECTED"],
            "description": (
                "Settlement recommendation: APPROVED if the proposed "
                "resolution is within risk limits, REJECTED otherwise."
            ),
        },
        "confidence_interval": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence in the decision as a float in [0.0, 1.0].",
        },
        "reasoning_summary": {
            "type": "string",
            "maxLength": 2000,
            "description": (
                "Concise explanation of the factors that "
                "led to the decision, 300 words or fewer."
            ),
        },
    },
    "required": ["decision", "confidence_interval", "reasoning_summary"],
    "additionalProperties": False,
}

RESPONSE_FORMAT_PARAM: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "mediator_verdict",
        "strict": True,
        "schema": MEDIATOR_RESPONSE_SCHEMA,
    },
}


def inject_response_format(data: dict[str, Any]) -> dict[str, Any]:
    """Set response_format on a LiteLLM call payload to force JSON mode.

    If response_format is already present it is NOT overwritten so callers
    can opt-in to a custom schema by setting it beforehand.
    """
    if "response_format" not in data:
        data["response_format"] = RESPONSE_FORMAT_PARAM
    return data


class MediatorResponseError(ValueError):
    """Raised when the model response does not match the expected schema."""


def validate_response(
    raw: str,
    *,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """Parse and validate a raw JSON string against the Mediator schema.

    Parameters
    ----------
    raw:
        The raw JSON string from the LLM.
    confidence_threshold:
        If provided, any response whose ``confidence_interval`` is strictly
        below this value will have its ``decision`` set to ``"NEEDS_REVIEW"``
        regardless of the LLM's original verdict.  Pass ``None`` to use
        :data:`DEFAULT_CONFIDENCE_THRESHOLD` (0.85).  Pass ``0.0`` to
        disable the check entirely.

    Returns the parsed dict on success; raises MediatorResponseError on
    any violation.  The returned dict may contain ``decision == "NEEDS_REVIEW"``
    which is *not* part of the LLM schema but is injected post-validation.
    """
    threshold = (
        confidence_threshold if confidence_threshold is not None
        else DEFAULT_CONFIDENCE_THRESHOLD
    )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MediatorResponseError(f"Response is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise MediatorResponseError(
            f"Expected a JSON object, got {type(payload).__name__}"
        )

    missing = {"decision", "confidence_interval", "reasoning_summary"} - payload.keys()
    if missing:
        raise MediatorResponseError(f"Missing required fields: {missing}")

    decision = payload["decision"]
    if decision not in ("APPROVED", "REJECTED"):
        raise MediatorResponseError(
            f"decision must be APPROVED or REJECTED, got {decision!r}"
        )

    ci = payload["confidence_interval"]
    if not isinstance(ci, (int, float)):
        raise MediatorResponseError(
            f"confidence_interval must be a number, got {type(ci).__name__}"
        )
    if not (0.0 <= ci <= 1.0):
        raise MediatorResponseError(
            f"confidence_interval must be in [0.0, 1.0], got {ci}"
        )

    reasoning = payload["reasoning_summary"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise MediatorResponseError("reasoning_summary must be a non-empty string")

    extra = set(payload.keys()) - {"decision", "confidence_interval", "reasoning_summary"}
    if extra:
        raise MediatorResponseError(f"Unexpected extra fields: {extra}")

    if ci < threshold:
        payload["decision"] = "NEEDS_REVIEW"

    return payload
