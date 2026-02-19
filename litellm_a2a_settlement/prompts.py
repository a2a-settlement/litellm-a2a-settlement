"""System prompt templates for the AI Mediator persona.

The mediator evaluates financial-dispute *Negotiation Transcripts* against
configurable risk limits and returns a structured JSON verdict.  All prompt
content deliberately avoids exposing raw PII — the redaction layer strips
sensitive values before the prompt is assembled.

Usage::

    from litellm_a2a_settlement.prompts import build_mediator_messages

    messages = build_mediator_messages(
        negotiation_transcript="...",
        risk_limits={"max_settlement_amount": 50_000, ...},
    )
"""

from __future__ import annotations

import json
from typing import Any

from .schema import MEDIATOR_RESPONSE_SCHEMA

# ---------------------------------------------------------------------------
# Core system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an impartial AI Mediator employed by a regulated financial dispute \
resolution platform.  Your sole function is to evaluate a Negotiation \
Transcript and produce a binding settlement recommendation.

## Operating Rules

1. **Objectivity** — Base your decision exclusively on the evidence present \
in the Negotiation Transcript.  Do not speculate or introduce external facts.
2. **Risk Limit Compliance** — The organisation enforces the risk limits \
supplied below.  If the disputed amount or proposed resolution would breach \
any limit, you MUST reject the settlement.
3. **No PII Handling** — All personally identifiable information has been \
redacted and replaced with generic tokens (e.g. [REDACTED_EMAIL_1]).  Do \
NOT attempt to de-anonymise, guess, or reconstruct original values.
4. **Confidentiality** — Do not reproduce, quote, or reference the raw \
transcript verbatim in your output.
5. **Structured Output Only** — Respond with a single JSON object that \
conforms exactly to the schema below.  No prose, no markdown fences, no \
additional keys.

## Required Response Schema

```json
{schema}
```

### Field Definitions

- **decision** — ``"APPROVED"`` if the proposed resolution is within risk \
limits and adequately supported by the transcript evidence; ``"REJECTED"`` \
otherwise.
- **confidence_interval** — Your confidence in the decision expressed as a \
float between 0.0 (no confidence) and 1.0 (absolute certainty).
- **reasoning_summary** — A concise (≤ 300 word) explanation of the factors \
that led to your decision.  Reference risk limits and transcript evidence by \
section or party label, never by raw PII.

## Risk Limits in Effect

```json
{risk_limits}
```

Evaluate the Negotiation Transcript provided in the user message against \
these limits and respond with the JSON object described above.\
"""

# ---------------------------------------------------------------------------
# Default risk limits (callers should override for production)
# ---------------------------------------------------------------------------

DEFAULT_RISK_LIMITS: dict[str, Any] = {
    "max_settlement_amount_usd": 100_000,
    "max_single_party_liability_pct": 80,
    "min_confidence_threshold": 0.70,
    "allowed_dispute_categories": [
        "payment_discrepancy",
        "service_level_breach",
        "unauthorised_transaction",
        "contractual_dispute",
    ],
    "require_both_parties_represented": True,
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def render_system_prompt(
    risk_limits: dict[str, Any] | None = None,
) -> str:
    """Render the full Mediator system prompt with embedded risk limits.

    Parameters
    ----------
    risk_limits:
        Risk-limit dictionary.  Falls back to ``DEFAULT_RISK_LIMITS`` when
        ``None``.
    """
    limits = risk_limits or DEFAULT_RISK_LIMITS
    return _SYSTEM_PROMPT.format(
        schema=json.dumps(MEDIATOR_RESPONSE_SCHEMA, indent=2),
        risk_limits=json.dumps(limits, indent=2),
    )


def build_mediator_messages(
    negotiation_transcript: str,
    *,
    risk_limits: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build a complete ``messages`` list ready for a LiteLLM ``completion`` call.

    Returns
    -------
    list[dict]
        ``[{role: "system", content: ...}, {role: "user", content: ...}]``
    """
    system = render_system_prompt(risk_limits)
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "## Negotiation Transcript\n\n"
                f"{negotiation_transcript}\n\n"
                "Evaluate the above transcript and return your JSON verdict."
            ),
        },
    ]
