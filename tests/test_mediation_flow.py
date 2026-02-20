"""End-to-end mediation flow tests using mock LLM responses.

These tests exercise the full pipeline — prompt construction, PII redaction,
schema validation, confidence thresholds, and transcript truncation — without
making any real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from litellm_a2a_settlement.prompts import build_mediator_messages, truncate_transcript
from litellm_a2a_settlement.redaction import PiiRedactor, redact_message_content
from litellm_a2a_settlement.schema import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    validate_response,
)


# ---------------------------------------------------------------------------
# Local helpers (mirrors conftest.py but importable)
# ---------------------------------------------------------------------------

def _make_mediator_response(
    *,
    decision: str = "APPROVED",
    confidence: float = 0.92,
    reasoning: str = "Evidence supports the proposed resolution within risk limits.",
) -> str:
    return json.dumps({
        "decision": decision,
        "confidence_interval": confidence,
        "reasoning_summary": reasoning,
    })


_MOCK_APPROVED = _make_mediator_response()

_MOCK_REJECTED = _make_mediator_response(
    decision="REJECTED",
    confidence=0.88,
    reasoning="Proposed amount exceeds the single-party liability limit.",
)

_MOCK_LOW_CONFIDENCE = _make_mediator_response(
    decision="APPROVED",
    confidence=0.62,
    reasoning="Transcript evidence is ambiguous; both parties present conflicting claims.",
)


class _MockLLMCompletion:
    def __init__(self, content: str, model: str = "gpt-4o") -> None:
        self.choices = [MagicMock()]
        self.choices[0].message.content = content
        self.model = model
        self.usage = MagicMock(total_tokens=150, prompt_tokens=120, completion_tokens=30)


# ---------------------------------------------------------------------------
# Full pipeline: build messages → redact → validate response
# ---------------------------------------------------------------------------

SAMPLE_TRANSCRIPT = """\
[2025-06-01 09:00] Party A (alice@corp.com): We request $45,000 for SLA breach.
[2025-06-01 09:15] Party B (bob@vendor.io): We dispute the claim. Max liability is 60%.
[2025-06-01 10:00] Party A: Attached proof — TX 0xaabbccddee1122334455667788990011aabbccdd.
[2025-06-01 10:30] Party B: Counter-offer: $27,000 settlement.
[2025-06-01 11:00] Party A: Agreed to $27,000 pending mediator approval.
"""


class TestFullMediationPipeline:
    def test_approved_flow(self):
        messages = build_mediator_messages(SAMPLE_TRANSCRIPT)
        redacted_messages, token_map = redact_message_content(messages)

        assert "alice@corp.com" not in redacted_messages[1]["content"]
        assert "bob@vendor.io" not in redacted_messages[1]["content"]
        assert "0xaabbccddee" not in redacted_messages[1]["content"]
        assert len(token_map) >= 3

        result = validate_response(_MOCK_APPROVED, confidence_threshold=0.85)
        assert result["decision"] == "APPROVED"
        assert result["confidence_interval"] == 0.92

    def test_rejected_flow(self):
        messages = build_mediator_messages(SAMPLE_TRANSCRIPT)
        _, token_map = redact_message_content(messages)

        result = validate_response(_MOCK_REJECTED)
        assert result["decision"] == "REJECTED"

    def test_low_confidence_flagged_for_review(self):
        result = validate_response(_MOCK_LOW_CONFIDENCE)
        assert result["decision"] == "NEEDS_REVIEW"
        assert result["confidence_interval"] == 0.62

    def test_low_confidence_passes_with_lower_threshold(self):
        result = validate_response(
            _MOCK_LOW_CONFIDENCE, confidence_threshold=0.50,
        )
        assert result["decision"] == "APPROVED"

    def test_threshold_disabled_with_zero(self):
        result = validate_response(
            _MOCK_LOW_CONFIDENCE, confidence_threshold=0.0,
        )
        assert result["decision"] == "APPROVED"

    def test_boundary_confidence_equals_threshold(self):
        raw = _make_mediator_response(confidence=0.85)
        result = validate_response(raw, confidence_threshold=0.85)
        assert result["decision"] == "APPROVED"

    def test_boundary_confidence_just_below_threshold(self):
        raw = _make_mediator_response(confidence=0.8499)
        result = validate_response(raw, confidence_threshold=0.85)
        assert result["decision"] == "NEEDS_REVIEW"


# ---------------------------------------------------------------------------
# Redaction + hash tokens in pipeline
# ---------------------------------------------------------------------------

class TestRedactionTokensInPipeline:
    def test_redacted_tokens_contain_hashes(self):
        messages = build_mediator_messages(SAMPLE_TRANSCRIPT)
        _, token_map = redact_message_content(messages)

        for token in token_map:
            assert ":" in token, f"Token {token} missing hash separator"
            parts = token.rstrip("]").split(":")
            assert len(parts[-1]) == 6, f"Hash in {token} is not 6 hex chars"

    def test_same_value_produces_same_hash(self):
        r = PiiRedactor()
        result = r.redact("alice@test.com and alice@test.com")
        tokens = [t for t in result.token_map if "EMAIL" in t]
        assert len(tokens) == 1


# ---------------------------------------------------------------------------
# Transcript truncation in pipeline
# ---------------------------------------------------------------------------

class TestTranscriptTruncationInPipeline:
    def test_short_transcript_unchanged(self):
        result = truncate_transcript("Short transcript.", max_tokens=1000)
        assert result == "Short transcript."

    def test_long_transcript_preserves_tail(self):
        lines = [f"[line {i}] Content for line {i}\n" for i in range(500)]
        full = "".join(lines)
        truncated = truncate_transcript(full, max_tokens=200)

        assert "line 499" in truncated
        assert "truncated" in truncated
        assert len(truncated) < len(full)

    def test_truncation_marker_present(self):
        lines = [f"Line {i}: {'x' * 50}\n" for i in range(500)]
        full = "".join(lines)
        truncated = truncate_transcript(full, max_tokens=100)
        assert "earlier transcript truncated" in truncated

    def test_build_messages_applies_truncation(self):
        lines = [f"[{i}] Entry with data {'y' * 80}\n" for i in range(1000)]
        full = "".join(lines)
        messages = build_mediator_messages(full, max_transcript_tokens=500)
        user_content = messages[1]["content"]
        assert "truncated" in user_content
        assert len(user_content) < len(full)

    def test_build_messages_no_truncation_when_short(self):
        messages = build_mediator_messages("Short.")
        assert "truncated" not in messages[1]["content"]


# ---------------------------------------------------------------------------
# MockLLMCompletion validates as expected response object
# ---------------------------------------------------------------------------

class TestMockLLMCompletion:
    def test_mock_has_correct_structure(self):
        mock = _MockLLMCompletion(_MOCK_APPROVED)
        content = mock.choices[0].message.content
        parsed = json.loads(content)
        assert parsed["decision"] == "APPROVED"
        assert 0 <= parsed["confidence_interval"] <= 1

    def test_mock_usage_tracking(self):
        mock = _MockLLMCompletion(_MOCK_APPROVED)
        assert mock.usage.total_tokens == 150
