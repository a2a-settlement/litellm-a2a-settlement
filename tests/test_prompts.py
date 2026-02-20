"""Tests for the Mediator system prompt templates."""

from __future__ import annotations

import json

from litellm_a2a_settlement.prompts import (
    DEFAULT_MAX_TRANSCRIPT_TOKENS,
    DEFAULT_RISK_LIMITS,
    build_mediator_messages,
    render_system_prompt,
    truncate_transcript,
)
from litellm_a2a_settlement.schema import MEDIATOR_RESPONSE_SCHEMA


class TestRenderSystemPrompt:
    def test_contains_schema(self):
        prompt = render_system_prompt()
        assert '"decision"' in prompt
        assert '"confidence_interval"' in prompt
        assert '"reasoning_summary"' in prompt

    def test_contains_default_risk_limits(self):
        prompt = render_system_prompt()
        assert "max_settlement_amount_usd" in prompt
        assert "100000" in prompt

    def test_custom_risk_limits(self):
        limits = {"max_amount": 5000, "currency": "EUR"}
        prompt = render_system_prompt(risk_limits=limits)
        assert '"max_amount": 5000' in prompt
        assert '"currency": "EUR"' in prompt
        assert "max_settlement_amount_usd" not in prompt

    def test_instructs_json_only(self):
        prompt = render_system_prompt()
        assert "Structured Output Only" in prompt
        assert "No prose" in prompt

    def test_instructs_no_pii(self):
        prompt = render_system_prompt()
        assert "REDACTED" in prompt
        assert "de-anonymise" in prompt

    def test_mediator_persona(self):
        prompt = render_system_prompt()
        assert "impartial AI Mediator" in prompt
        assert "financial dispute" in prompt


class TestBuildMediatorMessages:
    def test_returns_system_and_user(self):
        msgs = build_mediator_messages("Transcript here.")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_message_contains_transcript(self):
        msgs = build_mediator_messages("Party A proposed $10,000.")
        assert "Party A proposed $10,000." in msgs[1]["content"]

    def test_user_message_has_evaluation_instruction(self):
        msgs = build_mediator_messages("Some transcript.")
        assert "Evaluate" in msgs[1]["content"]
        assert "JSON verdict" in msgs[1]["content"]

    def test_system_prompt_embeds_schema(self):
        msgs = build_mediator_messages("T")
        system = msgs[0]["content"]
        assert "APPROVED" in system
        assert "REJECTED" in system

    def test_custom_limits_flow_through(self):
        msgs = build_mediator_messages("T", risk_limits={"cap": 999})
        assert '"cap": 999' in msgs[0]["content"]


class TestDefaultRiskLimits:
    def test_is_serialisable(self):
        dumped = json.dumps(DEFAULT_RISK_LIMITS)
        assert json.loads(dumped) is not None

    def test_has_required_keys(self):
        assert "max_settlement_amount_usd" in DEFAULT_RISK_LIMITS
        assert "allowed_dispute_categories" in DEFAULT_RISK_LIMITS


class TestTruncateTranscript:
    def test_short_transcript_unchanged(self):
        text = "Party A: I agree."
        assert truncate_transcript(text, max_tokens=1000) == text

    def test_preserves_most_recent_lines(self):
        lines = [f"[line {i}] data\n" for i in range(200)]
        full = "".join(lines)
        result = truncate_transcript(full, max_tokens=50)
        assert "line 199" in result
        assert "line 0" not in result

    def test_truncation_marker_included(self):
        lines = [f"[line {i}] {'x' * 40}\n" for i in range(200)]
        full = "".join(lines)
        result = truncate_transcript(full, max_tokens=50)
        assert "earlier transcript truncated" in result

    def test_default_budget_constant(self):
        assert DEFAULT_MAX_TRANSCRIPT_TOKENS == 12_000

    def test_build_messages_passes_max_tokens(self):
        lines = [f"[{i}] content {'y' * 100}\n" for i in range(500)]
        full = "".join(lines)
        msgs = build_mediator_messages(full, max_transcript_tokens=100)
        assert "truncated" in msgs[1]["content"]
