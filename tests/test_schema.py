"""Tests for the Mediator response schema and validation."""

from __future__ import annotations

import json

import pytest

from litellm_a2a_settlement.schema import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MEDIATOR_RESPONSE_SCHEMA,
    RESPONSE_FORMAT_PARAM,
    MediatorResponseError,
    inject_response_format,
    validate_response,
)


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

class TestSchemaConstants:
    def test_schema_has_required_fields(self):
        assert set(MEDIATOR_RESPONSE_SCHEMA["required"]) == {
            "decision",
            "confidence_interval",
            "reasoning_summary",
        }

    def test_schema_disallows_additional_properties(self):
        assert MEDIATOR_RESPONSE_SCHEMA["additionalProperties"] is False

    def test_decision_enum_values(self):
        props = MEDIATOR_RESPONSE_SCHEMA["properties"]
        assert props["decision"]["enum"] == ["APPROVED", "REJECTED"]

    def test_response_format_param_structure(self):
        assert RESPONSE_FORMAT_PARAM["type"] == "json_schema"
        assert RESPONSE_FORMAT_PARAM["json_schema"]["strict"] is True
        assert RESPONSE_FORMAT_PARAM["json_schema"]["name"] == "mediator_verdict"


# ---------------------------------------------------------------------------
# inject_response_format
# ---------------------------------------------------------------------------

class TestInjectResponseFormat:
    def test_adds_response_format(self):
        data: dict = {"model": "gpt-4o", "messages": []}
        inject_response_format(data)
        assert data["response_format"] == RESPONSE_FORMAT_PARAM

    def test_does_not_overwrite_existing(self):
        custom = {"type": "json_object"}
        data: dict = {"response_format": custom}
        inject_response_format(data)
        assert data["response_format"] is custom

    def test_returns_data_dict(self):
        data: dict = {}
        result = inject_response_format(data)
        assert result is data


# ---------------------------------------------------------------------------
# validate_response
# ---------------------------------------------------------------------------

def _valid_json(**overrides: object) -> str:
    base = {
        "decision": "APPROVED",
        "confidence_interval": 0.85,
        "reasoning_summary": "The evidence supports approval.",
    }
    base.update(overrides)
    return json.dumps(base)


class TestValidateResponseValid:
    def test_approved(self):
        result = validate_response(_valid_json())
        assert result["decision"] == "APPROVED"

    def test_rejected(self):
        result = validate_response(_valid_json(decision="REJECTED"))
        assert result["decision"] == "REJECTED"

    def test_boundary_confidence_zero(self):
        result = validate_response(_valid_json(confidence_interval=0.0))
        assert result["confidence_interval"] == 0.0

    def test_boundary_confidence_one(self):
        result = validate_response(_valid_json(confidence_interval=1.0))
        assert result["confidence_interval"] == 1.0

    def test_integer_confidence_accepted(self):
        result = validate_response(_valid_json(confidence_interval=1))
        assert result["confidence_interval"] == 1


class TestValidateResponseInvalid:
    def test_not_json(self):
        with pytest.raises(MediatorResponseError, match="not valid JSON"):
            validate_response("this is not json")

    def test_json_array(self):
        with pytest.raises(MediatorResponseError, match="JSON object"):
            validate_response("[]")

    def test_missing_decision(self):
        raw = json.dumps({
            "confidence_interval": 0.5,
            "reasoning_summary": "reason",
        })
        with pytest.raises(MediatorResponseError, match="Missing required"):
            validate_response(raw)

    def test_missing_confidence(self):
        raw = json.dumps({
            "decision": "APPROVED",
            "reasoning_summary": "reason",
        })
        with pytest.raises(MediatorResponseError, match="Missing required"):
            validate_response(raw)

    def test_missing_reasoning(self):
        raw = json.dumps({
            "decision": "APPROVED",
            "confidence_interval": 0.5,
        })
        with pytest.raises(MediatorResponseError, match="Missing required"):
            validate_response(raw)

    def test_invalid_decision_value(self):
        with pytest.raises(MediatorResponseError, match="APPROVED or REJECTED"):
            validate_response(_valid_json(decision="MAYBE"))

    def test_confidence_below_zero(self):
        with pytest.raises(MediatorResponseError, match="\\[0.0, 1.0\\]"):
            validate_response(_valid_json(confidence_interval=-0.1))

    def test_confidence_above_one(self):
        with pytest.raises(MediatorResponseError, match="\\[0.0, 1.0\\]"):
            validate_response(_valid_json(confidence_interval=1.01))

    def test_confidence_not_number(self):
        with pytest.raises(MediatorResponseError, match="must be a number"):
            validate_response(_valid_json(confidence_interval="high"))

    def test_reasoning_empty_string(self):
        with pytest.raises(MediatorResponseError, match="non-empty string"):
            validate_response(_valid_json(reasoning_summary=""))

    def test_reasoning_whitespace_only(self):
        with pytest.raises(MediatorResponseError, match="non-empty string"):
            validate_response(_valid_json(reasoning_summary="   "))

    def test_extra_fields_rejected(self):
        raw = json.dumps({
            "decision": "APPROVED",
            "confidence_interval": 0.8,
            "reasoning_summary": "ok",
            "extra_field": True,
        })
        with pytest.raises(MediatorResponseError, match="Unexpected extra"):
            validate_response(raw)


# ---------------------------------------------------------------------------
# Confidence threshold → NEEDS_REVIEW
# ---------------------------------------------------------------------------

class TestConfidenceThreshold:
    def test_default_threshold_value(self):
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.85

    def test_below_default_threshold_flags_review(self):
        result = validate_response(_valid_json(confidence_interval=0.70))
        assert result["decision"] == "NEEDS_REVIEW"

    def test_at_threshold_stays_approved(self):
        result = validate_response(_valid_json(confidence_interval=0.85))
        assert result["decision"] == "APPROVED"

    def test_above_threshold_stays_approved(self):
        result = validate_response(_valid_json(confidence_interval=0.95))
        assert result["decision"] == "APPROVED"

    def test_rejected_below_threshold_becomes_review(self):
        result = validate_response(
            _valid_json(decision="REJECTED", confidence_interval=0.50),
        )
        assert result["decision"] == "NEEDS_REVIEW"

    def test_custom_threshold(self):
        result = validate_response(
            _valid_json(confidence_interval=0.70),
            confidence_threshold=0.60,
        )
        assert result["decision"] == "APPROVED"

    def test_threshold_zero_disables_check(self):
        result = validate_response(
            _valid_json(confidence_interval=0.01),
            confidence_threshold=0.0,
        )
        assert result["decision"] == "APPROVED"

    def test_threshold_one_flags_everything_below(self):
        result = validate_response(
            _valid_json(confidence_interval=0.99),
            confidence_threshold=1.0,
        )
        assert result["decision"] == "NEEDS_REVIEW"
