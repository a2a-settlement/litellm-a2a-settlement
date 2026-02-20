"""Shared fixtures for the test suite."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from litellm_a2a_settlement.config import AgentSettlementConfig, SettlementConfig
from litellm_a2a_settlement.handler import SettlementHandler


def _make_mediator_json(
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


class MockLLMCompletion:
    """Drop-in replacement for a litellm.completion response object."""

    def __init__(self, content: str, model: str = "gpt-4o") -> None:
        self.choices = [MagicMock()]
        self.choices[0].message.content = content
        self.model = model
        self.usage = MagicMock(total_tokens=150, prompt_tokens=120, completion_tokens=30)


@pytest.fixture
def mock_llm_approved():
    """Patch litellm.acompletion to return an APPROVED mediator verdict."""
    response = MockLLMCompletion(_make_mediator_json())
    with patch("litellm.acompletion", return_value=response) as m:
        yield m


@pytest.fixture
def mock_llm_rejected():
    """Patch litellm.acompletion to return a REJECTED mediator verdict."""
    response = MockLLMCompletion(_make_mediator_json(
        decision="REJECTED",
        confidence=0.88,
        reasoning="Proposed amount exceeds the single-party liability limit.",
    ))
    with patch("litellm.acompletion", return_value=response) as m:
        yield m


@pytest.fixture
def mock_llm_low_confidence():
    """Patch litellm.acompletion to return a low-confidence verdict."""
    response = MockLLMCompletion(_make_mediator_json(
        decision="APPROVED",
        confidence=0.62,
        reasoning="Transcript evidence is ambiguous.",
    ))
    with patch("litellm.acompletion", return_value=response) as m:
        yield m
