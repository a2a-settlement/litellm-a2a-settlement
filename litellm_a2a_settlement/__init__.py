"""LiteLLM callback for A2A Settlement Exchange escrow-based task settlement."""

from __future__ import annotations

from .config import AgentSettlementConfig, SettlementConfig
from .handler import SettlementHandler, handler_instance
from .prompts import (
    DEFAULT_MAX_TRANSCRIPT_TOKENS,
    build_mediator_messages,
    render_system_prompt,
    truncate_transcript,
)
from .redaction import PiiRedactor, RedactionError, RedactionResult, redact_message_content, redact_payload
from .schema import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MEDIATOR_RESPONSE_SCHEMA,
    RESPONSE_FORMAT_PARAM,
    MediatorResponseError,
    inject_response_format,
    validate_response,
)

__all__ = [
    "AgentSettlementConfig",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "MEDIATOR_RESPONSE_SCHEMA",
    "MediatorResponseError",
    "PiiRedactor",
    "RESPONSE_FORMAT_PARAM",
    "RedactionError",
    "RedactionResult",
    "SettlementConfig",
    "SettlementHandler",
    "DEFAULT_MAX_TRANSCRIPT_TOKENS",
    "build_mediator_messages",
    "handler_instance",
    "inject_response_format",
    "redact_message_content",
    "redact_payload",
    "render_system_prompt",
    "truncate_transcript",
    "validate_response",
]
__version__ = "0.1.0"
