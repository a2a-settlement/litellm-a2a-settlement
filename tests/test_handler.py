"""Tests for SettlementHandler — pre-call, success, and failure hooks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from litellm_a2a_settlement.config import AgentSettlementConfig, SettlementConfig
from litellm_a2a_settlement.handler import (
    SettlementHandler,
    _ESCROW_META_KEY,
    _MODEL_META_KEY,
    _infer_task_type,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return SettlementConfig(
        exchange_url="http://localhost:3000",
        payer_api_key="test_key",
        default_tokens_per_call=10,
        default_ttl_minutes=60,
        agents={
            "a2a/scraper": AgentSettlementConfig(
                account_id="acc_scraper",
                tokens_per_call=25,
                task_type="scraping",
            )
        },
    )


@pytest.fixture
def handler(config):
    with patch("litellm_a2a_settlement.handler.SettlementExchangeClient"):
        h = SettlementHandler(config)
        h._client = MagicMock()
        return h


@pytest.fixture
def mock_auth():
    return MagicMock()


@pytest.fixture
def mock_cache():
    return MagicMock()


# ---------------------------------------------------------------------------
# Helper to build call data dicts
# ---------------------------------------------------------------------------

def call_data(model: str, call_id: str = "call-abc", metadata: dict | None = None):
    return {
        "model": model,
        "litellm_call_id": call_id,
        "metadata": metadata or {},
        "messages": [{"role": "user", "content": "do something"}],
    }


def success_kwargs(model: str, escrow_id: str):
    return {
        "model": model,
        "litellm_params": {
            "metadata": {
                _ESCROW_META_KEY: escrow_id,
                _MODEL_META_KEY: model,
            }
        },
    }


def failure_kwargs(model: str, escrow_id: str, exception: Exception | None = None):
    kw = success_kwargs(model, escrow_id)
    kw["exception"] = exception or RuntimeError("timeout")
    return kw


# ---------------------------------------------------------------------------
# Pre-call hook
# ---------------------------------------------------------------------------

class TestPreCallHook:
    async def test_creates_escrow_for_a2a_model(self, handler, mock_auth, mock_cache):
        handler._client.create_escrow.return_value = {"escrow_id": "esc_001"}

        data = call_data("a2a/scraper")
        result = await handler.async_pre_call_hook(mock_auth, mock_cache, data, "completion")

        handler._client.create_escrow.assert_called_once_with(
            provider_id="acc_scraper",
            amount=25,
            task_id="call-abc",
            task_type="scraping",
            ttl_minutes=60,
        )
        assert result["metadata"][_ESCROW_META_KEY] == "esc_001"
        assert result["metadata"][_MODEL_META_KEY] == "a2a/scraper"

    async def test_skips_non_a2a_model(self, handler, mock_auth, mock_cache):
        data = call_data("gpt-4o")
        result = await handler.async_pre_call_hook(mock_auth, mock_cache, data, "completion")

        handler._client.create_escrow.assert_not_called()
        assert _ESCROW_META_KEY not in (result.get("metadata") or {})

    async def test_skips_when_disabled(self, mock_auth, mock_cache):
        config = SettlementConfig(payer_api_key="k", enabled=False)
        with patch("litellm_a2a_settlement.handler.SettlementExchangeClient"):
            h = SettlementHandler(config)
            h._client = MagicMock()

        data = call_data("a2a/agent")
        await h.async_pre_call_hook(mock_auth, mock_cache, data, "completion")
        h._client.create_escrow.assert_not_called()

    async def test_skips_when_no_payer_key(self, mock_auth, mock_cache):
        config = SettlementConfig(payer_api_key=None)
        with patch("litellm_a2a_settlement.handler.SettlementExchangeClient"):
            h = SettlementHandler(config)
            h._client = MagicMock()

        data = call_data("a2a/agent")
        await h.async_pre_call_hook(mock_auth, mock_cache, data, "completion")
        h._client.create_escrow.assert_not_called()

    async def test_skips_when_no_provider_id(self, handler, mock_auth, mock_cache):
        data = call_data("a2a/unknown-agent")
        await handler.async_pre_call_hook(mock_auth, mock_cache, data, "completion")
        handler._client.create_escrow.assert_not_called()

    async def test_uses_metadata_provider_id_fallback(self, handler, mock_auth, mock_cache):
        handler._client.create_escrow.return_value = {"escrow_id": "esc_002"}
        data = call_data("a2a/dynamic", metadata={"a2a_provider_id": "acc_dynamic"})

        await handler.async_pre_call_hook(mock_auth, mock_cache, data, "completion")

        handler._client.create_escrow.assert_called_once()
        call_kwargs = handler._client.create_escrow.call_args.kwargs
        assert call_kwargs["provider_id"] == "acc_dynamic"

    async def test_exchange_failure_is_non_blocking(self, handler, mock_auth, mock_cache):
        handler._client.create_escrow.side_effect = RuntimeError("exchange down")

        data = call_data("a2a/scraper")
        result = await handler.async_pre_call_hook(mock_auth, mock_cache, data, "completion")

        assert result is not None
        assert _ESCROW_META_KEY not in (result.get("metadata") or {})

    async def test_initialises_metadata_if_none(self, handler, mock_auth, mock_cache):
        handler._client.create_escrow.return_value = {"escrow_id": "esc_003"}
        data = call_data("a2a/scraper")
        data["metadata"] = None

        result = await handler.async_pre_call_hook(mock_auth, mock_cache, data, "completion")
        assert result["metadata"][_ESCROW_META_KEY] == "esc_003"


# ---------------------------------------------------------------------------
# Success hook
# ---------------------------------------------------------------------------

class TestSuccessHook:
    async def test_releases_escrow(self, handler):
        handler._client.release_escrow.return_value = {"amount_paid": 25}
        kwargs = success_kwargs("a2a/scraper", "esc_001")

        await handler.async_log_success_event(kwargs, None, None, None)

        handler._client.release_escrow.assert_called_once_with(escrow_id="esc_001")

    async def test_no_escrow_id_skips(self, handler):
        await handler.async_log_success_event({}, None, None, None)
        handler._client.release_escrow.assert_not_called()

    async def test_release_failure_is_non_blocking(self, handler):
        handler._client.release_escrow.side_effect = RuntimeError("exchange down")
        kwargs = success_kwargs("a2a/scraper", "esc_001")

        await handler.async_log_success_event(kwargs, None, None, None)


# ---------------------------------------------------------------------------
# Failure hook
# ---------------------------------------------------------------------------

class TestFailureHook:
    async def test_refunds_escrow(self, handler):
        handler._client.refund_escrow.return_value = {"amount_returned": 25}
        kwargs = failure_kwargs("a2a/scraper", "esc_001", RuntimeError("timeout"))

        await handler.async_log_failure_event(kwargs, None, None, None)

        handler._client.refund_escrow.assert_called_once_with(
            escrow_id="esc_001",
            reason="timeout",
        )

    async def test_no_escrow_id_skips(self, handler):
        await handler.async_log_failure_event({}, None, None, None)
        handler._client.refund_escrow.assert_not_called()

    async def test_refund_failure_is_non_blocking(self, handler):
        handler._client.refund_escrow.side_effect = RuntimeError("exchange down")
        kwargs = failure_kwargs("a2a/scraper", "esc_001")

        await handler.async_log_failure_event(kwargs, None, None, None)

    async def test_reason_truncated_to_256_chars(self, handler):
        handler._client.refund_escrow.return_value = {"amount_returned": 0}
        long_msg = "x" * 500
        kwargs = failure_kwargs("a2a/scraper", "esc_001", RuntimeError(long_msg))

        await handler.async_log_failure_event(kwargs, None, None, None)

        call_kwargs = handler._client.refund_escrow.call_args.kwargs
        assert len(call_kwargs["reason"]) <= 256


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestInferTaskType:
    def test_a2a_prefix_stripped(self):
        assert _infer_task_type("a2a/web-scraper") == "web-scraper"

    def test_other_prefix_stripped(self):
        assert _infer_task_type("openai/gpt-4o") == "gpt-4o"

    def test_no_slash(self):
        assert _infer_task_type("my-agent") == "my-agent"

    def test_empty_string(self):
        assert _infer_task_type("") == "a2a-task"
