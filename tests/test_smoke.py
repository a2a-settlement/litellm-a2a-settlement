"""Smoke tests — verify core imports and basic wiring work."""

from __future__ import annotations


def test_litellm_importable():
    import litellm  # noqa: F401


def test_package_importable():
    from litellm_a2a_settlement import (
        AgentSettlementConfig,
        SettlementConfig,
        SettlementHandler,
    )
    assert SettlementConfig is not None
    assert SettlementHandler is not None
    assert AgentSettlementConfig is not None


def test_config_round_trip():
    from litellm_a2a_settlement.config import AgentSettlementConfig, SettlementConfig

    cfg = SettlementConfig(
        exchange_url="http://localhost:3000",
        payer_api_key="test_key",
        default_tokens_per_call=10,
        agents={
            "a2a/test": AgentSettlementConfig(account_id="acc_1", tokens_per_call=20)
        },
    )
    assert cfg.should_settle("a2a/test") is True
    assert cfg.should_settle("gpt-4o") is False
    assert cfg.tokens_for("a2a/test") == 20
    assert cfg.tokens_for("a2a/other") == 10
    assert cfg.provider_account_id("a2a/test") == "acc_1"


def test_handler_instantiation_with_mock():
    from unittest.mock import patch

    from litellm_a2a_settlement.config import SettlementConfig
    from litellm_a2a_settlement.handler import SettlementHandler

    cfg = SettlementConfig(
        exchange_url="http://localhost:3000",
        payer_api_key="test_key",
    )
    with patch("litellm_a2a_settlement.handler.SettlementExchangeClient"):
        h = SettlementHandler(cfg)
        assert h.config is cfg
