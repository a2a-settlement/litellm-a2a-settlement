"""Tests for SettlementConfig and AgentSettlementConfig."""

from __future__ import annotations

import pytest

from litellm_a2a_settlement.config import AgentSettlementConfig, SettlementConfig


class TestSettlementConfig:
    def test_defaults(self):
        cfg = SettlementConfig(payer_api_key="key123")
        assert cfg.exchange_url == "https://exchange.a2a-settlement.org"
        assert cfg.default_tokens_per_call == 10
        assert cfg.default_ttl_minutes == 60
        assert cfg.enabled is True
        assert cfg.agents == {}

    def test_env_var_exchange_url(self, monkeypatch):
        monkeypatch.setenv("A2A_EXCHANGE_URL", "http://localhost:3000")
        cfg = SettlementConfig()
        assert cfg.exchange_url == "http://localhost:3000"

    def test_env_var_payer_key(self, monkeypatch):
        monkeypatch.setenv("A2A_PAYER_API_KEY", "secret")
        cfg = SettlementConfig()
        assert cfg.payer_api_key == "secret"

    def test_env_var_tokens(self, monkeypatch):
        monkeypatch.setenv("A2A_DEFAULT_TOKENS_PER_CALL", "50")
        cfg = SettlementConfig()
        assert cfg.default_tokens_per_call == 50

    def test_env_var_disabled(self, monkeypatch):
        monkeypatch.setenv("A2A_SETTLEMENT_ENABLED", "false")
        cfg = SettlementConfig()
        assert cfg.enabled is False

    def test_env_var_disabled_zero(self, monkeypatch):
        monkeypatch.setenv("A2A_SETTLEMENT_ENABLED", "0")
        cfg = SettlementConfig()
        assert cfg.enabled is False

    def test_tokens_for_default(self):
        cfg = SettlementConfig(payer_api_key="k", default_tokens_per_call=15)
        assert cfg.tokens_for("a2a/some-agent") == 15

    def test_tokens_for_override(self):
        cfg = SettlementConfig(
            payer_api_key="k",
            default_tokens_per_call=10,
            agents={
                "a2a/scraper": AgentSettlementConfig(
                    account_id="acc_1", tokens_per_call=30
                )
            },
        )
        assert cfg.tokens_for("a2a/scraper") == 30
        assert cfg.tokens_for("a2a/other") == 10

    def test_ttl_for_override(self):
        cfg = SettlementConfig(
            payer_api_key="k",
            default_ttl_minutes=60,
            agents={"a2a/fast": AgentSettlementConfig(account_id="acc_2", ttl_minutes=5)},
        )
        assert cfg.ttl_for("a2a/fast") == 5
        assert cfg.ttl_for("a2a/other") == 60

    def test_provider_account_id(self):
        cfg = SettlementConfig(
            payer_api_key="k",
            agents={"a2a/x": AgentSettlementConfig(account_id="acc_xyz")},
        )
        assert cfg.provider_account_id("a2a/x") == "acc_xyz"
        assert cfg.provider_account_id("a2a/unknown") is None

    def test_task_type_for(self):
        cfg = SettlementConfig(
            payer_api_key="k",
            agents={"a2a/x": AgentSettlementConfig(account_id="acc_1", task_type="scraping")},
        )
        assert cfg.task_type_for("a2a/x") == "scraping"
        assert cfg.task_type_for("a2a/other") is None


class TestShouldSettle:
    def test_no_payer_key_returns_false(self):
        cfg = SettlementConfig(payer_api_key=None)
        assert cfg.should_settle("a2a/agent") is False

    def test_disabled_returns_false(self):
        cfg = SettlementConfig(payer_api_key="k", enabled=False)
        assert cfg.should_settle("a2a/agent") is False

    def test_a2a_prefix_returns_true(self):
        cfg = SettlementConfig(payer_api_key="k")
        assert cfg.should_settle("a2a/agent") is True

    def test_registered_agent_no_prefix_returns_true(self):
        cfg = SettlementConfig(
            payer_api_key="k",
            agents={"my-custom-agent": AgentSettlementConfig(account_id="acc_1")},
        )
        assert cfg.should_settle("my-custom-agent") is True

    def test_plain_llm_returns_false(self):
        cfg = SettlementConfig(payer_api_key="k")
        assert cfg.should_settle("gpt-4o") is False
        assert cfg.should_settle("openai/gpt-4o") is False
