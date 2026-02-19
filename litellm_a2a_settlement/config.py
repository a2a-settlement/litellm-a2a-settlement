"""Configuration for litellm-a2a-settlement."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


EXCHANGE_DEFAULT = "https://exchange.a2a-settlement.org"


def _env_bool(key: str, default: bool = True) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() not in ("false", "0", "no", "off")


@dataclass
class AgentSettlementConfig:
    """Per-agent override for settlement parameters."""

    account_id: str
    tokens_per_call: int | None = None
    ttl_minutes: int | None = None
    task_type: str | None = None


@dataclass
class SettlementConfig:
    """Runtime configuration.

    All fields can be set via env vars or passed directly.

    Env vars:
        A2A_EXCHANGE_URL              — exchange base URL
        A2A_PAYER_API_KEY             — API key of the paying agent
        A2A_DEFAULT_TOKENS_PER_CALL   — tokens to escrow per call (default: 10)
        A2A_DEFAULT_TTL_MINUTES       — escrow expiry in minutes (default: 60)
        A2A_SETTLEMENT_ENABLED        — set to 'false' to disable settlement
    """

    exchange_url: str = field(
        default_factory=lambda: os.environ.get("A2A_EXCHANGE_URL", EXCHANGE_DEFAULT)
    )
    payer_api_key: str | None = field(
        default_factory=lambda: os.environ.get("A2A_PAYER_API_KEY")
    )
    default_tokens_per_call: int = field(
        default_factory=lambda: int(os.environ.get("A2A_DEFAULT_TOKENS_PER_CALL", "10"))
    )
    default_ttl_minutes: int = field(
        default_factory=lambda: int(os.environ.get("A2A_DEFAULT_TTL_MINUTES", "60"))
    )
    enabled: bool = field(
        default_factory=lambda: _env_bool("A2A_SETTLEMENT_ENABLED", default=True)
    )
    agents: dict[str, AgentSettlementConfig] = field(default_factory=dict)

    def should_settle(self, model: str) -> bool:
        if not self.payer_api_key or not self.enabled:
            return False
        if model.startswith("a2a/"):
            return True
        return model in self.agents

    def tokens_for(self, model: str) -> int:
        agent = self.agents.get(model)
        if agent and agent.tokens_per_call is not None:
            return agent.tokens_per_call
        return self.default_tokens_per_call

    def ttl_for(self, model: str) -> int:
        agent = self.agents.get(model)
        if agent and agent.ttl_minutes is not None:
            return agent.ttl_minutes
        return self.default_ttl_minutes

    def provider_account_id(self, model: str) -> str | None:
        agent = self.agents.get(model)
        return agent.account_id if agent else None

    def task_type_for(self, model: str) -> str | None:
        agent = self.agents.get(model)
        return agent.task_type if agent else None
