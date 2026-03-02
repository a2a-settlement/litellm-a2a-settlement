# litellm-a2a-settlement

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![CI](https://github.com/a2a-settlement/litellm-a2a-settlement/actions/workflows/ci.yml/badge.svg)](https://github.com/a2a-settlement/litellm-a2a-settlement/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/a2a-settlement/litellm-a2a-settlement/graph/badge.svg)](https://codecov.io/gh/a2a-settlement/litellm-a2a-settlement)

**Escrow-based settlement for LiteLLM A2A agent calls.**

A single LiteLLM callback handler that wraps every A2A agent call with
escrow — tokens are held before the call, released on success, and
refunded on failure. Settlement runs against the live
[A2A Settlement Exchange](https://exchange.a2a-settlement.org).

```
LiteLLM Proxy
  │
  ├── async_pre_call_hook     →  create_escrow  (hold tokens)
  │         ↓
  │   A2A Agent call
  │         ↓
  ├── async_log_success_event →  release_escrow (pay the agent)
  └── async_log_failure_event →  refund_escrow  (return tokens)
```

## How it fits with A2A-SE

This package is a thin integration layer. The settlement protocol,
exchange, and SDK all live in
[a2a-settlement/a2a-settlement](https://github.com/a2a-settlement/a2a-settlement).
This package wires that SDK into LiteLLM's callback system.

| Concern | Package |
|---|---|
| Settlement protocol & spec | `a2a-settlement` |
| Exchange (the running service) | `sandbox.a2a-settlement.org` (default for testing), production at `exchange.a2a-settlement.org` |
| Python SDK | `a2a-settlement` (pip) |
| **LiteLLM integration** | **`litellm-a2a-settlement`** (this package) |

## Installation

```bash
pip install litellm-a2a-settlement
```

## Quick start — proxy config

**1. Register your payer account** on the exchange (one-time):

```python
from a2a_settlement.client import SettlementExchangeClient

c = SettlementExchangeClient("https://sandbox.a2a-settlement.org")
r = c.register_account(
    bot_name="MyProxy",
    developer_id="myorg",
    developer_name="My Org",
    contact_email="me@example.com",
)
print("API key:", r["api_key"])
print("Account:", r["account"]["id"])
```

**2. Register each downstream A2A agent** and note their account IDs.

**3. Set environment variables:**

```bash
export A2A_EXCHANGE_URL=https://sandbox.a2a-settlement.org
export A2A_PAYER_API_KEY=<your payer api key>
```

**4. Add to `proxy_config.yaml`:**

```yaml
model_list:
  - model_name: a2a/web-scraper
    litellm_params:
      model: a2a/web-scraper
      base_url: http://localhost:10001

litellm_settings:
  callbacks: litellm_a2a_settlement.handler.handler_instance
```

**5. Start the proxy:**

```bash
litellm --config proxy_config.yaml
```

Every call to `a2a/*` models now automatically settles.

## Programmatic setup

```python
from litellm_a2a_settlement import SettlementHandler, SettlementConfig, AgentSettlementConfig
import litellm

config = SettlementConfig(
    payer_api_key="your_api_key",
    default_tokens_per_call=10,
    agents={
        "a2a/web-scraper": AgentSettlementConfig(
            account_id="acc_abc123",
            tokens_per_call=25,
            ttl_minutes=30,
            task_type="web-scraping",
        ),
    },
)

litellm.callbacks = [SettlementHandler(config)]
```

## What triggers settlement

Settlement fires when **both** conditions are true:

1. The model string starts with `a2a/` **or** is explicitly listed in `agents`.
2. `payer_api_key` is set and `enabled` is `True`.

Standard LLM calls (`gpt-4o`, `claude-3-5-sonnet`, etc.) are never settled.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `A2A_EXCHANGE_URL` | `https://sandbox.a2a-settlement.org` | Exchange base URL (no `/v1`) |
| `A2A_PAYER_API_KEY` | — | Payer account API key (required) |
| `A2A_DEFAULT_TOKENS_PER_CALL` | `10` | Tokens escrowed per call |
| `A2A_DEFAULT_TTL_MINUTES` | `60` | Escrow expiry |
| `A2A_SETTLEMENT_ENABLED` | `true` | Set to `false` to pause settlement |

## Resilience

Settlement failures are **non-blocking**. If the exchange is unreachable
the agent call proceeds normally and a warning is logged.

## Running tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Related Projects

| Project | Description |
|---------|-------------|
| [a2a-settlement](https://github.com/a2a-settlement/a2a-settlement) | Core exchange + SDK |
| [langgraph-a2a-settlement](https://github.com/a2a-settlement/langgraph-a2a-settlement) | LangGraph integration |
| [crewai-a2a-settlement](https://github.com/a2a-settlement/crewai-a2a-settlement) | CrewAI integration |
| [adk-a2a-settlement](https://github.com/a2a-settlement/adk-a2a-settlement) | Google ADK integration |
| [a2a-settlement-mcp](https://github.com/a2a-settlement/a2a-settlement-mcp) | MCP server for any client |

## License

MIT
