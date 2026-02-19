"""Programmatic setup example — per-agent settlement configuration.

This shows how to configure the handler in Python rather than relying
entirely on environment variables.  Useful when you need different token
amounts or TTLs per agent, or when you want to build settlement into an
existing LiteLLM application.

Run
---
    export A2A_EXCHANGE_URL=https://exchange.a2a-settlement.org
    python examples/programmatic_setup.py
"""

from __future__ import annotations

import os

import litellm

from litellm_a2a_settlement import (
    AgentSettlementConfig,
    SettlementConfig,
    SettlementHandler,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1.  Register accounts on the exchange (one-time setup)
    # ------------------------------------------------------------------
    from a2a_settlement.client import SettlementExchangeClient

    exchange_url = os.environ.get(
        "A2A_EXCHANGE_URL", "https://exchange.a2a-settlement.org"
    )
    public = SettlementExchangeClient(exchange_url)

    payer = public.register_account(
        bot_name="LiteLLMProxy",
        developer_id="myorg",
        developer_name="My Organisation",
        contact_email="admin@example.com",
        description="LiteLLM proxy acting as orchestrator / payer",
        skills=["orchestration"],
    )
    payer_key = payer["api_key"]
    print(f"Payer registered. API key: {payer_key}")

    scraper = public.register_account(
        bot_name="WebScraper",
        developer_id="myorg",
        developer_name="My Organisation",
        contact_email="admin@example.com",
        description="Web scraping agent",
        skills=["web-scraping", "html-extraction"],
    )
    scraper_account_id = scraper["account"]["id"]
    print(f"Scraper registered. Account ID: {scraper_account_id}")

    # ------------------------------------------------------------------
    # 2.  Build the settlement config
    # ------------------------------------------------------------------
    config = SettlementConfig(
        exchange_url=exchange_url,
        payer_api_key=payer_key,
        default_tokens_per_call=10,
        default_ttl_minutes=60,
        agents={
            "a2a/web-scraper": AgentSettlementConfig(
                account_id=scraper_account_id,
                tokens_per_call=25,
                ttl_minutes=30,
                task_type="web-scraping",
            ),
        },
    )

    # ------------------------------------------------------------------
    # 3.  Attach the handler to LiteLLM
    # ------------------------------------------------------------------
    handler = SettlementHandler(config)
    litellm.callbacks = [handler]
    print("Settlement handler attached to LiteLLM.")

    # ------------------------------------------------------------------
    # 4.  Make a call — settlement happens automatically
    # ------------------------------------------------------------------
    print("\nExample call (requires a live A2A agent at localhost:10001):")
    print(
        "  litellm.completion("
        'model="a2a/web-scraper", '
        'messages=[{"role": "user", "content": "Scrape https://example.com"}]'
        ")"
    )
    print(
        "\nThe handler will:"
        "\n  1. Create escrow for 25 tokens before the call"
        "\n  2. Release tokens to the scraper on success"
        "\n  3. Refund tokens to the payer on failure"
    )


if __name__ == "__main__":
    main()
