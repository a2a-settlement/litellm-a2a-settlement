"""Tests for the PII redaction filter."""

from __future__ import annotations

import pytest

from litellm_a2a_settlement.redaction import (
    EMAIL,
    ETH_WALLET,
    BTC_WALLET,
    SSN,
    PHONE,
    CREDIT_CARD,
    DOB,
    IP_ADDRESS,
    DID_ID,
    JWT_CREDENTIAL,
    VERIFIABLE_CREDENTIAL,
    PiiRedactor,
    RedactionResult,
    redact_message_content,
    redact_payload,
)


# ---------------------------------------------------------------------------
# PiiRedactor unit tests
# ---------------------------------------------------------------------------

class TestPiiRedactorEmail:
    def test_single_email(self):
        r = PiiRedactor()
        result = r.redact("Contact alice@example.com for details.")
        assert "[REDACTED_EMAIL_1]" in result.redacted_text
        assert "alice@example.com" not in result.redacted_text
        assert result.token_map["[REDACTED_EMAIL_1]"] == "alice@example.com"

    def test_multiple_emails(self):
        r = PiiRedactor()
        result = r.redact("From alice@a.com to bob@b.org about it.")
        assert "[REDACTED_EMAIL_1]" in result.redacted_text
        assert "[REDACTED_EMAIL_2]" in result.redacted_text
        assert len(result.token_map) == 2

    def test_duplicate_email_reuses_token(self):
        r = PiiRedactor()
        result = r.redact("alice@a.com cc alice@a.com")
        assert result.redacted_text.count("[REDACTED_EMAIL_1]") == 2
        assert len(result.token_map) == 1


class TestPiiRedactorSSN:
    def test_ssn_pattern(self):
        r = PiiRedactor()
        result = r.redact("SSN: 123-45-6789")
        assert "[REDACTED_SSN_1]" in result.redacted_text
        assert "123-45-6789" not in result.redacted_text


class TestPiiRedactorPhone:
    def test_us_phone(self):
        r = PiiRedactor()
        result = r.redact("Call +1 555-123-4567 now.")
        assert "555-123-4567" not in result.redacted_text
        assert any("PHONE" in k for k in result.token_map)

    def test_phone_with_parens(self):
        r = PiiRedactor()
        result = r.redact("Call (555) 123-4567 now.")
        assert "123-4567" not in result.redacted_text


class TestPiiRedactorWallets:
    def test_ethereum_address(self):
        r = PiiRedactor()
        addr = "0x" + "a1" * 20
        result = r.redact(f"Send to {addr}")
        assert "[REDACTED_ETH_WALLET_1]" in result.redacted_text
        assert addr not in result.redacted_text

    def test_bitcoin_address_legacy(self):
        r = PiiRedactor()
        addr = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
        result = r.redact(f"BTC: {addr}")
        assert addr not in result.redacted_text
        assert any("BTC_WALLET" in k for k in result.token_map)

    def test_bitcoin_bech32(self):
        r = PiiRedactor()
        addr = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
        result = r.redact(f"BTC: {addr}")
        assert addr not in result.redacted_text


class TestPiiRedactorVC:
    def test_vc_json_block(self):
        r = PiiRedactor()
        vc = '{"@context": "https://www.w3.org/2018/credentials/v1", "type": ["VerifiableCredential"], "credentialSubject": {"id": "did:example:123"}}'
        result = r.redact(f"Here is the credential: {vc}")
        assert "@context" not in result.redacted_text
        assert "[REDACTED_VERIFIABLE_CREDENTIAL_1]" in result.redacted_text

    def test_jwt_token(self):
        r = PiiRedactor()
        jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123signature"
        result = r.redact(f"JWT: {jwt}")
        assert "eyJ" not in result.redacted_text


class TestPiiRedactorDID:
    def test_did_identifier(self):
        r = PiiRedactor()
        result = r.redact("Subject: did:web:example.com:user:42")
        assert "did:web:" not in result.redacted_text
        assert any("DID" in k for k in result.token_map)


class TestPiiRedactorIPAddress:
    def test_ipv4(self):
        r = PiiRedactor()
        result = r.redact("Server at 192.168.1.100 responded.")
        assert "192.168.1.100" not in result.redacted_text


class TestPiiRedactorConfig:
    def test_category_filter(self):
        r = PiiRedactor(categories={EMAIL})
        result = r.redact("alice@a.com SSN 123-45-6789")
        assert "alice@a.com" not in result.redacted_text
        assert "123-45-6789" in result.redacted_text

    def test_no_matches_returns_original(self):
        r = PiiRedactor()
        result = r.redact("No sensitive data here.")
        assert result.redacted_text == "No sensitive data here."
        assert result.token_map == {}


# ---------------------------------------------------------------------------
# Payload-level helpers
# ---------------------------------------------------------------------------

class TestRedactMessageContent:
    def test_redacts_string_content(self):
        messages = [
            {"role": "user", "content": "My email is alice@test.com"},
        ]
        redacted, token_map = redact_message_content(messages)
        assert "alice@test.com" not in redacted[0]["content"]
        assert len(token_map) == 1

    def test_redacts_multipart_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Send to alice@test.com"},
                    {"type": "image_url", "image_url": {"url": "http://img.png"}},
                ],
            },
        ]
        redacted, token_map = redact_message_content(messages)
        assert "alice@test.com" not in redacted[0]["content"][0]["text"]
        assert redacted[0]["content"][1] == messages[0]["content"][1]

    def test_preserves_non_content_fields(self):
        messages = [{"role": "system", "content": "Hello", "name": "sys"}]
        redacted, _ = redact_message_content(messages)
        assert redacted[0]["role"] == "system"
        assert redacted[0]["name"] == "sys"

    def test_does_not_mutate_originals(self):
        original = {"role": "user", "content": "alice@test.com"}
        messages = [original]
        redacted, _ = redact_message_content(messages)
        assert original["content"] == "alice@test.com"


class TestRedactPayload:
    def test_redacts_in_place(self):
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "SSN 123-45-6789"}],
        }
        token_map = redact_payload(data)
        assert "123-45-6789" not in data["messages"][0]["content"]
        assert len(token_map) == 1

    def test_no_messages_returns_empty(self):
        assert redact_payload({"model": "gpt-4o"}) == {}

    def test_empty_messages_returns_empty(self):
        assert redact_payload({"model": "gpt-4o", "messages": []}) == {}
