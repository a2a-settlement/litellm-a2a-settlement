"""Tests for the PII redaction filter."""

from __future__ import annotations

import pytest

from litellm_a2a_settlement.redaction import (
    EMAIL,
    ETH_WALLET,
    BTC_WALLET,
    LTC_WALLET,
    XRP_ADDRESS,
    ADA_ADDRESS,
    COSMOS_ADDRESS,
    TRX_ADDRESS,
    XMR_ADDRESS,
    SOL_WALLET,
    SSN,
    PHONE,
    CREDIT_CARD,
    DOB,
    IP_ADDRESS,
    DID_ID,
    JWT_CREDENTIAL,
    VERIFIABLE_CREDENTIAL,
    PiiRedactor,
    PiiLeakScanResult,
    RedactionResult,
    _short_hash,
    redact_message_content,
    redact_payload,
    scan_text_for_pii,
)


# ---------------------------------------------------------------------------
# PiiRedactor unit tests
# ---------------------------------------------------------------------------

class TestPiiRedactorEmail:
    def test_single_email(self):
        r = PiiRedactor()
        result = r.redact("Contact alice@example.com for details.")
        digest = _short_hash("alice@example.com")
        expected_token = f"[REDACTED_EMAIL_1:{digest}]"
        assert expected_token in result.redacted_text
        assert "alice@example.com" not in result.redacted_text
        assert result.token_map[expected_token] == "alice@example.com"

    def test_multiple_emails(self):
        r = PiiRedactor()
        result = r.redact("From alice@a.com to bob@b.org about it.")
        assert "REDACTED_EMAIL_1:" in result.redacted_text
        assert "REDACTED_EMAIL_2:" in result.redacted_text
        assert len(result.token_map) == 2

    def test_duplicate_email_reuses_token(self):
        r = PiiRedactor()
        result = r.redact("alice@a.com cc alice@a.com")
        digest = _short_hash("alice@a.com")
        expected_token = f"[REDACTED_EMAIL_1:{digest}]"
        assert result.redacted_text.count(expected_token) == 2
        assert len(result.token_map) == 1


class TestPiiRedactorSSN:
    def test_ssn_pattern(self):
        r = PiiRedactor()
        result = r.redact("SSN: 123-45-6789")
        assert "REDACTED_SSN_1:" in result.redacted_text
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
        assert "REDACTED_ETH_WALLET_1:" in result.redacted_text
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
        assert "REDACTED_VERIFIABLE_CREDENTIAL_1:" in result.redacted_text

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


# ---------------------------------------------------------------------------
# Output scanning (double-pass)
# ---------------------------------------------------------------------------

class TestScanTextForPii:
    def test_clean_output_returns_clean(self):
        result = scan_text_for_pii("The settlement is APPROVED with high confidence.")
        assert result.clean is True
        assert result.findings == []

    def test_detects_pii_pattern_in_output(self):
        result = scan_text_for_pii(
            "The claimant's email is alice@example.com which was referenced."
        )
        assert result.clean is False
        assert len(result.findings) >= 1
        assert any(f.source == "pattern" and f.category == "EMAIL" for f in result.findings)

    def test_detects_original_value_echo(self):
        token_map = {"[REDACTED_SSN_1:abc123]": "123-45-6789"}
        text = "The SSN 123-45-6789 was found in records."
        result = scan_text_for_pii(text, original_values=token_map)
        assert result.clean is False
        assert any(f.matched_text == "123-45-6789" for f in result.findings)

    def test_mixed_findings(self):
        token_map = {"[REDACTED_CUSTOM_1:abc123]": "Project Nightfall"}
        text = "Project Nightfall and new email bob@evil.com appeared."
        result = scan_text_for_pii(text, original_values=token_map)
        assert result.clean is False
        sources = {f.source for f in result.findings}
        assert "pattern" in sources
        assert "echo" in sources

    def test_redaction_tokens_not_flagged(self):
        text = (
            "The decision references [REDACTED_EMAIL_1:a3f2c8] "
            "and [REDACTED_SSN_1:d4e5f6] as relevant parties."
        )
        result = scan_text_for_pii(text)
        assert result.clean is True

    def test_echo_deduplicates_with_pattern(self):
        """If a value is caught by both pattern and echo, only one finding."""
        token_map = {"[REDACTED_EMAIL_1:aabbcc]": "alice@example.com"}
        text = "Leaked: alice@example.com"
        result = scan_text_for_pii(text, original_values=token_map)
        assert result.clean is False
        matched_texts = [f.matched_text for f in result.findings]
        assert matched_texts.count("alice@example.com") == 1


# ---------------------------------------------------------------------------
# Expanded crypto wallet patterns
# ---------------------------------------------------------------------------

class TestPiiRedactorCryptoExpanded:
    def test_litecoin_bech32(self):
        r = PiiRedactor()
        addr = "ltc1qw508d6qejxtdg4y5r3zarvary0c5xw7kgmn4n9"
        result = r.redact(f"LTC: {addr}")
        assert addr not in result.redacted_text
        assert any("LTC_WALLET" in k for k in result.token_map)

    def test_xrp_address(self):
        r = PiiRedactor()
        addr = "rN7Drz9TBajFrzBJFEn1p2hSgkDLDhHqcY"
        result = r.redact(f"XRP: {addr}")
        assert addr not in result.redacted_text
        assert any("XRP_ADDRESS" in k for k in result.token_map)

    def test_cardano_address(self):
        r = PiiRedactor()
        addr = "addr1" + "a" * 58
        result = r.redact(f"ADA: {addr}")
        assert addr not in result.redacted_text
        assert any("ADA_ADDRESS" in k for k in result.token_map)

    def test_cosmos_address(self):
        r = PiiRedactor()
        addr = "cosmos1" + "a" * 38
        result = r.redact(f"ATOM: {addr}")
        assert addr not in result.redacted_text
        assert any("COSMOS_ADDRESS" in k for k in result.token_map)

    def test_tron_address(self):
        r = PiiRedactor()
        addr = "T" + "A" * 33
        result = r.redact(f"TRX: {addr}")
        assert addr not in result.redacted_text
        assert any("TRX_ADDRESS" in k for k in result.token_map)

    def test_monero_address(self):
        r = PiiRedactor()
        addr = "4" + "A" * 94
        result = r.redact(f"XMR: {addr}")
        assert addr not in result.redacted_text
        assert any("XMR_ADDRESS" in k for k in result.token_map)

    def test_solana_now_active(self):
        r = PiiRedactor()
        addr = "7EcDhSYGxXyscszYEp35KHN8vvw3svAuLKTzXwCFLtV"
        result = r.redact(f"SOL: {addr}")
        assert addr not in result.redacted_text
        assert any("SOL_WALLET" in k for k in result.token_map)
