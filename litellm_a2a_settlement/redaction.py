"""Pre-processing redaction filter for PII, wallet addresses, and Verifiable Credentials.

Strips sensitive data from payloads before they reach the LLM, replacing
each occurrence with a deterministic generic token so that the LLM can still
reason about entity relationships without seeing raw values.

Token format:  ``[REDACTED_<CATEGORY>_<N>:<hash>]``
(e.g. ``[REDACTED_EMAIL_1:a3f2c8]``)

The trailing 6-hex-char hash is a truncated SHA-256 of the original value.
This allows the LLM to recognise that *something* was present and to correlate
repeated references without ever seeing the raw data.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any


def _short_hash(value: str) -> str:
    """Return a 6-character hex digest of *value* for redaction tokens."""
    return hashlib.sha256(value.encode()).hexdigest()[:6]


class RedactionError(RuntimeError):
    """Raised when the redaction filter fails.

    Under the fail-closed policy the entire LLM call MUST be aborted when
    this exception is raised rather than sending raw data to the model.
    """


# ---------------------------------------------------------------------------
# Pattern registry — order matters: more specific patterns first
# ---------------------------------------------------------------------------

_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
)

_PHONE = re.compile(
    r"(?<!\w)"
    r"(?:\+?1[\s.\-]?)?"
    r"(?:\(?\d{3}\)?[\s.\-]?)"
    r"\d{3}[\s.\-]?\d{4}"
    r"(?!\w)"
)

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

_CREDIT_CARD = re.compile(r"\b(?:\d[ \-]?){13,19}\b")

_DATE_OF_BIRTH = re.compile(
    r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b"
)

_IP_V4 = re.compile(
    r"\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}\b"
)

# Crypto wallet addresses
_ETH_ADDRESS = re.compile(r"\b0x[0-9a-fA-F]{40}\b")
_BTC_ADDRESS = re.compile(r"\b(?:bc1|[13])[a-zA-HJ-NP-Z0-9]{25,62}\b")
_SOLANA_ADDRESS = re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")

# W3C Verifiable Credential JSON blobs — match objects containing the
# ``VerifiableCredential`` type and a ``@context`` field.  Supports one
# level of nested braces (e.g. ``credentialSubject: {... }``).
_VC_JSON_BLOCK = re.compile(
    r"\{(?:[^{}]|\{[^{}]*\})*\"@context\"(?:[^{}]|\{[^{}]*\})*\"VerifiableCredential\"(?:[^{}]|\{[^{}]*\})*\}",
    re.DOTALL,
)

# JWT-encoded Verifiable Credentials (compact JWS: header.payload.signature)
_JWT_TOKEN = re.compile(
    r"\beyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"
)

# DID identifiers  (did:method:specific-id)
_DID = re.compile(r"\bdid:[a-z0-9]+:[A-Za-z0-9._:\-]+\b")


# ---------------------------------------------------------------------------
# Redaction category enum-like constants
# ---------------------------------------------------------------------------

EMAIL = "EMAIL"
PHONE = "PHONE"
SSN = "SSN"
CREDIT_CARD = "CREDIT_CARD"
DOB = "DOB"
IP_ADDRESS = "IP_ADDRESS"
ETH_WALLET = "ETH_WALLET"
BTC_WALLET = "BTC_WALLET"
SOL_WALLET = "SOL_WALLET"
VERIFIABLE_CREDENTIAL = "VERIFIABLE_CREDENTIAL"
JWT_CREDENTIAL = "JWT_CREDENTIAL"
DID_ID = "DID"

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (SSN, _SSN),
    (EMAIL, _EMAIL),
    (PHONE, _PHONE),
    (CREDIT_CARD, _CREDIT_CARD),
    (DOB, _DATE_OF_BIRTH),
    (IP_ADDRESS, _IP_V4),
    (ETH_WALLET, _ETH_ADDRESS),
    (BTC_WALLET, _BTC_ADDRESS),
    (DID_ID, _DID),
    (JWT_CREDENTIAL, _JWT_TOKEN),
    (VERIFIABLE_CREDENTIAL, _VC_JSON_BLOCK),
]


# ---------------------------------------------------------------------------
# Redactor
# ---------------------------------------------------------------------------

@dataclass
class RedactionResult:
    """Holds the redacted text and a mapping from tokens back to originals."""

    redacted_text: str
    token_map: dict[str, str] = field(default_factory=dict)


class PiiRedactor:
    """Deterministic, regex-based redaction engine.

    Each unique sensitive value gets a stable token within one redaction pass.
    The token map is returned so the caller can audit what was stripped.
    """

    def __init__(
        self,
        *,
        extra_patterns: list[tuple[str, re.Pattern[str]]] | None = None,
        categories: set[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        extra_patterns:
            Additional ``(category, compiled_regex)`` pairs appended after
            the built-in patterns.
        categories:
            If provided, only these categories are redacted.  ``None`` means
            all categories are active.
        """
        patterns = list(_PATTERNS)
        if extra_patterns:
            patterns.extend(extra_patterns)
        if categories is not None:
            patterns = [(c, p) for c, p in patterns if c in categories]
        self._patterns = patterns

    def redact(self, text: str) -> RedactionResult:
        """Return a ``RedactionResult`` with all sensitive values replaced."""
        token_map: dict[str, str] = {}
        counters: dict[str, int] = {}
        seen: dict[str, str] = {}

        for category, pattern in self._patterns:
            def _replacer(m: re.Match[str], _cat: str = category) -> str:
                value = m.group(0)
                if value in seen:
                    return seen[value]
                count = counters.get(_cat, 0) + 1
                counters[_cat] = count
                digest = _short_hash(value)
                token = f"[REDACTED_{_cat}_{count}:{digest}]"
                seen[value] = token
                token_map[token] = value
                return token

            text = pattern.sub(_replacer, text)

        return RedactionResult(redacted_text=text, token_map=token_map)


# ---------------------------------------------------------------------------
# Payload-level helpers
# ---------------------------------------------------------------------------

_DEFAULT_REDACTOR = PiiRedactor()


def redact_message_content(
    messages: list[dict[str, Any]],
    redactor: PiiRedactor | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Redact every ``content`` field in a LiteLLM message list.

    Returns the (possibly mutated) messages and the combined token map.
    """
    r = redactor or _DEFAULT_REDACTOR
    combined_map: dict[str, str] = {}
    out: list[dict[str, Any]] = []

    for msg in messages:
        msg = dict(msg)  # shallow copy
        content = msg.get("content")
        if isinstance(content, str):
            result = r.redact(content)
            msg["content"] = result.redacted_text
            combined_map.update(result.token_map)
        elif isinstance(content, list):
            new_parts: list[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part = dict(part)
                    result = r.redact(part.get("text", ""))
                    part["text"] = result.redacted_text
                    combined_map.update(result.token_map)
                new_parts.append(part)
            msg["content"] = new_parts
        out.append(msg)

    return out, combined_map


def redact_payload(
    data: dict[str, Any],
    redactor: PiiRedactor | None = None,
) -> dict[str, str]:
    """Redact message contents inside a LiteLLM call ``data`` dict in-place.

    Returns the combined token map of all replacements made.
    """
    messages = data.get("messages")
    if not messages:
        return {}
    redacted, token_map = redact_message_content(messages, redactor)
    data["messages"] = redacted
    return token_map
