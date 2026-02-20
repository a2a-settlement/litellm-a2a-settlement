"""LiteLLM CustomLogger that wraps A2A agent calls with escrow-based settlement.

Lifecycle
---------
1. async_pre_call_hook      — before the call:   create escrow, redact PII,
                               inject structured-output schema
2. async_log_success_event  — after success:      release escrow
3. async_log_failure_event  — after failure:      refund escrow
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from a2a_settlement.client import SettlementExchangeClient

from .config import SettlementConfig
from .redaction import PiiRedactor, RedactionError, redact_payload, scan_text_for_pii
from .schema import inject_response_format

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError as exc:
    raise ImportError("litellm is required: pip install litellm") from exc

try:
    from litellm.proxy.proxy_server import UserAPIKeyAuth, DualCache
except ImportError:
    UserAPIKeyAuth = Any  # type: ignore[assignment,misc]
    DualCache = Any       # type: ignore[assignment,misc]

logger = logging.getLogger("a2a_settlement.litellm")

_ESCROW_META_KEY     = "_a2a_se_escrow_id"
_MODEL_META_KEY      = "_a2a_se_model"
_REDACTION_META_KEY  = "_a2a_se_redaction_map"


class SettlementHandler(CustomLogger):
    """LiteLLM callback handler that settles A2A agent calls via escrow.

    The pre-call hook also:
    * **Redacts PII** from message contents before they leave the proxy.
    * **Injects response_format** to force the LLM into structured JSON
      mode (mediator verdict schema).

    Usage in proxy_config.yaml::

        litellm_settings:
          callbacks: litellm_a2a_settlement.handler.handler_instance

    Or programmatically::

        from litellm_a2a_settlement import SettlementHandler, SettlementConfig
        import litellm
        litellm.callbacks = [SettlementHandler(SettlementConfig(...))]
    """

    def __init__(
        self,
        config: SettlementConfig | None = None,
        *,
        redactor: PiiRedactor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SettlementConfig()
        self._client = SettlementExchangeClient(
            base_url=self.config.exchange_url,
            api_key=self.config.payer_api_key,
        )
        self._redactor = redactor or PiiRedactor()

    # ------------------------------------------------------------------
    # Pre-call hook
    # ------------------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict | None:
        model: str = data.get("model", "")

        # --- PII redaction (runs for ALL models, not just a2a/) -----------
        # Fail-closed: if redaction raises, wrap in RedactionError and let it
        # propagate so the call is aborted rather than sending raw PII.
        try:
            token_map = redact_payload(data, self._redactor)
            if token_map:
                if "metadata" not in data or data["metadata"] is None:
                    data["metadata"] = {}
                data["metadata"][_REDACTION_META_KEY] = token_map
                logger.info(
                    "Redacted %d PII token(s) from payload for %s",
                    len(token_map), model,
                )
        except Exception as exc:
            logger.error(
                "PII redaction failed for %s — aborting call (fail-closed policy)",
                model, exc_info=True,
            )
            raise RedactionError(
                f"PII redaction failed for model {model}; call aborted to "
                f"prevent raw data from reaching the LLM."
            ) from exc

        # --- Structured JSON output enforcement ---------------------------
        inject_response_format(data)

        if not self.config.should_settle(model):
            return data

        provider_id = self.config.provider_account_id(model)
        if not provider_id:
            metadata = data.get("metadata") or {}
            provider_id = metadata.get("a2a_provider_id")

        if not provider_id:
            logger.debug("No provider_id for %s — skipping escrow", model)
            return data

        task_id   = data.get("litellm_call_id") or f"litellm-{uuid.uuid4().hex[:12]}"
        tokens    = self.config.tokens_for(model)
        ttl       = self.config.ttl_for(model)
        task_type = self.config.task_type_for(model) or _infer_task_type(model)

        try:
            escrow = await asyncio.to_thread(
                self._client.create_escrow,
                provider_id=provider_id,
                amount=tokens,
                task_id=task_id,
                task_type=task_type,
                ttl_minutes=ttl,
            )
            escrow_id = escrow["escrow_id"]
            logger.info(
                "Escrow %s created: %d tokens held for %s (task %s)",
                escrow_id, tokens, model, task_id,
            )
        except Exception:
            logger.warning(
                "Failed to create escrow for %s — proceeding without settlement",
                model, exc_info=True,
            )
            return data

        if "metadata" not in data or data["metadata"] is None:
            data["metadata"] = {}
        data["metadata"][_ESCROW_META_KEY] = escrow_id
        data["metadata"][_MODEL_META_KEY]  = model
        return data

    # ------------------------------------------------------------------
    # Success hook — release escrow
    # ------------------------------------------------------------------

    async def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any,
    ) -> None:
        escrow_id, model = _extract_escrow_meta(kwargs)
        if not escrow_id:
            return

        # --- Double-pass: scan LLM output for PII leaks -------------------
        response_text = _extract_response_text(response_obj)
        if response_text:
            token_map = _extract_redaction_map(kwargs)
            scan = scan_text_for_pii(
                response_text,
                redactor=self._redactor,
                original_values=token_map,
            )
            if not scan.clean:
                categories = {f.category for f in scan.findings}
                logger.warning(
                    "PII leak detected in LLM response for %s "
                    "(escrow %s): %d finding(s) in categories %s — "
                    "refunding instead of releasing",
                    model or "agent", escrow_id,
                    len(scan.findings), categories,
                )
                try:
                    result = await asyncio.to_thread(
                        self._client.refund_escrow,
                        escrow_id=escrow_id,
                        reason="PII leak detected in LLM response",
                    )
                    logger.info(
                        "Escrow %s refunded (PII leak): %d tokens returned",
                        escrow_id, result.get("amount_returned", 0),
                    )
                except Exception:
                    logger.warning(
                        "Failed to refund escrow %s after PII leak",
                        escrow_id, exc_info=True,
                    )
                return

        try:
            result = await asyncio.to_thread(
                self._client.release_escrow, escrow_id=escrow_id,
            )
            logger.info(
                "Escrow %s released: %d tokens paid to %s",
                escrow_id, result.get("amount_paid", 0), model or "agent",
            )
        except Exception:
            logger.warning("Failed to release escrow %s", escrow_id, exc_info=True)

    # ------------------------------------------------------------------
    # Failure hook — refund escrow
    # ------------------------------------------------------------------

    async def async_log_failure_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any,
    ) -> None:
        escrow_id, model = _extract_escrow_meta(kwargs)
        if not escrow_id:
            return
        exception = kwargs.get("exception")
        reason = str(exception) if exception else "LiteLLM call failed"
        try:
            result = await asyncio.to_thread(
                self._client.refund_escrow,
                escrow_id=escrow_id,
                reason=reason[:256],
            )
            logger.info(
                "Escrow %s refunded: %d tokens returned (%s failed: %s)",
                escrow_id, result.get("amount_returned", 0),
                model or "agent", reason[:80],
            )
        except Exception:
            logger.warning("Failed to refund escrow %s", escrow_id, exc_info=True)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _extract_escrow_meta(kwargs: dict) -> tuple[str | None, str | None]:
    """Pull stashed escrow_id and model out of litellm kwargs."""
    metadata: dict = (
        kwargs.get("litellm_params", {}).get("metadata")
        or kwargs.get("metadata")
        or {}
    )
    return metadata.get(_ESCROW_META_KEY), metadata.get(_MODEL_META_KEY)


def _infer_task_type(model: str) -> str:
    """Derive a readable task_type from the model string."""
    if "/" in model:
        _, name = model.split("/", 1)
        return name
    return model or "a2a-task"


def _extract_response_text(response_obj: Any) -> str | None:
    """Best-effort extraction of the text content from a LiteLLM ModelResponse."""
    try:
        return response_obj.choices[0].message.content  # type: ignore[union-attr]
    except (AttributeError, IndexError, TypeError):
        return None


def _extract_redaction_map(kwargs: dict) -> dict[str, str] | None:
    """Retrieve the token map stashed during the pre-call redaction pass."""
    metadata: dict = (
        kwargs.get("litellm_params", {}).get("metadata")
        or kwargs.get("metadata")
        or {}
    )
    return metadata.get(_REDACTION_META_KEY)


#: Default handler instance. Configure via environment variables.
#: Reference in proxy_config.yaml:
#:   callbacks: litellm_a2a_settlement.handler.handler_instance
try:
    handler_instance = SettlementHandler()
except Exception:
    handler_instance = None  # type: ignore[assignment]
