"""LiteLLM CustomLogger that wraps A2A agent calls with escrow-based settlement.

Lifecycle
---------
1. async_pre_call_hook      — before the call:   create escrow
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

_ESCROW_META_KEY = "_a2a_se_escrow_id"
_MODEL_META_KEY  = "_a2a_se_model"


class SettlementHandler(CustomLogger):
    """LiteLLM callback handler that settles A2A agent calls via escrow.

    Usage in proxy_config.yaml::

        litellm_settings:
          callbacks: litellm_a2a_settlement.handler.handler_instance

    Or programmatically::

        from litellm_a2a_settlement import SettlementHandler, SettlementConfig
        import litellm
        litellm.callbacks = [SettlementHandler(SettlementConfig(...))]
    """

    def __init__(self, config: SettlementConfig | None = None) -> None:
        super().__init__()
        self.config = config or SettlementConfig()
        self._client = SettlementExchangeClient(
            base_url=self.config.exchange_url,
            api_key=self.config.payer_api_key,
        )

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


#: Default handler instance. Configure via environment variables.
#: Reference in proxy_config.yaml:
#:   callbacks: litellm_a2a_settlement.handler.handler_instance
try:
    handler_instance = SettlementHandler()
except Exception:
    handler_instance = None  # type: ignore[assignment]
