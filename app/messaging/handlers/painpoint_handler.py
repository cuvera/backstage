# =============================
# FILE: app/messaging/handlers/painpoint_handler.py
# PURPOSE (simplified):
#   - Validate incoming message
#   - Run PainPointAgent barrier
#   - If accepted, enrich and persist
#   - Return True to ACK on success or non-painpoint; False to NACK on transient errors
# =============================

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from aio_pika.abc import AbstractIncomingMessage

from app.services.agents.painpoint_agent import PainPointAgent
from app.services.painpoint_service import PainPointsService

logger = logging.getLogger(__name__)


async def painpoint_handler(message: AbstractIncomingMessage):
    """
    Consume one `painpoint.captured` message from RabbitMQ.

    The message body is expected to be JSON-encoded. On success we simply
    return from the context manager, which ACKs the message. If we raise an
    exception, the message will be re-queued (see `message.process`).
    """
    async with message.process(requeue=True):
        envelope = _decode_body(message)
        payload = _extract_payload(envelope)
        should_ack = await _process_payload(payload)
        if not should_ack:
            # Raising inside the context triggers a NACK + requeue.
            raise RuntimeError("Transient error processing painpoint message")


def _decode_body(message: AbstractIncomingMessage) -> Dict[str, Any]:
    try:
        body = message.body.decode("utf-8")
        return json.loads(body)
    except UnicodeDecodeError:
        logger.error("[PainPointHandler] Invalid UTF-8 body; dropping message")
        return {}
    except json.JSONDecodeError:
        logger.error("[PainPointHandler] Invalid JSON payload; dropping message")
        return {}


def _extract_payload(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """
    Support both the shared message envelope and raw payload formats.
    """
    if not envelope:
        return {}

    payload = envelope.get("payload") if isinstance(envelope, dict) else None
    if isinstance(payload, dict):
        metadata = envelope.get("metadata") or {}
        message_id = payload.get("message_id") or metadata.get("messageId")
        tenant_id = payload.get("tenant_id") or metadata.get("tenantId")

        if message_id:
            payload["message_id"] = message_id
        if tenant_id:
            payload["tenant_id"] = tenant_id

        return payload

    return envelope


async def _process_payload(message: Dict[str, Any]) -> bool:
    """
    Core business logic for handling painpoint messages.

    Returns True to ACK; False to requeue/retry.
    """
    required = ("message_id", "tenant_id", "user_id", "raw_text")
    if any(not message.get(k) for k in required):
        logger.error("[PainPointHandler] Missing required fields; ack to avoid poison loops")
        return True

    context = {
        "message_id": message["message_id"],
        "tenant_id": message["tenant_id"],
        "user_id": message["user_id"],
        "session_id": message.get("session_id"),
        "department": message.get("department"),
        "source": message.get("source", "chat"),
        "metadata": message.get("metadata", {}),
    }
    raw_text = (message.get("raw_text") or "").strip()

    try:
        agent = PainPointAgent()

        gate = agent.barrier(raw_text, context)
        if not gate.get("is_painpoint"):
            logger.info("[PainPointHandler] Not a pain point; ack")
            return True

        payload = agent.enrich(raw_text, context)

        service = await PainPointsService.from_default()
        await service.create_painpoint(payload)

        logger.info("[PainPointHandler] Stored pain point; ack")
        return True

    except Exception as e:
        logger.exception("[PainPointHandler] Error; will requeue: %s", e)
        return False
