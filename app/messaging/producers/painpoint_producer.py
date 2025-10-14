# =============================
# FILE: app/messaging/producers/painpoint_producer.py
# PURPOSE:
#   Small helper to publish a 'painpoint.captured' message
#   from your chat pipeline to RabbitMQ.
#
# HOW TO USE:
#   1) Import `send_painpoint_captured` in your chat flow (where you receive user text).
#   2) Call it with tenant_id, user_id, raw_text, and optional context.
#   3) It generates a message_id and publishes to the correct routing key.
# =============================

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from app.messaging.handlers.settings import PAINPOINTS_ROUTING_KEY  # "painpoint.captured"
from app.messaging.message_strcuture import generate_message
from app.messaging.producer import producer

logger = logging.getLogger(__name__)


def send_painpoint_captured(
    *,
    tenant_id: str,
    user_id: str,
    raw_text: str,
    session_id: Optional[str] = None,
    department: Optional[str] = None,
    source: str = "chat",
    metadata: Optional[Dict[str, Any]] = None,
    message_id: Optional[str] = None,
) -> str:
    """
    Build and publish a painpoint.captured event using the shared message envelope.

    Returns:
        message_id (str) used for idempotency and tracing.
    """
    envelope = _build_envelope(
        tenant_id=tenant_id,
        user_id=user_id,
        raw_text=raw_text,
        session_id=session_id,
        department=department,
        source=source,
        metadata=metadata,
        message_id=message_id,
    )

    _dispatch(envelope)

    msg_id = envelope["payload"]["message_id"]
    logger.info(
        "[PainPointProducer] Published painpoint.captured | tenant=%s user=%s msg=%s",
        tenant_id,
        user_id,
        msg_id,
    )

    return msg_id


def _build_envelope(
    *,
    tenant_id: str,
    user_id: str,
    raw_text: str,
    session_id: Optional[str],
    department: Optional[str],
    source: str,
    metadata: Optional[Dict[str, Any]],
    message_id: Optional[str],
) -> Dict[str, Any]:
    event_metadata = {
        "tenantId": tenant_id,
        "eventType": PAINPOINTS_ROUTING_KEY,
    }
    payload = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "raw_text": raw_text,
        "session_id": session_id,
        "department": department,
        "source": source,
        "metadata": metadata or {},
    }

    message = generate_message(event_metadata, payload)

    # Prefer caller-specified message id if provided.
    if message_id:
        message["metadata"]["messageId"] = message_id

    message["payload"]["message_id"] = message["metadata"]["messageId"]
    return message


async def _publish(message: Dict[str, Any]) -> None:
    await producer.connect()
    await producer.send(PAINPOINTS_ROUTING_KEY, message)


def _dispatch(message: Dict[str, Any]) -> None:
    async def _run():
        try:
            await _publish(message)
        except Exception:
            logger.exception("[PainPointProducer] Failed to publish painpoint message")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_run())
    else:
        task = loop.create_task(_run())
        task.add_done_callback(_log_task_exception)


def _log_task_exception(task: asyncio.Task) -> None:
    if exc := task.exception():
        logger.exception("[PainPointProducer] Async publish failed: %s", exc)
