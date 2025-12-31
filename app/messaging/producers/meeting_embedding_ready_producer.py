# =============================
# FILE: app/messaging/producers/meeting_embedding_ready_producer.py
# PURPOSE:
#   Helper to publish meeting embedding ready events to RabbitMQ queue
#   'meetings.embedding.ready' when meeting analysis finishes successfully.
#
# HOW TO USE:
#   1) Import `send_meeting_embedding_ready` in your orchestrator.
#   2) Call it with meeting_id, tenant_id, and optional context after successful analysis.
#   3) It generates a message_id and publishes to the correct queue.
# =============================

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from app.messaging.message_strcuture import generate_message
from app.messaging.producer import producer

logger = logging.getLogger(__name__)

# Queue name for meeting embedding ready events
MEETING_EMBEDDING_READY_QUEUE = "meeting.embedding.ready"


def send_meeting_embedding_ready(
    *,
    meeting_id: str,
    tenant_id: str,
    platform: Optional[str] = None,
) -> str:
    """
    Build and publish a meeting embedding ready event using the shared message envelope.

    Args:
        meeting_id: Unique identifier for the meeting
        tenant_id: Tenant identifier
        platform: Platform type (google, offline, etc.) (optional)

    Returns:
        message_id (str) used for idempotency and tracing.
    """
    envelope = _build_envelope(
        meeting_id=meeting_id,
        tenant_id=tenant_id,
        platform=platform,
    )

    _dispatch(envelope)

    logger.info(
        "[MeetingEmbeddingReadyProducer] Published embedding ready | meeting=%s tenant=%s platform=%s",
        meeting_id,
        tenant_id,
        platform
    )

def _build_envelope(
    *,
    meeting_id: str,
    tenant_id: str,
    platform: Optional[str],
) -> Dict[str, Any]:
    event_metadata = {
        "tenantId": tenant_id,
        "eventType": "meeting.embedding.ready",
    }
    payload = {
        "meetingId": meeting_id,
        "tenantId": tenant_id,
        "platform": platform,
    }

    message = generate_message(event_metadata, payload)

    return message


def _dispatch(message: Dict[str, Any]) -> None:
    async def _run():
        try:
            await producer.send(MEETING_EMBEDDING_READY_QUEUE, message)
        except Exception:
            logger.exception("[MeetingEmbeddingReadyProducer] Failed to publish meeting embedding ready message")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_run())
    else:
        task = loop.create_task(_run())
        task.add_done_callback(_log_task_exception)


def _log_task_exception(task: asyncio.Task) -> None:
    if exc := task.exception():
        logger.exception("[MeetingEmbeddingReadyProducer] Async publish failed: %s", exc)
