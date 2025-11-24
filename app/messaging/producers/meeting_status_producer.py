# =============================
# FILE: app/messaging/producers/meeting_status_producer.py
# PURPOSE:
#   Helper to publish meeting status updates to RabbitMQ queue
#   'meetings.offline.status' for offline meeting processing.
#
# HOW TO USE:
#   1) Import `send_meeting_status` in your orchestrator or processing flow.
#   2) Call it with meeting_id, status, platform, and optional context.
#   3) It generates a message_id and publishes to the correct queue.
# =============================

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from app.messaging.message_strcuture import generate_message
from app.messaging.producer import producer

logger = logging.getLogger(__name__)

# Queue name for offline meeting status updates
MEETING_STATUS_QUEUE = "meetings.offline.status"


def send_meeting_status(
    *,
    meeting_id: str,
    status: str,
    platform: str = "offline",
    tenant_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    message_id: Optional[str] = None,
) -> str:
    """
    Build and publish a meeting status update event using the shared message envelope.

    Args:
        meeting_id: Unique identifier for the meeting
        status: Meeting status (pending, completed, failed)
        platform: Platform type (default: "offline")
        tenant_id: Tenant identifier (optional)
        session_id: Session identifier (optional)
        metadata: Additional metadata (optional)
        message_id: Custom message ID for idempotency (optional)

    Returns:
        message_id (str) used for idempotency and tracing.
    """
    envelope = _build_envelope(
        meeting_id=meeting_id,
        status=status,
        platform=platform,
        tenant_id=tenant_id,
        session_id=session_id,
        metadata=metadata,
        message_id=message_id,
    )

    _dispatch(envelope)

    msg_id = envelope["payload"]["message_id"]
    logger.info(
        "[MeetingStatusProducer] Published meeting status | meeting=%s status=%s platform=%s msg=%s",
        meeting_id,
        status,
        platform,
        msg_id,
    )

    return msg_id


def _build_envelope(
    *,
    meeting_id: str,
    status: str,
    platform: str,
    tenant_id: Optional[str],
    session_id: Optional[str],
    metadata: Optional[Dict[str, Any]],
    message_id: Optional[str],
) -> Dict[str, Any]:
    event_metadata = {
        "tenantId": tenant_id,
        "eventType": "meeting.status.updated",
    }
    payload = {
        "meeting_id": meeting_id,
        "status": status,
        "platform": platform,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "metadata": metadata or {},
    }

    message = generate_message(event_metadata, payload)

    # Prefer caller-specified message id if provided.
    if message_id:
        message["metadata"]["messageId"] = message_id

    message["payload"]["message_id"] = message["metadata"]["messageId"]
    return message


def _dispatch(message: Dict[str, Any]) -> None:
    async def _run():
        try:
            await producer.send(MEETING_STATUS_QUEUE, message)
        except Exception:
            logger.exception("[MeetingStatusProducer] Failed to publish meeting status message")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_run())
    else:
        task = loop.create_task(_run())
        task.add_done_callback(_log_task_exception)


def _log_task_exception(task: asyncio.Task) -> None:
    if exc := task.exception():
        logger.exception("[MeetingStatusProducer] Async publish failed: %s", exc)