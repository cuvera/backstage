# =============================
# FILE: app/messaging/producers/transcription_v2_producer.py
# PURPOSE:
#   Producer to publish Transcription V2 completion events to RabbitMQ.
#   Sends minimal payload with segment classifications for downstream processing.
#
# HOW TO USE:
#   1) Import `send_transcription_v2_ready` in your V2 service
#   2) Call it with meeting_id, tenant_id, platform, and segment_classifications
#   3) It generates a message_id and publishes to the V2 queue
# =============================

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from app.messaging.message_strcuture import generate_message
from app.messaging.producer import producer

logger = logging.getLogger(__name__)

# Queue name for V2 transcription completion events
TRANSCRIPTION_V2_QUEUE = "transcription.v2.ready"


def send_transcription_v2_ready(
    *,
    meeting_id: str,
    tenant_id: str,
    platform: str,
    status: str,
    segment_classifications: Optional[Dict[str, Any]] = None,
    processing_stats: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    message_id: Optional[str] = None,
) -> str:
    """
    Build and publish a Transcription V2 completion event.

    Args:
        meeting_id: Unique identifier for the meeting
        tenant_id: Tenant identifier
        platform: Platform type (e.g., "offline", "online")
        status: Processing status (completed, failed)
        segment_classifications: Classified segment clusters (optional)
        processing_stats: Processing statistics (optional)
        error: Error message if status is failed (optional)
        message_id: Custom message ID for idempotency (optional)

    Returns:
        message_id (str) used for idempotency and tracing
    """
    envelope = _build_envelope(
        meeting_id=meeting_id,
        tenant_id=tenant_id,
        platform=platform,
        status=status,
        segment_classifications=segment_classifications,
        processing_stats=processing_stats,
        error=error,
        message_id=message_id,
    )

    _dispatch(envelope)

    msg_id = envelope["payload"]["message_id"]
    logger.info(
        "[TranscriptionV2Producer] Published V2 completion | meeting=%s tenant=%s "
        "platform=%s status=%s msg=%s",
        meeting_id,
        tenant_id,
        platform,
        status,
        msg_id,
    )

    return msg_id


def _build_envelope(
    *,
    meeting_id: str,
    tenant_id: str,
    platform: str,
    status: str,
    segment_classifications: Optional[Dict[str, Any]],
    processing_stats: Optional[Dict[str, Any]],
    error: Optional[str],
    message_id: Optional[str],
) -> Dict[str, Any]:
    """Build message envelope for V2 transcription event."""

    event_metadata = {
        "tenantId": tenant_id,
        "eventType": "transcription.v2.ready",
    }

    payload = {
        "meeting_id": meeting_id,
        "tenant_id": tenant_id,
        "platform": platform,
        "status": status,
    }

    # Add optional fields if present
    if segment_classifications is not None:
        payload["segment_classifications"] = segment_classifications

    if processing_stats is not None:
        payload["processing_stats"] = processing_stats

    if error is not None:
        payload["error"] = error

    message = generate_message(event_metadata, payload)

    # Prefer caller-specified message id if provided
    if message_id:
        message["metadata"]["messageId"] = message_id

    message["payload"]["message_id"] = message["metadata"]["messageId"]
    return message


def _dispatch(message: Dict[str, Any]) -> None:
    """Dispatch message to RabbitMQ with async handling."""

    async def _run():
        try:
            await producer.send(TRANSCRIPTION_V2_QUEUE, message)
        except Exception:
            logger.exception(
                "[TranscriptionV2Producer] Failed to publish V2 transcription message"
            )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_run())
    else:
        task = loop.create_task(_run())
        task.add_done_callback(_log_task_exception)


def _log_task_exception(task: asyncio.Task) -> None:
    """Log any exceptions from async dispatch task."""
    if exc := task.exception():
        logger.exception("[TranscriptionV2Producer] Async publish failed: %s", exc)
