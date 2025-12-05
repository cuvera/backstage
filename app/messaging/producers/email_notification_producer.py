# =============================
# FILE: app/messaging/producers/email_notification_producer.py
# PURPOSE:
#   Small helper to publish an 'email.notification' message
#   from your meeting processing pipeline to RabbitMQ.
#
# HOW TO USE:
#   1) Import `send_email_notification` in your meeting processing flow.
#   2) Call it with the meeting notification payload.
#   3) It generates a message_id and publishes to the correct routing key.
# =============================

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.messaging.handlers.settings import EMAIL_NOTIFICATIONS_ROUTING_KEY
from app.messaging.message_strcuture import generate_message
from app.messaging.producer import producer

logger = logging.getLogger(__name__)


def send_email_notification(
    *,
    attendees: List[Dict[str, str]],
    organizer: Dict[str, str],
    title: str,
    startTime: str,
    endTime: str,
    duration: str,
    summary: str,
    redirectUrl: str,
    noOfKeyPoints: int,
    noOfActionItems: int,
    tenant_id: str,
    message_id: Optional[str] = None,
) -> str:
    """
    Build and publish an email.notification event using the shared message envelope.

    Args:
        attendees: List of attendee objects with name and email
        organizer: Organizer object with name and email
        title: Meeting title
        startTime: Meeting start time
        endTime: Meeting end time
        summary: Meeting summary
        redirectUrl: URL to redirect to
        noOfKeyPoints: Number of key points
        noOfActionItems: Number of action items
        tenant_id: Tenant identifier
        message_id: Optional message ID for idempotency

    Returns:
        message_id (str) used for idempotency and tracing.
    """
    envelope = _build_envelope(
        attendees=attendees,
        organizer=organizer,
        title=title,
        startTime=startTime,
        endTime=endTime,
        duration=duration,
        summary=summary,
        redirectUrl=redirectUrl,
        noOfKeyPoints=noOfKeyPoints,
        noOfActionItems=noOfActionItems,
        tenant_id=tenant_id,
        message_id=message_id,
    )

    _dispatch(envelope)

    msg_id = envelope["payload"]["message_id"]
    logger.info(
        "[EmailNotificationProducer] Published email.notification | tenant=%s title=%s msg=%s",
        tenant_id,
        title,
        msg_id,
    )

    return msg_id


def _build_envelope(
    *,
    attendees: List[Dict[str, str]],
    organizer: Dict[str, str],
    title: str,
    startTime: str,
    endTime: str,
    duration: str,
    summary: str,
    redirectUrl: str,
    noOfKeyPoints: int,
    noOfActionItems: int,
    tenant_id: str,
    message_id: Optional[str],
) -> Dict[str, Any]:
    event_metadata = {
        "tenantId": tenant_id,
        "eventType": EMAIL_NOTIFICATIONS_ROUTING_KEY,
    }
    payload = {
        "attendees": attendees,
        "organizer": organizer,
        "title": title,
        "startTime": startTime,
        "endTime": endTime,
        "duration": duration,
        "summary": summary,
        "redirectUrl": redirectUrl,
        "noOfKeyPoints": noOfKeyPoints,
        "noOfActionItems": noOfActionItems,
    }

    message = generate_message(event_metadata, payload)

    # Prefer caller-specified message id if provided.
    if message_id:
        message["metadata"]["messageId"] = message_id

    message["payload"]["message_id"] = message["metadata"]["messageId"]
    return message


async def _publish(message: Dict[str, Any]) -> None:
    await producer.connect()
    await producer.send(EMAIL_NOTIFICATIONS_ROUTING_KEY, message)


def _dispatch(message: Dict[str, Any]) -> None:
    async def _run():
        try:
            await _publish(message)
        except Exception:
            logger.exception("[EmailNotificationProducer] Failed to publish email notification message")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_run())
    else:
        task = loop.create_task(_run())
        task.add_done_callback(_log_task_exception)


def _log_task_exception(task: asyncio.Task) -> None:
    if exc := task.exception():
        logger.exception("[EmailNotificationProducer] Async publish failed: %s", exc)