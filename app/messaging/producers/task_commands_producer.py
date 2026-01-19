# =============================
# FILE: app/messaging/producers/task_commands_producer.py
# PURPOSE:
#   Helper to publish task creation commands to RabbitMQ queue
#   'task-management.task-commands' for task management system.
#
# HOW TO USE:
#   1) Import `send_task_creation_command` in your orchestrator or processing flow.
#   2) Call it with meeting details and tasks list.
#   3) It generates a message_id and publishes to the correct queue.
# =============================

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.messaging.message_strcuture import generate_message
from app.messaging.producer import producer
from app.core.config import settings

logger = logging.getLogger(__name__)


def send_task_creation_command(
    *,
    meeting_id: str,
    tenant_id: str,
    meeting_title: str,
    meeting_date: str,
    tasks: List[Dict[str, Any]],
    message_id: Optional[str] = None,
) -> str:
    """
    Build and publish a task creation command event using the shared message envelope.

    Args:
        meeting_id: Unique identifier for the meeting
        tenant_id: Tenant identifier
        meeting_title: Title of the meeting
        meeting_date: Meeting date in ISO format
        tasks: List of task objects with title, description, assignees, priority, dueDate, tags
        message_id: Custom message ID for idempotency (optional)

    Returns:
        message_id (str) used for idempotency and tracing.
    """
    envelope = _build_envelope(
        meeting_id=meeting_id,
        tenant_id=tenant_id,
        meeting_title=meeting_title,
        meeting_date=meeting_date,
        tasks=tasks,
        message_id=message_id,
    )

    _dispatch(envelope)

    msg_id = envelope["metadata"]["messageId"]
    logger.info(
        "[TaskCommandsProducer] Published task creation command | meeting=%s tenant=%s tasks=%d msg=%s",
        meeting_id,
        tenant_id,
        len(tasks),
        msg_id,
    )

    return msg_id


def _build_envelope(
    *,
    meeting_id: str,
    tenant_id: str,
    meeting_title: str,
    meeting_date: str,
    tasks: List[Dict[str, Any]],
    message_id: Optional[str],
) -> Dict[str, Any]:
    event_metadata = {
        "tenantId": tenant_id,
        "eventType": "task.command.create",
    }

    payload = {
        "commandType": "create",
        "source": {
            "type": "meetings",
            "id": meeting_id,
            "metadata": {
                "meetingTitle": meeting_title,
                "meetingDate": meeting_date
            }
        },
        "payload": {
            "tasks": tasks
        }
    }

    message = generate_message(event_metadata, payload)

    # Prefer caller-specified message id if provided.
    if message_id:
        message["metadata"]["messageId"] = message_id

    return message


def _dispatch(message: Dict[str, Any]) -> None:
    async def _run():
        try:
            queue_name = settings.TASK_COMMANDS_QUEUE
            await producer.send(queue_name, message)
        except Exception:
            logger.exception("[TaskCommandsProducer] Failed to publish task creation command message")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_run())
    else:
        task = loop.create_task(_run())
        task.add_done_callback(_log_task_exception)


def _log_task_exception(task: asyncio.Task) -> None:
    if exc := task.exception():
        logger.exception("[TaskCommandsProducer] Async publish failed: %s", exc)
