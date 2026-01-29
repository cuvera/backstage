"""
Transcription Event Handler
Consumes meeting.completed and interview.completed events from transcription.events exchange
Centralized transcription service for the platform
"""
import json
import logging
from aio_pika.abc import AbstractIncomingMessage

from app.services.transcription_orchestrator import TranscriptionOrchestrator, AudioTranscriptionError

logger = logging.getLogger(__name__)


async def transcription_handler(message: AbstractIncomingMessage) -> None:
    """
    Handle transcription events from meeting and interview services

    Routing keys: meeting.completed, interview.completed

    Message structure:
    {
        "metadata": {
            "tenantId": "...",
            "messageId": "..."
        },
        "payload": {
            "type": "meeting" | "interview",
            "mode": "online" | "offline",
            "platform": "google" | "zoom" | ...,
            "id": "...",
            "title": "...",
            "audioUrl": "...",
            "participants": [...],
            "speakerTimeframes": [...],
            ...
        }
    }

    Args:
        message: RabbitMQ message
    """
    message_body = None
    metadata = {}

    try:
        async with message.process():
            # Parse message body
            message_body = json.loads(message.body.decode())

            # Extract metadata and payload
            metadata = message_body.get("metadata", {})
            payload = message_body.get("payload", {})

            message_id = metadata.get("messageId", "unknown")
            tenant_id = metadata.get("tenantId") or payload.get("tenantId")
            content_type = payload.get("type")  # "meeting" or "interview"
            content_id = payload.get("id")
            routing_key = message.routing_key

            logger.info(
                f"[TranscriptionHandler] Received event | "
                f"routing_key={routing_key} message_id={message_id} "
                f"type={content_type} id={content_id} tenant_id={tenant_id}"
            )

            # Validate required fields
            if not payload:
                logger.error(f"[TranscriptionHandler] No payload | message_id={message_id}")
                return

            if not content_type:
                logger.error(f"[TranscriptionHandler] No type in payload | message_id={message_id}")
                return

            if content_type not in ["meeting", "interview"]:
                logger.warning(
                    f"[TranscriptionHandler] Unexpected type '{content_type}' | "
                    f"message_id={message_id} id={content_id} - Skipping"
                )
                return

            if not content_id:
                logger.error(f"[TranscriptionHandler] No id in payload | message_id={message_id}")
                return

            if not payload.get("audioUrl"):
                logger.error(
                    f"[TranscriptionHandler] No audioUrl | "
                    f"message_id={message_id} id={content_id} type={content_type}"
                )
                return

            # Process transcription (same pipeline for both meetings and interviews)
            logger.info(
                f"[TranscriptionHandler] Starting transcription | "
                f"type={content_type} id={content_id} tenant_id={tenant_id}"
            )

            orchestrator = TranscriptionOrchestrator()
            await orchestrator.transcribe_audio(payload)

            logger.info(
                f"[TranscriptionHandler] Successfully completed | "
                f"type={content_type} message_id={message_id} id={content_id}"
            )

    except json.JSONDecodeError as e:
        logger.error(
            f"[TranscriptionHandler] Failed to decode JSON | "
            f"error={e} body={message.body.decode()[:200]}"
        )
        # Reject malformed messages (don't requeue)
        await message.reject(requeue=False)

    except AudioTranscriptionError as e:
        logger.error(
            f"[TranscriptionHandler] Transcription failed | "
            f"error={e} message_id={metadata.get('messageId', 'unknown')}"
        )
        # Don't requeue - transcription errors are usually permanent
        # (invalid audio, missing file, processing errors, etc.)

    except Exception as e:
        logger.exception(
            f"[TranscriptionHandler] Unexpected error | "
            f"error={e} message_id={metadata.get('messageId', 'unknown') if message_body else 'unknown'}"
        )
        # Requeue for transient errors (network, temporary unavailability)
        await message.reject(requeue=True)
