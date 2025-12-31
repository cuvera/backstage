import asyncio
import json
import logging
from typing import Any, Dict

from aio_pika.abc import AbstractIncomingMessage
from aiormq.exceptions import ChannelInvalidStateError

from app.core.config import settings
from app.services.meeting_analysis_orchestrator import (
    MeetingAnalysisOrchestrator,
    MeetingAnalysisOrchestratorError,
    MeetingAlreadyProcessedException
)

logger = logging.getLogger(__name__)

# Global semaphore to limit concurrent meeting processing
_processing_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_MEETINGS)


async def _safe_ack(message: AbstractIncomingMessage) -> bool:
    """
    Safely acknowledge a message, handling channel invalid state errors.

    Returns:
        True if acknowledgment was successful, False if channel is invalid
    """
    try:
        await message.ack()
        return True
    except ChannelInvalidStateError:
        logger.warning("Cannot acknowledge message - channel is invalid (connection likely closed)")
        return False


async def _safe_reject(message: AbstractIncomingMessage, requeue: bool = False) -> bool:
    """
    Safely reject a message, handling channel invalid state errors.

    Returns:
        True if rejection was successful, False if channel is invalid
    """
    try:
        await message.reject(requeue=requeue)
        return True
    except ChannelInvalidStateError:
        logger.warning("Cannot reject message - channel is invalid (connection likely closed)")
        return False


def _transform_recorder_to_meeting_payload(recorder_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform recorder event payload to MeetingAnalysisOrchestrator expected format.

    Recorder payload → Meeting payload mapping:
    - meeting_id → _id
    - tenant_id → tenantId
    - audio_url → fileUrl (S3 key)
    - metadata.source_platform or default → platform ("offline")
    - settings or default → bucket

    Args:
        recorder_payload: Recorder event payload containing:
            - meeting_id: Meeting identifier
            - tenant_id: Tenant identifier
            - audio_url: S3 key path to audio file
            - file_name: Original filename
            - content_type: Audio content type
            - size: File size in bytes
            - duration_ms: Duration in milliseconds
            - status: Processing status
            - metadata: Additional metadata

    Returns:
        Transformed payload compatible with MeetingAnalysisOrchestrator
    """

    print(f"recorder_payload: {recorder_payload}")

    metadata = recorder_payload.get("metadata", {})

    # Determine platform from metadata or default to offline
    platform = metadata.get("platform", "offline")

    # Determine bucket from settings
    bucket = recorder_payload.get("bucket")

    transformed_payload = {
        "_id": recorder_payload.get("meeting_id"),
        "tenantId": recorder_payload.get("tenant_id"),
        "platform": platform,
        "bucket": bucket,
        "fileUrl": recorder_payload.get("audio_url"),
    }

    logger.info(
        f"Transformed recorder payload - meeting_id={transformed_payload['_id']}, "
        f"platform={platform}, audio_url={recorder_payload.get('audio_url')}"
    )

    return transformed_payload


async def _process_recorder_async(payload: Dict[str, Any]) -> None:
    """
    Process recorder completion event asynchronously without holding the message.
    This prevents timeout issues by allowing early acknowledgment.

    Transforms the recorder payload to meeting orchestrator format and delegates
    to MeetingAnalysisOrchestrator for full processing pipeline.

    Uses a semaphore to limit concurrent processing and prevent system overload.
    """
    processing_service = None

    # Acquire semaphore before processing to limit concurrency
    async with _processing_semaphore:
        try:
            meeting_id = payload.get('meeting_id', 'unknown')
            concurrent_count = settings.MAX_CONCURRENT_MEETINGS - _processing_semaphore._value
            logger.info(
                f"Starting background processing for recorder event meeting {meeting_id} "
                f"(concurrent: {concurrent_count}/{settings.MAX_CONCURRENT_MEETINGS})"
            )

            # Transform recorder payload to orchestrator format
            orchestrator_payload = _transform_recorder_to_meeting_payload(payload)

            # Initialize processing service
            processing_service = MeetingAnalysisOrchestrator()

            # Process the meeting using existing orchestrator
            result = await processing_service.analyze_meeting(orchestrator_payload)

            if result:
                logger.info(f"Background processing completed successfully for recorder event meeting {meeting_id}")
            else:
                logger.error(f"Background processing failed for recorder event meeting: {meeting_id}")

        except MeetingAlreadyProcessedException as e:
            logger.info(f"Meeting already processed or being processed (recorder event): {e}")

        except MeetingAnalysisOrchestratorError as e:
            logger.error(f"Meeting processing service error (recorder event): {e}")

        except Exception as e:
            logger.error(f"Unexpected error in background recorder event processing: {e}", exc_info=True)

        finally:
            # Cleanup processing service if it was created
            if processing_service and hasattr(processing_service, 'transcription_service'):
                try:
                    await processing_service.transcription_service.cleanup()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup processing service: {cleanup_error}")


async def recorder_completed_handler(message: AbstractIncomingMessage) -> None:
    """
    Handle meeting recorder completion events from RabbitMQ.

    This handler validates recorder-specific payloads, transforms them to the format
    expected by MeetingAnalysisOrchestrator, and processes them asynchronously to
    prevent timeout issues.

    Expected message structure:
    {
        "metadata": {
            "tenantId": "...",
            "eventType": "meeting.recorder.completed",
            ...
        },
        "payload": {
            "meeting_id": "...",
            "tenant_id": "...",
            "user_id": "...",
            "audio_url": "s3-key-path",
            "file_name": "recording.m4a",
            "content_type": "audio/m4a",
            "size": 1048576,
            "duration_ms": 3600000,
            "status": "pending",
            "metadata": {
                "title": "Meeting Title",
                "description": "Meeting Description",
                "source": "meeting-recording-service"
            }
        }
    }
    """
    processing_service = None
    try:
        logger.info("Received recorder completion message")

        # Parse message body
        try:
            message_body = message.body.decode('utf-8')
            event_data = json.loads(message_body)
            logger.info(f"Parsed recorder event data: {event_data.get('metadata', {}).get('eventType', 'Unknown')}")
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse message body: {e}")
            await _safe_reject(message, requeue=False)  # Don't requeue malformed messages
            return

        # Validate basic message structure
        if not _validate_recorder_message_structure(event_data):
            logger.error("Invalid recorder message structure")
            await _safe_reject(message, requeue=False)
            return

        # Extract payload for processing
        payload = event_data.get('payload', {})
        meeting_id = payload.get('meeting_id', 'unknown')

        # Validate recorder-specific payload
        if not _validate_recorder_payload(payload):
            logger.error(f"Invalid recorder payload for meeting {meeting_id}")
            await _safe_reject(message, requeue=False)
            return

        # Acknowledge message immediately after validation to prevent timeout
        ack_success = await _safe_ack(message)
        if ack_success:
            logger.info(f"Acknowledged recorder message for meeting {meeting_id}, starting background processing")
        else:
            logger.warning(f"Failed to acknowledge recorder message for meeting {meeting_id}, but continuing with processing")

        # Start background processing without waiting for completion
        asyncio.create_task(_process_recorder_async(payload))

    except Exception as e:
        logger.error(f"Critical error in recorder completion handler: {e}", exc_info=True)
        await _safe_reject(message, requeue=False)  # Don't requeue critical handler errors


def _validate_recorder_message_structure(event_data: Dict[str, Any]) -> bool:
    """
    Validate the basic structure of the recorder event message.

    Args:
        event_data: The parsed JSON event data

    Returns:
        True if the message has the required basic structure
    """
    try:
        # Check for required top-level fields
        required_fields = ["metadata", "payload"]
        if not all(field in event_data for field in required_fields):
            logger.error(f"Missing required top-level fields. Required: {required_fields}")
            return False

        metadata = event_data["metadata"]

        # Check required metadata fields
        required_metadata = ["tenantId", "eventType"]
        if not all(field in metadata for field in required_metadata):
            logger.error(f"Missing required metadata fields. Required: {required_metadata}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating recorder message structure: {e}")
        return False


def _validate_recorder_payload(payload: Dict[str, Any]) -> bool:
    """
    Validate recorder-specific payload structure.

    Args:
        payload: The recorder event payload

    Returns:
        True if the payload has all required recorder-specific fields
    """
    try:
        # Check required recorder payload fields
        required_fields = ["meeting_id", "tenant_id", "audio_url"]

        if not all(field in payload for field in required_fields):
            missing = [f for f in required_fields if f not in payload]
            logger.error(f"Missing required recorder payload fields: {missing}")
            return False

        # Validate that required fields are not empty
        for field in required_fields:
            if not payload.get(field):
                logger.error(f"Required field '{field}' is empty")
                return False

        return True

    except Exception as e:
        logger.error(f"Error validating recorder payload: {e}")
        return False
