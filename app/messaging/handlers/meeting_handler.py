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


async def _process_meeting_async(payload: Dict[str, Any]) -> None:
    """
    Process meeting asynchronously without holding the message.
    This prevents timeout issues by allowing early acknowledgment.

    Uses a semaphore to limit concurrent processing and prevent system overload.
    """
    processing_service = None

    # Acquire semaphore before processing to limit concurrency
    async with _processing_semaphore:
        try:
            meeting_id = payload.get('_id', 'unknown')
            concurrent_count = settings.MAX_CONCURRENT_MEETINGS - _processing_semaphore._value
            logger.info(
                f"Starting background processing for meeting {meeting_id} "
                f"(concurrent: {concurrent_count}/{settings.MAX_CONCURRENT_MEETINGS})"
            )

            # Initialize processing service
            processing_service = MeetingAnalysisOrchestrator()

            # Process the meeting
            result = await processing_service.analyze_meeting(payload)

            if result:
                logger.info(f"Background processing completed successfully for meeting {meeting_id}")
            else:
                logger.error(f"Background processing failed for meeting: {meeting_id}")

        except MeetingAlreadyProcessedException as e:
            logger.info(f"Meeting already processed or being processed: {e}")

        except MeetingAnalysisOrchestratorError as e:
            logger.error(f"Meeting processing service error: {e}")

        except Exception as e:
            logger.error(f"Unexpected error in background meeting processing: {e}", exc_info=True)

        finally:
            # Cleanup processing service if it was created
            if processing_service and hasattr(processing_service, 'transcription_service'):
                try:
                    await processing_service.transcription_service.cleanup()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup processing service: {cleanup_error}")


async def meeting_handler(message: AbstractIncomingMessage) -> None:
    """
    Handle meeting recording events from RabbitMQ.
    
    This handler validates and acknowledges messages quickly, then processes them
    asynchronously to prevent timeout issues.
    """
    processing_service = None
    try:
        logger.info("Received meeting processing message")
        
        # Parse message body
        try:
            message_body = message.body.decode('utf-8')
            event_data = json.loads(message_body)
            logger.info(f"Parsed meeting event data: {event_data.get('metadata', {}).get('eventType', 'Unknown')}")
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse message body: {e}")
            await _safe_reject(message, requeue=False)  # Don't requeue malformed messages
            return
        
        # Validate basic message structure
        if not _validate_message_structure(event_data):
            logger.error("Invalid message structure")
            await _safe_reject(message, requeue=False)
            return
        
        # Extract payload for processing
        payload = event_data.get('payload', {})
        meeting_id = payload.get('_id', 'unknown')
        
        # Acknowledge message immediately after validation to prevent timeout
        ack_success = await _safe_ack(message)
        if ack_success:
            logger.info(f"Acknowledged message for meeting {meeting_id}, starting background processing")
        else:
            logger.warning(f"Failed to acknowledge message for meeting {meeting_id}, but continuing with processing")
        
        # Start background processing without waiting for completion
        asyncio.create_task(_process_meeting_async(payload))
        
    except Exception as e:
        logger.error(f"Critical error in meeting handler: {e}", exc_info=True)
        await _safe_reject(message, requeue=False)  # Don't requeue critical handler errors


def _validate_message_structure(event_data: Dict[str, Any]) -> bool:
    """
    Validate the basic structure of the meeting event message.
    
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
        payload = event_data["payload"]
        
        # Check required metadata fields
        required_metadata = ["tenantId", "eventType"]
        if not all(field in metadata for field in required_metadata):
            logger.error(f"Missing required metadata fields. Required: {required_metadata}")
            return False
        
        # Check required payload fields - simplified for generic use
        required_payload = ["_id"]
        if not all(field in payload for field in required_payload):
            logger.error(f"Missing required payload fields. Required: {required_payload}")
            return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error validating message structure: {e}")
        return False


def _is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable.
    
    Args:
        error: The exception that occurred
        
    Returns:
        True if the error should trigger a retry
    """
    # Convert error to string for pattern matching
    error_str = str(error).lower()
    
    # Non-retryable errors (permanent failures)
    non_retryable_patterns = [
        "validation",
        "invalid event",
        "no adapter found",
        "file not found",
        "missing required",
        "malformed",
        "parse error",
        "duplicate key error",
        "e11000",
        "already processed",
        "currently being processed"
    ]
    
    for pattern in non_retryable_patterns:
        if pattern in error_str:
            return False
    
    # Retryable errors (temporary failures)
    retryable_patterns = [
        "connection",
        "timeout",
        "network",
        "database",
        "s3",
        "temporary",
        "unavailable",
        "rate limit"
    ]
    
    for pattern in retryable_patterns:
        if pattern in error_str:
            return True
    
    # Default to retryable for unknown errors
    return True


def _extract_meeting_info(event_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract basic meeting information for logging.
    
    Args:
        event_data: The parsed JSON event data
        
    Returns:
        Dictionary with meeting information
    """
    try:
        metadata = event_data.get("metadata", {})
        payload = event_data.get("payload", {})
        
        return {
            "tenant_id": metadata.get("tenantId") or payload.get("tenantId", "unknown"),
            "meeting_id": str(payload.get("_id", "unknown")),
            "event_type": metadata.get("eventType", "unknown"),
            "platform": metadata.get("source", "unknown")
        }
    except Exception:
        return {
            "tenant_id": "unknown",
            "meeting_id": "unknown", 
            "event_type": "unknown",
            "platform": "unknown"
        }