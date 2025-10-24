import json
import logging
from typing import Any, Dict

from aio_pika.abc import AbstractIncomingMessage

from app.services.meeting_processing_service import (
    MeetingProcessingService, 
    MeetingProcessingServiceError,
    MeetingAlreadyProcessedException
)

logger = logging.getLogger(__name__)


async def meeting_handler(message: AbstractIncomingMessage) -> None:
    """
    Handle meeting recording events from RabbitMQ.
    
    This handler processes meeting recording events and triggers the complete
    processing pipeline including audio merging, transcription, and analysis.
    """
    try:
        logger.info("Received meeting processing message")
        
        # Parse message body
        try:
            message_body = message.body.decode('utf-8')
            event_data = json.loads(message_body)
            logger.info(f"Parsed meeting event data: {event_data.get('metadata', {}).get('eventType', 'Unknown')}")
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse message body: {e}")
            await message.reject(requeue=False)  # Don't requeue malformed messages
            return
        
        # Validate basic message structure
        if not _validate_message_structure(event_data):
            logger.error("Invalid message structure")
            await message.reject(requeue=False)
            return
        
        # Initialize processing service
        processing_service = MeetingProcessingService()
        
        # Process the meeting event directly with raw payload
        try:
            result = await processing_service.process_meeting_event(event_data)
            
            if result["success"]:
                logger.info(f"Successfully processed meeting {result['meeting_id']}")
                await message.ack()
            else:
                logger.error(f"Meeting processing failed: {result.get('error')}")
                await message.reject(requeue=False)  # Requeue for retry
                
        except MeetingAlreadyProcessedException as e:
            logger.info(f"Meeting already processed or being processed: {e}")
            await message.ack()  # Acknowledge message as successfully handled
            
        except MeetingProcessingServiceError as e:
            logger.error(f"Meeting processing service error: {e}")
            
            # Check if this is a retryable error
            if _is_retryable_error(e):
                logger.info("Error is retryable, requeuing message")
                await message.reject(requeue=False)
            else:
                logger.info("Error is not retryable, rejecting message")
                await message.reject(requeue=False)
                
        except Exception as e:
            logger.error(f"Unexpected error in meeting processing: {e}", exc_info=True)
            await message.reject(requeue=False)  # Requeue unexpected errors for retry
    
    except Exception as e:
        logger.error(f"Critical error in meeting handler: {e}", exc_info=True)
        try:
            await message.reject(requeue=False)  # Don't requeue critical handler errors
        except Exception as reject_error:
            logger.error(f"Failed to reject message: {reject_error}")
    
    finally:
        # Cleanup processing service if it was created
        if processing_service and hasattr(processing_service, 'transcription_service'):
            try:
                await processing_service.transcription_service.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup processing service: {cleanup_error}")


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