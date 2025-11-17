import asyncio
import json
import logging
from typing import Any, Dict

from aio_pika.abc import AbstractIncomingMessage
from aiormq.exceptions import ChannelInvalidStateError

from app.services.meeting_analysis_orchestrator import (
    MeetingAnalysisOrchestrator,
    MeetingAnalysisOrchestratorError,
    MeetingAlreadyProcessedException
)
from app.repository import MeetingMetadataRepository

logger = logging.getLogger(__name__)


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


async def _fetch_and_validate_meeting(meeting_id: str, tenant_id: str) -> Dict[str, Any]:
    """
    Fetch meeting data from backstage collections and validate retry conditions.
    
    Args:
        meeting_id: The meeting ID to fetch
        tenant_id: The tenant ID for validation
        
    Returns:
        Meeting data dictionary
        
    Raises:
        ValueError: If meeting not found or cannot be retried
    """
    # Initialize repository
    meeting_repo = await MeetingMetadataRepository.from_default()
    
    # Try Google meetings first, then offline
    meeting_data = {}
    
    meetings = await meeting_repo.get_meeting_metadata(meeting_id, "google")
    if meetings:
        # Check if meeting can be retried
        id = meetings.get("id")
        file_url = meetings.get("file_url")
        
        # Ensure required fields exist for processing
        meeting_data["_id"] = id
        meeting_data["tenantId"] = tenant_id
        meeting_data["platform"] = "google"
        meeting_data["fileUrl"] = file_url
        return meeting_data
    else:
        meeting_data["_id"] = meeting_id
        meeting_data["tenantId"] = tenant_id
        meeting_data["platform"] = "offline"
        return meeting_data

async def _process_meeting_retry_async(payload: Dict[str, Any]) -> None:
    """
    Process meeting retry asynchronously without holding the message.
    This prevents timeout issues by allowing early acknowledgment.
    """
    orchestrator = None
    try:
        meeting_id = payload.get("meeting_id")
        tenant_id = payload.get("tenant_id")
        
        logger.info(f"Starting retry processing for meeting {meeting_id}")
        
        # 1. Fetch and validate meeting data from backstage collections
        meeting_data = await _fetch_and_validate_meeting(meeting_id, tenant_id)
        print("meeting_data: ", meeting_data)
        
        # 2. Initialize orchestrator and update status to processing
        orchestrator = MeetingAnalysisOrchestrator()
        platform = meeting_data.get("platform", "google")
        await orchestrator._update_meeting_status(meeting_id, platform, 'analysing')
        
        # 3. Call existing analyze_meeting method
        result = await orchestrator.analyze_meeting(meeting_data)
        
        if result.get("success"):
            logger.info(f"Retry processing completed successfully for meeting {meeting_id}")
        else:
            logger.error(f"Retry processing failed for meeting {meeting_id}: {result.get('error')}")
            
    except ValueError as e:
        # Validation errors (meeting not found, can't be retried, etc.)
        logger.error(f"Retry validation failed for meeting {payload.get('meeting_id')}: {e}")
        # Don't update status for validation errors - leave as is
        
    except MeetingAlreadyProcessedException as e:
        logger.info(f"Meeting already processed during retry: {e}")
        
    except MeetingAnalysisOrchestratorError as e:
        logger.error(f"Meeting retry processing service error: {e}")
        # Update status to failed
        if orchestrator:
            try:
                meeting_id = payload.get("meeting_id")
                platform = meeting_data.get("platform", "google") if 'meeting_data' in locals() else "google"
                await orchestrator._update_meeting_status(meeting_id, platform, 'failed')
            except Exception as status_error:
                logger.error(f"Failed to update meeting status after retry error: {status_error}")
        
    except Exception as e:
        logger.error(f"Unexpected error in retry processing for meeting {payload.get('meeting_id')}: {e}", exc_info=True)
        # Update status to failed
        if orchestrator:
            try:
                meeting_id = payload.get("meeting_id")
                platform = meeting_data.get("platform", "google") if 'meeting_data' in locals() else "google"
                await orchestrator._update_meeting_status(meeting_id, platform, 'failed')
            except Exception as status_error:
                logger.error(f"Failed to update meeting status after unexpected error: {status_error}")


async def meeting_retry_handler(message: AbstractIncomingMessage) -> None:
    """
    Handle meeting analysis retry requests from cognitive-service.
    
    This handler validates and acknowledges messages quickly, then processes them
    asynchronously to prevent timeout issues.
    """
    try:
        logger.info("Received meeting retry message")
        
        # Parse message body
        try:
            message_body = message.body.decode('utf-8')
            event_data = json.loads(message_body)
            logger.info(f"Parsed retry event data: {event_data.get('metadata', {}).get('eventType', 'Unknown')}")
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse retry message body: {e}")
            await _safe_reject(message, requeue=False)  # Don't requeue malformed messages
            return
        
        # Validate basic message structure
        if not _validate_retry_message_structure(event_data):
            logger.error("Invalid retry message structure")
            await _safe_reject(message, requeue=False)
            return
        
        # Extract payload for processing
        payload = event_data.get('payload', {})
        meeting_id = payload.get('meeting_id', '')
        
        # Acknowledge message immediately after validation to prevent timeout
        ack_success = await _safe_ack(message)
        if ack_success:
            logger.info(f"Acknowledged retry message for meeting {meeting_id}, starting background processing")
        else:
            logger.warning(f"Failed to acknowledge retry message for meeting {meeting_id}, but continuing with processing")
        
        # Start background processing without waiting for completion
        asyncio.create_task(_process_meeting_retry_async(payload))
        
    except Exception as e:
        logger.error(f"Critical error in meeting retry handler: {e}", exc_info=True)
        await _safe_reject(message, requeue=False)  # Don't requeue critical handler errors


def _validate_retry_message_structure(event_data: Dict[str, Any]) -> bool:
    """
    Validate the basic structure of the meeting retry message.
    
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
        
        # Check required payload fields for retry
        required_payload = ["meeting_id", "tenant_id"]
        if not all(field in payload for field in required_payload):
            logger.error(f"Missing required retry payload fields. Required: {required_payload}")
            return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error validating retry message structure: {e}")
        return False