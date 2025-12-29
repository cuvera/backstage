import json
import logging
import asyncio
from typing import Dict, Any
from aio_pika.abc import AbstractIncomingMessage
from aiormq.exceptions import ChannelInvalidStateError

from app.services.meeting_embedding_service import MeetingEmbeddingService

logger = logging.getLogger(__name__)

async def _safe_ack(message: AbstractIncomingMessage) -> bool:
    try:
        await message.ack()
        return True
    except ChannelInvalidStateError:
        logger.warning("Cannot acknowledge message - channel is invalid")
        return False

async def _safe_reject(message: AbstractIncomingMessage, requeue: bool = False) -> bool:
    try:
        await message.reject(requeue=requeue)
        return True
    except ChannelInvalidStateError:
        logger.warning("Cannot reject message - channel is invalid")
        return False

async def _process_embedding_async(payload: Dict[str, Any], tenant_id: str) -> None:
    """
    Background task to process embeddings.
    Fetches transcript and metadata from the database.
    """
    try:
        meeting_id = payload.get("meetingId") or payload.get("meeting_id") or payload.get("_id")
        
        if not meeting_id:
            logger.error(f"[MeetingEmbeddingHandler] Missing meeting_id in payload")
            return

        embedding_service = MeetingEmbeddingService()
        await embedding_service.process_meeting_for_rag(
            meeting_id=str(meeting_id),
            tenant_id=tenant_id
        )
        logger.info(f"[MeetingEmbeddingHandler] Successfully processed embeddings for meeting {meeting_id}")

    except Exception as e:
        logger.error(f"[MeetingEmbeddingHandler] Error processing embedding: {e}", exc_info=True)

async def meeting_embedding_handler(message: AbstractIncomingMessage) -> None:
    """
    Consumes 'meeting.analysis.ready_for_embedding' events.
    """
    try:
        message_body = message.body.decode('utf-8')
        event_data = json.loads(message_body)
        
        metadata = event_data.get("metadata", {})
        payload = event_data.get("payload", {})
        tenant_id = metadata.get("tenantId") or payload.get("tenantId")
        
        if not tenant_id:
            logger.error("[MeetingEmbeddingHandler] Missing tenantId in message")
            await _safe_reject(message, requeue=False)
            return

        # Ack immediately to prevent RabbitMQ timeout during heavy embedding work
        await _safe_ack(message)
        
        # Process in background
        asyncio.create_task(_process_embedding_async(payload, tenant_id))
        
    except Exception as e:
        logger.error(f"[MeetingEmbeddingHandler] Failed to handle embedding message: {e}")
        await _safe_reject(message, requeue=False)
