"""
Transcription Producer - Publishes transcription events to RabbitMQ

Publishes to exchange: audio.transcription
Routing keys:
  - transcription.v1 (V1 transcription complete)
  - transcription.v2 (V2 transcription complete)
  - transcription.status (failure events)
"""

import logging
import uuid
from typing import Dict, Any, List

from app.messaging.producer import producer

logger = logging.getLogger(__name__)

EXCHANGE_NAME = "transcription.exchange"
ROUTING_KEY_V1 = "transcription.v1"
ROUTING_KEY_V2 = "transcription.v2"
ROUTING_KEY_STATUS = "transcription.status"

async def publish_v1(
    meeting_id: str,
    tenant_id: str,
    platform: str,
    mode: str,
    transcriptions: List[Dict[str, Any]],
    speakers: List[Dict[str, Any]],
    metadata: Dict[str, Any]
) -> str:
    """
    Publish V1 transcription completion event.

    Args:
        meeting_id: Meeting identifier
        tenant_id: Tenant identifier
        platform: Platform (google, zoom, etc.)
        mode: online or offline
        transcriptions: List of transcription segments
        speakers: List of speaker summaries
        metadata: Transcription metadata

    Returns:
        message_id for tracking
    """
    message_id = str(uuid.uuid4())

    message = {
        "metadata": {
            "tenantId": tenant_id,
            "messageId": message_id
        },
        "payload": {
            "type": "meeting",
            "mode": mode,
            "platform": platform,
            "id": meeting_id,
            "tenantId": tenant_id,
            "transcriptions": transcriptions,
            "speakers": speakers,
            "metadata": metadata
        }
    }

    await producer.publish_to_exchange(
        exchange_name=EXCHANGE_NAME,
        routing_key=ROUTING_KEY_V1,
        message=message
    )

    logger.info(
        f"[TranscriptionProducer] Published V1 | meeting={meeting_id} tenant={tenant_id} "
        f"mode={mode} segments={len(transcriptions)} msg={message_id}"
    )

    return message_id


async def publish_v2(
    meeting_id: str,
    tenant_id: str,
    platform: str,
    mode: str,
    transcription_v2: Dict[str, Any]
) -> str:
    """
    Publish V2 transcription completion event.

    Args:
        meeting_id: Meeting identifier
        tenant_id: Tenant identifier
        platform: Platform (google, zoom, etc.)
        mode: online or offline
        transcription_v2: V2 transcription data with segments

    Returns:
        message_id for tracking
    """
    message_id = str(uuid.uuid4())

    # Extract segments from transcription_v2
    segments = transcription_v2.get("segments", [])
    metadata = transcription_v2.get("metadata", {})

    message = {
        "metadata": {
            "tenantId": tenant_id,
            "messageId": message_id
        },
        "payload": {
            "type": "meeting",
            "mode": mode,
            "platform": platform,
            "id": meeting_id,
            "tenantId": tenant_id,
            "segments": segments,
            "metadata": metadata
        }
    }

    await producer.publish_to_exchange(
        exchange_name=EXCHANGE_NAME,
        routing_key=ROUTING_KEY_V2,
        message=message
    )

    logger.info(
        f"[TranscriptionProducer] Published V2 | meeting={meeting_id} tenant={tenant_id} "
        f"mode={mode} clusters={len(segments)} msg={message_id}"
    )

    return message_id


async def publish_status_failure(
    meeting_id: str,
    tenant_id: str,
    platform: str,
    mode: str
) -> str:
    """
    Publish transcription failure event.

    Args:
        meeting_id: Meeting identifier
        tenant_id: Tenant identifier
        platform: Platform (google, zoom, etc.)
        mode: online or offline

    Returns:
        message_id for tracking
    """
    message_id = str(uuid.uuid4())

    message = {
        "metadata": {
            "tenantId": tenant_id,
            "messageId": message_id
        },
        "payload": {
            "type": "meeting",
            "mode": mode,
            "platform": platform,
            "id": meeting_id,
            "tenantId": tenant_id,
            "status": "failed"
        }
    }

    await producer.publish_to_exchange(
        exchange_name=EXCHANGE_NAME,
        routing_key=ROUTING_KEY_STATUS,
        message=message
    )

    logger.info(
        f"[TranscriptionProducer] Published FAILURE | meeting={meeting_id} tenant={tenant_id} "
        f"mode={mode} msg={message_id}"
    )

    return message_id
