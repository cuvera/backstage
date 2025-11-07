from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.meeting_prep_service import MeetingPrepService, MeetingPrepServiceError

logger = logging.getLogger(__name__)

router = APIRouter()


class GeneratePrepPackRequest(BaseModel):
    """Request model for generating a prep pack."""
    meeting_id: str = Field(..., min_length=1, description="ID of the upcoming meeting")
    recurring_meeting_id: Optional[str] = Field(
        None, description="Optional override for recurring meeting ID"
    )
    previous_meeting_counts: Optional[int] = Field(
        None, ge=1, le=10, description="Number of previous meetings to analyze (1-10)"
    )
    tenant_id: str = Field(..., min_length=1, description="Tenant identifier")


class PrepPackResponse(BaseModel):
    """Response model for prep pack operations."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class PrepPackListResponse(BaseModel):
    """Response model for listing prep packs."""
    success: bool
    data: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    message: Optional[str] = None


@router.post("/generate", response_model=PrepPackResponse)
async def generate_prep_pack(request: GeneratePrepPackRequest) -> PrepPackResponse:
    """
    Generate a meeting prep pack and save it to MongoDB.
    
    This endpoint:
    1. Uses MeetingPrepAgent to analyze previous meetings
    2. Generates structured prep pack with insights and recommendations
    3. Saves the prep pack to MongoDB for future retrieval
    """
    try:
        service = await MeetingPrepService.from_default()
        
        result = await service.generate_and_save_prep_pack(
            meeting_id=request.meeting_id,
            recurring_meeting_id=request.recurring_meeting_id,
            previous_meeting_counts=request.previous_meeting_counts,
            context={"tenant_id": request.tenant_id},
        )
        
        logger.info(
            "[MeetingPrepAPI] Generated prep pack for meeting=%s tenant=%s",
            request.meeting_id,
            request.tenant_id,
        )
        
        return PrepPackResponse(
            success=True,
            data=result,
            message="Prep pack generated and saved successfully",
        )
        
    except MeetingPrepServiceError as exc:
        logger.error("[MeetingPrepAPI] Service error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("[MeetingPrepAPI] Unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{meeting_id}", response_model=PrepPackResponse)
async def get_prep_pack_by_meeting_id(
    meeting_id: str,
    tenant_id: str = Query(default="689ddc0411e4209395942bee", description="Tenant identifier"),
) -> PrepPackResponse:
    """
    Get a prep pack by meeting ID.
    
    Returns the prep pack associated with a specific meeting.
    """

    # try:
    #     service = await MeetingPrepService.from_default()
        
    #     prep_pack = await service.get_prep_pack_by_meeting_id(
    #         tenant_id=tenant_id,
    #         meeting_id=meeting_id,
    #     )
        
    #     if not prep_pack:
    #         raise HTTPException(
    #             status_code=404,
    #             detail=f"No prep pack found for meeting {meeting_id}",
    #         )
        
    #     logger.info(
    #         "[MeetingPrepAPI] Retrieved prep pack for meeting=%s tenant=%s",
    #         meeting_id,
    #         tenant_id,
    #     )
    prep_pack = {
        "id": meeting_id,
        "title": "Biot System: Multilanguage & Transcription Sync",
        "tenantId": tenant_id,
        "timezone": "Asia/Calcutta",
        "locale": "en-US",
        "purpose": "To align on a solution for accurately matching transcribed timestamps to speaker-mapped audio segments within the Biot system, enabling robust multilanguage support and advancing transcription capabilities.",
        "confidence": "medium",
        "expected_outcomes": [
            {
            "description": "Decision on the technical approach to accurately match transcribed timestamps to speaker-mapped audio segments in the Biot system.",
            "owner": "",
            "type": "decision"
            }
        ],
        "blocking_items": [
            {
            "title": "Accurate matching of transcribed timestamps to speaker-mapped audio segments",
            "owner": "",
            "eta": "",
            "impact": "Blocks full implementation and effectiveness of Biot system's multilanguage transcription capabilities and speaker diarization features.",
            "severity": "high",
            "status": "open"
            }
        ],
        "decision_queue": [
            {
            "id": "Biot-TS-Match-001",
            "title": "Solution for Transcribed Timestamp to Speaker-Mapped Audio Segment Matching",
            "needs": [
                "Proposed technical solution for timestamp synchronization",
                "Resource allocation for solution development and integration"
            ],
            "owner": ""
            }
        ],
        "key_points": [
            "The 'Biot' system is understood to support multilanguage capabilities.",
            "Multilanguage support prioritization is linked to voice activity detection, suggesting it may only be applied to active speakers.",
            "The 'Biot' system splits audio into various segments and maps them to individual speakers.",
            "A critical challenge is to accurately match transcribed timestamps to the corresponding speaker-mapped audio segments."
        ],
        "open_questions": [
            "How can transcribed timestamps be accurately matched to speaker-mapped audio segments?"
        ],
        "risks_issues": [
            "Delivery Risk: Inability to accurately match transcribed timestamps to speaker-mapped audio segments, hindering full Biot system functionality and multilanguage transcription accuracy."
        ],
        "leadership_asks": [
            "Approve the proposed technical solution for resolving the timestamp matching challenge.",
            "Allocate necessary engineering resources to implement the chosen solution for Biot system transcription."
        ],
        "previous_meetings_ref": [
            {
            "meeting_id": "68f9bde68bde1c2327cc15dd",
            "analysis_id": "68f9bde68bde1c2327cc15dd",
            "datetime": "2025-10-30T09:40:00.000Z"
            }
        ],
        "created_at": "2025-10-31T09:40:00.000Z",
        "updated_at": "2025-10-31T09:40:00.000Z"
    }

    return PrepPackResponse(
        success=True,
        data=prep_pack,
        message="Prep pack retrieved successfully",
    )
        
    # except HTTPException:
    #     raise
    # except Exception as exc:
    #     logger.exception("[MeetingPrepAPI] Unexpected error: %s", exc)
    #     raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/recurring/{recurring_meeting_id}", response_model=PrepPackResponse)
async def get_prep_pack_by_recurring_meeting_id(
    recurring_meeting_id: str,
    tenant_id: str = Query(default="689ddc0411e4209395942bee", description="Tenant identifier"),
) -> PrepPackResponse:
    """
    Get the latest prep pack by recurring meeting ID.
    
    Returns the most recent prep pack for a recurring meeting series.
    """
    try:
        service = await MeetingPrepService.from_default()
        
        prep_pack = await service.get_prep_pack_by_recurring_meeting_id(
            tenant_id=tenant_id,
            recurring_meeting_id=recurring_meeting_id,
        )
        
        if not prep_pack:
            raise HTTPException(
                status_code=404,
                detail=f"No prep pack found for recurring meeting {recurring_meeting_id}",
            )
        
        logger.info(
            "[MeetingPrepAPI] Retrieved prep pack for recurring_meeting=%s tenant=%s",
            recurring_meeting_id,
            tenant_id,
        )
        
        return PrepPackResponse(
            success=True,
            data=prep_pack,
            message="Prep pack retrieved successfully",
        )
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[MeetingPrepAPI] Unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/recurring/{recurring_meeting_id}/history", response_model=PrepPackListResponse)
async def get_prep_pack_history(
    recurring_meeting_id: str,
    tenant_id: str = Query(default="689ddc0411e4209395942bee", description="Tenant identifier"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of prep packs to return"),
) -> PrepPackListResponse:
    """
    Get historical prep packs for a recurring meeting series.
    
    Returns multiple prep packs for a recurring meeting, ordered by creation date (newest first).
    """
    try:
        service = await MeetingPrepService.from_default()
        
        prep_packs = await service.get_prep_packs_by_recurring_meeting_id(
            tenant_id=tenant_id,
            recurring_meeting_id=recurring_meeting_id,
            limit=limit,
        )
        
        logger.info(
            "[MeetingPrepAPI] Retrieved %d prep packs for recurring_meeting=%s tenant=%s",
            len(prep_packs),
            recurring_meeting_id,
            tenant_id,
        )
        
        return PrepPackListResponse(
            success=True,
            data=prep_packs,
            count=len(prep_packs),
            message=f"Retrieved {len(prep_packs)} prep packs",
        )
        
    except Exception as exc:
        logger.exception("[MeetingPrepAPI] Unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/recurring/{recurring_meeting_id}", response_model=PrepPackResponse)
async def delete_prep_pack(
    recurring_meeting_id: str,
    tenant_id: str = Query(default="689ddc0411e4209395942bee", description="Tenant identifier"),
) -> PrepPackResponse:
    """
    Delete a prep pack by recurring meeting ID.
    
    Removes the prep pack associated with a recurring meeting series.
    """
    try:
        service = await MeetingPrepService.from_default()
        
        result = await service.delete_prep_pack(
            tenant_id=tenant_id,
            recurring_meeting_id=recurring_meeting_id,
        )
        
        if result["deleted_count"] == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No prep pack found for recurring meeting {recurring_meeting_id}",
            )
        
        logger.info(
            "[MeetingPrepAPI] Deleted prep pack for recurring_meeting=%s tenant=%s",
            recurring_meeting_id,
            tenant_id,
        )
        
        return PrepPackResponse(
            success=True,
            data=result,
            message="Prep pack deleted successfully",
        )
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[MeetingPrepAPI] Unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")