from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.meeting import MeetingPlatform, ProcessingStatus


class MeetingParticipantResponse(BaseModel):
    name: str
    email: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None


class MeetingResponse(BaseModel):
    meeting_id: str
    tenant_id: str
    platform: MeetingPlatform
    title: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    audio_file_url: str
    bucket: str
    participants: List[MeetingParticipantResponse] = Field(default_factory=list)
    processing_status: ProcessingStatus
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
    retry_count: int = 0


class MeetingProcessingRequest(BaseModel):
    meeting_id: str
    tenant_id: str
    audio_file_url: str
    bucket: str = "recordings"
    platform: MeetingPlatform = MeetingPlatform.google
    title: Optional[str] = None
    participants: List[MeetingParticipantResponse] = Field(default_factory=list)


class ProcessingStatusResponse(BaseModel):
    meeting_id: str
    status: ProcessingStatus
    current_step: str
    progress_percentage: float = 0.0
    error_message: Optional[str] = None
    started_at: datetime
    updated_at: datetime


class TranscriptionResponse(BaseModel):
    meeting_id: str
    conversation: List[Dict[str, Any]]
    participants: List[Dict[str, Any]]
    duration_seconds: float
    language: Optional[str] = None
    confidence_score: Optional[float] = None
    created_at: datetime


class AnalysisResponse(BaseModel):
    meeting_id: str
    tenant_id: str
    session_id: str
    summary: str
    key_points: List[str] = Field(default_factory=list)
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    action_items: List[Dict[str, Any]] = Field(default_factory=list)
    risks_issues: List[Dict[str, Any]] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    confidence: str = "medium"
    created_at: str
    transcript_language: Optional[str] = None
    duration_sec: Optional[float] = None


class MeetingEventRequest(BaseModel):
    metadata: Dict[str, Any]
    payload: Dict[str, Any]
    headers: Dict[str, str] = Field(default_factory=dict)