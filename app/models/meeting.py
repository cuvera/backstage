from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    audio_merging = "audio_merging"
    transcribing = "transcribing"
    analyzing = "analyzing"
    completed = "completed"
    failed = "failed"
    retry = "retry"


class MeetingParticipant(BaseModel):
    name: str
    email: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None


class MeetingRecord(BaseModel):
    meeting_id: str
    tenant_id: str
    title: Optional[str] = None
    audio_file_url: str
    bucket: str = "recordings"
    processing_status: ProcessingStatus = ProcessingStatus.pending
    raw_payload: Dict[str, Any] = Field(default_factory=dict)  # Store original payload
    participants: List[MeetingParticipant] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class AudioProcessingResult(BaseModel):
    local_merged_file_path: str
    merged_file_s3_key: str
    total_files_merged: int
    total_duration_seconds: float
    file_size_bytes: int
    bucket_name: str


class TranscriptionResult(BaseModel):
    conversation: List[Dict[str, Any]]
    participants: List[Dict[str, Any]]
    duration_seconds: float
    language: Optional[str] = None
    confidence_score: Optional[float] = None


class ProcessingProgress(BaseModel):
    meeting_id: str
    status: ProcessingStatus
    current_step: str
    progress_percentage: float = 0.0
    error_message: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)