from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TranscriptionEntry(BaseModel):
    """Individual transcription entry with speaker diarization."""
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds") 
    speaker: str = Field(..., description="Speaker identifier/name")
    text: str = Field(..., description="Transcribed text")
    identification_score: float = Field(..., description="Confidence score for speaker identification")


class ProcessingMetadata(BaseModel):
    """Metadata about the transcription processing."""
    vox_scribe_version: str = Field(default="1.0", description="Version of vox_scribe pipeline")
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="When transcription was processed")
    audio_duration_seconds: Optional[float] = Field(None, description="Duration of audio file")
    known_speakers: Optional[int] = Field(None, description="Number of known speakers provided")


class TranscriptionDocument(BaseModel):
    """Complete transcription document for MongoDB storage."""
    tenant_id: str = Field(..., description="Tenant identifier for security")
    conversation: List[TranscriptionEntry] = Field(..., description="List of transcription entries")
    total_speakers: int = Field(..., description="Total number of unique speakers detected")
    processing_metadata: ProcessingMetadata = Field(..., description="Processing metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Document creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Document update timestamp")


class TranscriptionResponse(BaseModel):
    """Response model for transcription API endpoints."""
    meeting_id: str
    conversation: List[TranscriptionEntry]
    total_speakers: int
    duration_seconds: Optional[float] = None
    created_at: datetime