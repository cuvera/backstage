from datetime import datetime
from typing import Any, Dict, List, Optional
from bson.objectid import ObjectId
from pydantic import BaseModel, Field
from enum import Enum

class SentimentLabel(str, Enum):
    """Sentiment analysis labels."""
    positive = "positive"
    negative = "negative" 
    neutral = "neutral"
    mixed = "mixed"

class ParticipantSentiment(BaseModel):
    """Sentiment analysis for individual participant."""
    name: str = Field(..., description="Participant name")
    user_id: Optional[str] = Field(None, description="User ID if available") 
    sentiment: SentimentLabel = Field(..., description="Sentiment label for this participant")


class SentimentOverview(BaseModel):
    """Complete sentiment analysis overview."""
    overall: SentimentLabel = Field(..., description="Overall meeting sentiment")
    participant: List[ParticipantSentiment] = Field(default_factory=list, description="Per-participant sentiments")


class TranscriptionEntry(BaseModel):
    """Individual transcription entry with speaker diarization."""
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds") 
    speaker: str = Field(..., description="Speaker identifier/name")
    text: str = Field(..., description="Transcribed text")
    identification_score: float = Field(..., description="Confidence score for speaker identification")
    user_id: Optional[str] = Field(None, description="Optional user identifier")


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
    sentiments: SentimentOverview = Field(..., description="Sentiment analysis overview")
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