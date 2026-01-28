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


class TranscriptionSegment(BaseModel):
    """Individual transcription segment from Gemini output."""
    segment_id: Optional[int] = Field(..., description="Segment identifier")
    start: Optional[str] = Field(..., description="Start time in MM:SS format")
    end: Optional[str] = Field(..., description="End time in MM:SS format")
    transcription: Optional[str] = Field(..., description="Transcribed text")
    sentiment: Optional[SentimentLabel] = Field(..., description="Sentiment label for this segment")
    source_chunk: Optional[int] = Field(..., description="Source chunk identifier")
    chunk_start_time: Optional[str] = Field(..., description="Chunk start time in MM:SS format")
    chunk_end_time: Optional[str] = Field(..., description="Chunk end time in MM:SS format")
    speaker: Optional[str] = Field(..., description="Speaker name")


class SpeakerSummary(BaseModel):
    """Speaker summary with statistics."""
    speaker: str = Field(..., description="Speaker name")
    segments: int = Field(..., description="Number of segments")
    duration: str = Field(..., description="Total duration in MM:SS format")
    sentiment: SentimentLabel = Field(..., description="Average sentiment for speaker")


class TranscriptionMetadata(BaseModel):
    """Transcription processing metadata."""
    total_segments: int = Field(..., description="Total number of segments")
    successful_chunks: int = Field(..., description="Number of successfully processed chunks")
    failed_chunks: int = Field(..., description="Number of failed chunks")
    total_chunks: int = Field(..., description="Total number of chunks")
    has_speaker_mapping: bool = Field(..., description="Whether speaker mapping was applied")


class ProcessingMetadata(BaseModel):
    """Processing metadata from TranscriptionService."""
    platform: str = Field(..., description="Platform used (online/offline)")
    audio_file_path: str = Field(..., description="Path to audio file")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    chunk_duration_minutes: float = Field(..., description="Chunk duration in minutes")
    overlap_seconds: float = Field(..., description="Overlap between chunks in seconds")
    max_concurrent: int = Field(..., description="Maximum concurrent transcriptions")


class TranscriptionV1Document(BaseModel):
    """Complete transcription v1 document for MongoDB storage."""
    tenant_id: str = Field(..., description="Tenant identifier for security")
    transcriptions: List[TranscriptionSegment] = Field(..., description="List of transcription segments")
    speakers: List[SpeakerSummary] = Field(..., description="Speaker summary statistics")
    metadata: TranscriptionMetadata = Field(..., description="Transcription metadata")
    processing_metadata: ProcessingMetadata = Field(..., description="Processing metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Document creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Document update timestamp")


class TranscriptionV1Response(BaseModel):
    """Response model for transcription v1 API endpoints."""
    meeting_id: str
    tenant_id: str
    transcriptions: List[TranscriptionSegment]
    speakers: List[SpeakerSummary]
    metadata: TranscriptionMetadata
    processing_metadata: ProcessingMetadata
    created_at: datetime
    updated_at: datetime