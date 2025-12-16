"""
Schema for transcription chunk persistence.
Enables incremental saving and retry of individual chunks.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkInfo(BaseModel):
    """Information about the audio chunk."""
    chunk_id: int
    start_time: str
    end_time: str
    file_path: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None


class TranscriptionChunkResult(BaseModel):
    """Result of transcribing a single chunk."""
    transcriptions: List[Dict[str, Any]] = Field(default_factory=list)
    chunk_info: Optional[Dict[str, Any]] = None


class TranscriptionChunkDocument(BaseModel):
    """MongoDB document for a single transcription chunk."""
    meeting_id: str = Field(..., description="Meeting identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    chunk_id: int = Field(..., description="Chunk sequence number (1-based)")
    status: str = Field(..., description="Chunk status: processing, success, or failed")
    chunk_info: ChunkInfo = Field(..., description="Chunk metadata")
    result: Optional[TranscriptionChunkResult] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
