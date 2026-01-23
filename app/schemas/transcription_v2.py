from datetime import datetime
from typing import Dict, List, Optional
from bson.objectid import ObjectId
from pydantic import BaseModel, Field, ConfigDict


class ClusterTranscriptionSegment(BaseModel):
    """Individual transcription segment within a cluster."""
    segment_id: int = Field(..., description="Segment identifier from V1")
    start: str = Field(..., description="Start time in MM:SS format")
    end: str = Field(..., description="End time in MM:SS format")
    text: str = Field(default="", description="Transcribed text content")
    speaker: Optional[str] = Field(default="Unknown", description="Speaker name if available")
    sentiment: Optional[str] = Field(default=None, description="Sentiment label (positive/negative/neutral/mixed)")


class ClusterSegment(BaseModel):
    """A classified cluster of related transcription segments."""
    segment_cluster_id: int = Field(..., description="Cluster identifier")
    topic: List[str] = Field(default_factory=list, description="Topics discussed in this cluster")
    type: str = Field(..., description="Cluster type: actionable_item, decision, key_insight, question, general_discussion")
    start_time: str = Field(..., description="Cluster start time in MM:SS format")
    end_time: str = Field(..., description="Cluster end time in MM:SS format")
    duration: str = Field(..., description="Total cluster duration in MM:SS format")
    speakers: List[str] = Field(default_factory=list, description="List of speakers in this cluster")
    segment_count: int = Field(default=0, description="Number of segments in this cluster")
    transcriptions: List[ClusterTranscriptionSegment] = Field(default_factory=list, description="Segments belonging to this cluster")


class ClusterMetadata(BaseModel):
    """Aggregated metadata for transcription V2."""
    total_clusters: int = Field(default=0, description="Total number of clusters")
    total_segments: int = Field(default=0, description="Total number of segments across all clusters")
    clusters_by_type: Dict[str, int] = Field(default_factory=dict, description="Count of clusters by type")


class TranscriptionV2Document(BaseModel):
    """Complete transcription v2 document for MongoDB storage."""
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

    id: Optional[ObjectId] = Field(default=None, alias="_id", description="MongoDB document ID (meeting_id)")
    tenant_id: str = Field(..., description="Tenant identifier for security")
    meeting_id: str = Field(..., description="Meeting identifier")
    segments: List[ClusterSegment] = Field(default_factory=list, description="List of classified cluster segments")
    metadata: ClusterMetadata = Field(default_factory=ClusterMetadata, description="Aggregated metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Document creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Document update timestamp")


class TranscriptionV2Response(BaseModel):
    """Response model for transcription v2 API endpoints."""
    meeting_id: str
    tenant_id: str
    segments: List[ClusterSegment]
    metadata: ClusterMetadata
    created_at: datetime
    updated_at: datetime
