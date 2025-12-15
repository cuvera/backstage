import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING
from bson.objectid import ObjectId

from app.schemas.transcription_v1 import (
    TranscriptionV1Document,
    TranscriptionSegment,
    SpeakerSummary,
    TranscriptionMetadata,
    ProcessingMetadata,
    SentimentLabel
)
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class TranscriptionV1RepositoryError(Exception):
    """Raised when the transcription v1 repository cannot complete its task."""


class TranscriptionV1Repository(BaseRepository):
    """Repository for transcription v1 data persistence."""

    COLLECTION_NAME = "transcriptions_v1"

    def __init__(
        self,
        *,
        db: Optional[AsyncIOMotorDatabase] = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        super().__init__(db=db, collection_name=collection_name)

    @classmethod
    async def from_default(cls, collection_name: str = COLLECTION_NAME) -> "TranscriptionV1Repository":
        from app.db.mongodb import get_database

        db = await get_database()
        repository = cls(db=db, collection_name=collection_name)
        await repository.ensure_indexes()
        return repository

    async def ensure_indexes(self) -> None:
        """Create MongoDB indexes for optimal query performance."""
        collection = await self._ensure_collection()
        
        # Index for querying by tenant (compound with _id is automatic)
        await collection.create_index(
            [("tenant_id", ASCENDING), ("created_at", ASCENDING)],
            name="tenant_created_at"
        )
        
        logger.info("Created indexes for transcriptions_v1 collection")

    async def save_transcription(
        self, 
        meeting_id: str,
        tenant_id: str,
        transcription_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Save transcription v1 data to MongoDB.
        
        Args:
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier
            transcription_result: TranscriptionService output result
            
        Returns:
            Dictionary with save result
        """
        try:
            collection = await self._ensure_collection()
            
            # Convert transcription segments
            transcription_segments = []
            for segment_data in transcription_result.get("transcriptions", []):
                transcription_segments.append(TranscriptionSegment(
                    segment_id=segment_data.get("segment_id", 0),
                    start=segment_data.get("start", "00:00"),
                    end=segment_data.get("end", "00:00"),
                    transcription=segment_data.get("transcription", ""),
                    sentiment=SentimentLabel(segment_data.get("sentiment", "neutral").lower()),
                    source_chunk=segment_data.get("source_chunk", 0),
                    chunk_start_time=segment_data.get("chunk_start_time", "00:00"),
                    chunk_end_time=segment_data.get("chunk_end_time", "00:00"),
                    speaker=segment_data.get("speaker", "Unknown")
                ))
            
            # Convert speaker summaries
            speaker_summaries = []
            for speaker_data in transcription_result.get("speakers", []):
                speaker_summaries.append(SpeakerSummary(
                    speaker=speaker_data.get("speaker", "Unknown"),
                    segments=speaker_data.get("segments", 0),
                    duration=speaker_data.get("duration", "00:00"),
                    sentiment=SentimentLabel(speaker_data.get("sentiment", "neutral").lower())
                ))
            
            # Convert metadata
            metadata_data = transcription_result.get("metadata", {})
            metadata = TranscriptionMetadata(
                total_segments=metadata_data.get("total_segments", 0),
                successful_chunks=metadata_data.get("successful_chunks", 0),
                failed_chunks=metadata_data.get("failed_chunks", 0),
                total_chunks=metadata_data.get("total_chunks", 0),
                has_speaker_mapping=metadata_data.get("has_speaker_mapping", False)
            )
            
            # Convert processing metadata
            proc_metadata_data = transcription_result.get("processing_metadata", {})
            processing_metadata = ProcessingMetadata(
                platform=proc_metadata_data.get("platform", "unknown"),
                audio_file_path=proc_metadata_data.get("audio_file_path", ""),
                processing_time_ms=proc_metadata_data.get("processing_time_ms", 0.0),
                chunk_duration_minutes=proc_metadata_data.get("chunk_duration_minutes", 10.0),
                overlap_seconds=proc_metadata_data.get("overlap_seconds", 5.0),
                max_concurrent=proc_metadata_data.get("max_concurrent", 5)
            )
            
            # Create transcription v1 document
            transcription_doc = TranscriptionV1Document(
                tenant_id=tenant_id,
                transcriptions=transcription_segments,
                speakers=speaker_summaries,
                metadata=metadata,
                processing_metadata=processing_metadata,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Upsert document (replace if exists by _id + tenant_id)
            filter_query = {"_id": ObjectId(meeting_id), "tenant_id": tenant_id}
            doc_data = transcription_doc.model_dump()
            doc_data["_id"] = ObjectId(meeting_id)
            doc_data["updated_at"] = datetime.utcnow()
            
            result = await collection.replace_one(
                filter_query,
                doc_data,
                upsert=True
            )
            
            logger.info(
                "Saved transcription v1 for meeting=%s, tenant=%s, speakers=%d, segments=%d",
                meeting_id, tenant_id, len(speaker_summaries), len(transcription_segments)
            )
            
            return {
                "meeting_id": meeting_id,
                "tenant_id": tenant_id,
                "total_speakers": len(speaker_summaries),
                "total_segments": len(transcription_segments),
                "upserted_id": result.upserted_id,
                "modified_count": result.modified_count,
                "matched_count": result.matched_count,
                "collection": self._collection_name
            }
            
        except Exception as exc:
            logger.exception("Failed to save transcription v1 for meeting=%s: %s", meeting_id, exc)
            raise TranscriptionV1RepositoryError(f"Failed to save transcription v1: {exc}") from exc

    async def get_transcription(self, meeting_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """
        Get transcription v1 by meeting ID and tenant ID.
        
        Args:
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier
            
        Returns:
            Transcription v1 document or None if not found
        """
        try:
            collection = await self._ensure_collection()
            
            filter_query = {"_id": ObjectId(meeting_id), "tenant_id": tenant_id}
            doc = await collection.find_one(filter_query)
            
            if doc:
                # Convert _id back to meeting_id for response
                doc["meeting_id"] = str(doc.pop("_id"))
                logger.info("Retrieved transcription v1 for meeting=%s, tenant=%s", meeting_id, tenant_id)
                return doc
            
            logger.warning("Transcription v1 not found for meeting=%s, tenant=%s", meeting_id, tenant_id)
            return None
            
        except Exception as exc:
            logger.exception("Failed to get transcription v1 for meeting=%s: %s", meeting_id, exc)
            raise TranscriptionV1RepositoryError(f"Failed to get transcription v1: {exc}") from exc

    async def delete_transcription(self, meeting_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        Delete transcription v1 by meeting ID and tenant ID.
        
        Args:
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary with deletion result
        """
        try:
            collection = await self._ensure_collection()
            
            filter_query = {"_id": ObjectId(meeting_id), "tenant_id": tenant_id}
            result = await collection.delete_one(filter_query)
            
            logger.info(
                "Deleted transcription v1 for meeting=%s, tenant=%s, deleted_count=%d",
                meeting_id, tenant_id, result.deleted_count
            )
            
            return {
                "meeting_id": meeting_id,
                "tenant_id": tenant_id,
                "deleted_count": result.deleted_count,
                "collection": self._collection_name
            }
            
        except Exception as exc:
            logger.exception("Failed to delete transcription v1 for meeting=%s: %s", meeting_id, exc)
            raise TranscriptionV1RepositoryError(f"Failed to delete transcription v1: {exc}") from exc