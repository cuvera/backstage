import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import ASCENDING
from bson.objectid import ObjectId

from app.db.mongodb import get_database
from app.schemas.transcription import TranscriptionDocument, TranscriptionEntry, ProcessingMetadata

logger = logging.getLogger(__name__)

COLLECTION_NAME = "transcriptions"


class TranscriptionServiceError(Exception):
    """Raised when the transcription service cannot complete its task."""


class TranscriptionService:
    """Service for managing meeting transcriptions in MongoDB."""

    def __init__(self):
        self._db: Optional[AsyncIOMotorDatabase] = None
        self._collection_name = COLLECTION_NAME

    async def _get_db(self) -> AsyncIOMotorDatabase:
        """Get database connection."""
        if self._db is None:
            self._db = await get_database()
        return self._db

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get transcriptions collection."""
        db = await self._get_db()
        return db[self._collection_name]

    async def ensure_indexes(self) -> None:
        """Create MongoDB indexes for optimal query performance."""
        collection = await self._get_collection()
        
        # Index for querying by tenant (compound with _id is automatic)
        await collection.create_index(
            [("tenant_id", ASCENDING), ("created_at", ASCENDING)],
            name="tenant_created_at"
        )
        
        # Note: _id is automatically indexed by MongoDB, so no need to create separate index

        logger.info("Created indexes for transcriptions collection")

    async def save_transcription(
        self, 
        meeting_id: str,
        tenant_id: str,
        conversation: List[Dict[str, Any]],
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save transcription data to MongoDB.
        
        Args:
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier
            conversation: List of transcription entries from vox_scribe
            processing_metadata: Optional processing metadata
            
        Returns:
            Dictionary with save result
        """
        try:
            collection = await self._get_collection()
            
            # Convert conversation entries to proper format
            transcription_entries = []
            for entry in conversation:
                transcription_entries.append(TranscriptionEntry(
                    start_time=entry.get("start_time", 0.0),
                    end_time=entry.get("end_time", 0.0),
                    speaker=entry.get("speaker", "Unknown"),
                    text=entry.get("text", ""),
                    identification_score=entry.get("identification_score", 0.0)
                ))
            
            # Calculate total speakers
            unique_speakers = set(entry.speaker for entry in transcription_entries)
            total_speakers = len(unique_speakers)
            
            # Build processing metadata
            proc_metadata = ProcessingMetadata(
                vox_scribe_version=processing_metadata.get("vox_scribe_version", "1.0") if processing_metadata else "1.0",
                processed_at=datetime.utcnow(),
                audio_duration_seconds=processing_metadata.get("audio_duration_seconds") if processing_metadata else None,
                known_speakers=processing_metadata.get("known_speakers") if processing_metadata else None
            )
            
            # Create transcription document
            transcription_doc = TranscriptionDocument(
                tenant_id=tenant_id,
                conversation=transcription_entries,
                total_speakers=total_speakers,
                processing_metadata=proc_metadata,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Upsert document (replace if exists by _id + tenant_id)
            filter_query = {"_id": ObjectId(meeting_id), "tenant_id": ObjectId(tenant_id)}
            doc_data = transcription_doc.model_dump()
            doc_data["_id"] = ObjectId(meeting_id)
            doc_data["tenant_id"] = ObjectId(tenant_id)
            doc_data["updated_at"] = datetime.utcnow()
            
            result = await collection.replace_one(
                filter_query,
                doc_data,
                upsert=True
            )
            
            logger.info(
                "Saved transcription for meeting=%s, tenant=%s, speakers=%d, entries=%d",
                meeting_id, tenant_id, total_speakers, len(transcription_entries)
            )
            
            return {
                "meeting_id": meeting_id,
                "tenant_id": tenant_id,
                "total_speakers": total_speakers,
                "total_entries": len(transcription_entries),
                "upserted_id": result.upserted_id,
                "modified_count": result.modified_count,
                "matched_count": result.matched_count,
                "collection": self._collection_name
            }
            
        except Exception as exc:
            logger.exception("Failed to save transcription for meeting=%s: %s", meeting_id, exc)
            raise TranscriptionServiceError(f"Failed to save transcription: {exc}") from exc

    async def get_transcription(self, meeting_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """
        Get transcription by meeting ID and tenant ID.
        
        Args:
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier
            
        Returns:
            Transcription document or None if not found
        """
        try:
            collection = await self._get_collection()
            
            filter_query = {"_id": ObjectId(meeting_id), "tenant_id": ObjectId(tenant_id)}
            doc = await collection.find_one(filter_query)
            
            if doc:
                # Convert _id back to meeting_id for response
                doc["meeting_id"] = str(doc.pop("_id"))
                logger.info("Retrieved transcription for meeting=%s, tenant=%s", meeting_id, tenant_id)
                return doc
            
            logger.warning("Transcription not found for meeting=%s, tenant=%s", meeting_id, tenant_id)
            return None
            
        except Exception as exc:
            logger.exception("Failed to get transcription for meeting=%s: %s", meeting_id, exc)
            raise TranscriptionServiceError(f"Failed to get transcription: {exc}") from exc

    async def delete_transcription(self, meeting_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        Delete transcription by meeting ID and tenant ID.
        
        Args:
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary with deletion result
        """
        try:
            collection = await self._get_collection()
            
            filter_query = {"_id": ObjectId(meeting_id), "tenant_id": ObjectId(tenant_id)}
            result = await collection.delete_one(filter_query)
            
            logger.info(
                "Deleted transcription for meeting=%s, tenant=%s, deleted_count=%d",
                meeting_id, tenant_id, result.deleted_count
            )
            
            return {
                "meeting_id": meeting_id,
                "tenant_id": tenant_id,
                "deleted_count": result.deleted_count,
                "collection": self._collection_name
            }
            
        except Exception as exc:
            logger.exception("Failed to delete transcription for meeting=%s: %s", meeting_id, exc)
            raise TranscriptionServiceError(f"Failed to delete transcription: {exc}") from exc