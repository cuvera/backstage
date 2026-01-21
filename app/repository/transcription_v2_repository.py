import logging
from datetime import datetime
from typing import Any, Dict, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING
from bson.objectid import ObjectId

from app.schemas.transcription_v2 import (
    TranscriptionV2Document,
    ClusterSegment,
    ClusterTranscriptionSegment,
    ClusterMetadata
)
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class TranscriptionV2RepositoryError(Exception):
    """Raised when the transcription v2 repository cannot complete its task."""


class TranscriptionV2Repository(BaseRepository):
    """Repository for transcription v2 data persistence."""

    COLLECTION_NAME = "transcriptions_v2"

    def __init__(
        self,
        *,
        db: Optional[AsyncIOMotorDatabase] = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        super().__init__(db=db, collection_name=collection_name)

    @classmethod
    async def from_default(cls, collection_name: str = COLLECTION_NAME) -> "TranscriptionV2Repository":
        from app.db.mongodb import get_database

        db = await get_database()
        repository = cls(db=db, collection_name=collection_name)
        await repository.ensure_indexes()
        return repository

    async def ensure_indexes(self) -> None:
        """Create MongoDB indexes for optimal query performance."""
        collection = await self._ensure_collection()

        # Index for querying by tenant (compound with created_at for time-based queries)
        await collection.create_index(
            [("tenant_id", ASCENDING), ("created_at", ASCENDING)],
            name="tenant_created_at"
        )

        logger.info("Created indexes for transcriptions_v2 collection")

    async def save_transcription(
        self,
        meeting_id: str,
        tenant_id: str,
        transcription_v2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Save transcription v2 data to MongoDB.

        Args:
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier
            transcription_v2: V2 transcription output with segments and metadata

        Returns:
            Dictionary with save result
        """
        try:
            collection = await self._ensure_collection()

            # Parse segments
            segments = []
            for segment_data in transcription_v2.get("segments", []):
                # Parse transcriptions within each cluster
                transcriptions = []
                for trans_data in segment_data.get("transcriptions", []):
                    transcriptions.append(ClusterTranscriptionSegment(**trans_data))

                # Create cluster segment
                segment = ClusterSegment(
                    segment_cluster_id=segment_data.get("segment_cluster_id"),
                    topic=segment_data.get("topic", []),
                    type=segment_data.get("type"),
                    start_time=segment_data.get("start_time"),
                    end_time=segment_data.get("end_time"),
                    duration=segment_data.get("duration"),
                    speakers=segment_data.get("speakers", []),
                    segment_count=segment_data.get("segment_count", 0),
                    transcriptions=transcriptions
                )
                segments.append(segment)

            # Parse metadata
            metadata_data = transcription_v2.get("metadata", {})
            metadata = ClusterMetadata(
                total_clusters=metadata_data.get("total_clusters", 0),
                total_segments=metadata_data.get("total_segments", 0),
                clusters_by_type=metadata_data.get("clusters_by_type", {})
            )

            # Create transcription v2 document
            transcription_doc = TranscriptionV2Document(
                id=ObjectId(meeting_id),
                tenant_id=tenant_id,
                meeting_id=meeting_id,
                segments=segments,
                metadata=metadata,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Upsert document (replace if exists by _id + tenant_id)
            filter_query = {"_id": ObjectId(meeting_id), "tenant_id": tenant_id}
            doc_data = transcription_doc.model_dump(by_alias=True, exclude_none=True)
            doc_data["_id"] = ObjectId(meeting_id)  # Ensure _id remains as ObjectId, not string
            doc_data["updated_at"] = datetime.utcnow()

            result = await collection.replace_one(
                filter_query,
                doc_data,
                upsert=True
            )

            logger.info(
                "Saved transcription v2 for meeting=%s, tenant=%s, clusters=%d, total_segments=%d",
                meeting_id, tenant_id, len(segments), metadata.total_segments
            )

            return {
                "meeting_id": meeting_id,
                "tenant_id": tenant_id,
                "total_clusters": len(segments),
                "total_segments": metadata.total_segments,
                "upserted_id": result.upserted_id,
                "modified_count": result.modified_count,
                "matched_count": result.matched_count,
                "collection": self._collection_name
            }

        except Exception as exc:
            logger.exception("Failed to save transcription v2 for meeting=%s: %s", meeting_id, exc)
            raise TranscriptionV2RepositoryError(f"Failed to save transcription v2: {exc}") from exc

    async def get_by_meeting_id(self, meeting_id: str, tenant_id: str) -> Optional[TranscriptionV2Document]:
        """
        Get transcription v2 by meeting ID and tenant ID.

        Args:
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier

        Returns:
            TranscriptionV2Document or None if not found
        """
        try:
            collection = await self._ensure_collection()

            filter_query = {"_id": ObjectId(meeting_id), "tenant_id": tenant_id}
            doc = await collection.find_one(filter_query)

            if doc:
                # Parse MongoDB document back to Pydantic model
                transcription_doc = TranscriptionV2Document(**doc)
                logger.info("Retrieved transcription v2 for meeting=%s, tenant=%s", meeting_id, tenant_id)
                return transcription_doc

            logger.info("Transcription v2 not found for meeting=%s, tenant=%s", meeting_id, tenant_id)
            return None

        except Exception as exc:
            logger.exception("Failed to get transcription v2 for meeting=%s: %s", meeting_id, exc)
            raise TranscriptionV2RepositoryError(f"Failed to get transcription v2: {exc}") from exc

    async def delete_transcription(self, meeting_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        Delete transcription v2 by meeting ID and tenant ID.

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
                "Deleted transcription v2 for meeting=%s, tenant=%s, deleted_count=%d",
                meeting_id, tenant_id, result.deleted_count
            )

            return {
                "meeting_id": meeting_id,
                "tenant_id": tenant_id,
                "deleted_count": result.deleted_count,
                "collection": self._collection_name
            }

        except Exception as exc:
            logger.exception("Failed to delete transcription v2 for meeting=%s: %s", meeting_id, exc)
            raise TranscriptionV2RepositoryError(f"Failed to delete transcription v2: {exc}") from exc
