"""
Repository for persisting individual transcription chunks.
Enables incremental saving and efficient retry logic.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING

from app.schemas.transcription_chunk import (
    TranscriptionChunkDocument,
    ChunkInfo,
    TranscriptionChunkResult
)
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class TranscriptionChunkRepositoryError(Exception):
    """Raised when the transcription chunk repository cannot complete its task."""


class TranscriptionChunkRepository(BaseRepository):
    """Repository for transcription chunk data persistence."""

    COLLECTION_NAME = "transcription_chunks"

    def __init__(
        self,
        *,
        db: Optional[AsyncIOMotorDatabase] = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        super().__init__(db=db, collection_name=collection_name)

    @classmethod
    async def from_default(cls, collection_name: str = COLLECTION_NAME) -> "TranscriptionChunkRepository":
        from app.db.mongodb import get_database

        db = await get_database()
        repository = cls(db=db, collection_name=collection_name)
        await repository.ensure_indexes()
        return repository

    async def ensure_indexes(self) -> None:
        """Create MongoDB indexes for optimal query performance."""
        collection = await self._ensure_collection()

        # Compound index for querying chunks by meeting
        await collection.create_index(
            [("meeting_id", ASCENDING), ("tenant_id", ASCENDING), ("chunk_id", ASCENDING)],
            name="meeting_tenant_chunk",
            unique=True
        )

        # Index for querying by status
        await collection.create_index(
            [("meeting_id", ASCENDING), ("status", ASCENDING)],
            name="meeting_status"
        )

        logger.info("Created indexes for transcription_chunks collection")

    async def save_chunk(
        self,
        meeting_id: str,
        tenant_id: str,
        chunk_id: int,
        status: str,
        chunk_info: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save or update a transcription chunk.

        Args:
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier
            chunk_id: Chunk sequence number (1-based)
            status: Chunk status ("processing", "success", "failed")
            chunk_info: Chunk metadata (start_time, end_time, etc.)
            result: Transcription result if successful
            error: Error message if failed

        Returns:
            Dictionary with save result
        """
        try:
            collection = await self._ensure_collection()

            # Prepare chunk info
            if chunk_info:
                chunk_info_obj = ChunkInfo(
                    chunk_id=chunk_id,
                    start_time=chunk_info.get("start_time", "00:00"),
                    end_time=chunk_info.get("end_time", "00:00"),
                    file_path=chunk_info.get("file_path"),
                    segments=chunk_info.get("segments")
                )
            else:
                chunk_info_obj = ChunkInfo(
                    chunk_id=chunk_id,
                    start_time="00:00",
                    end_time="00:00"
                )

            # Prepare result
            result_obj = None
            if result:
                result_obj = TranscriptionChunkResult(
                    transcriptions=result.get("transcriptions", []),
                    chunk_info=result.get("chunk_info")
                )

            # Create document
            chunk_doc = TranscriptionChunkDocument(
                meeting_id=meeting_id,
                tenant_id=tenant_id,
                chunk_id=chunk_id,
                status=status,
                chunk_info=chunk_info_obj,
                result=result_obj,
                error=error,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Upsert document
            filter_query = {
                "meeting_id": meeting_id,
                "tenant_id": tenant_id,
                "chunk_id": chunk_id
            }

            doc_data = chunk_doc.model_dump()
            doc_data["updated_at"] = datetime.utcnow()

            db_result = await collection.replace_one(
                filter_query,
                doc_data,
                upsert=True
            )

            logger.info(
                f"Saved chunk {chunk_id} for meeting={meeting_id}, status={status}"
            )

            return {
                "meeting_id": meeting_id,
                "tenant_id": tenant_id,
                "chunk_id": chunk_id,
                "status": status,
                "upserted_id": db_result.upserted_id,
                "modified_count": db_result.modified_count
            }

        except Exception as exc:
            logger.exception(f"Failed to save chunk {chunk_id} for meeting={meeting_id}: {exc}")
            raise TranscriptionChunkRepositoryError(f"Failed to save chunk: {exc}") from exc

    async def get_chunks(
        self,
        meeting_id: str,
        tenant_id: str,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a meeting, optionally filtered by status.

        Args:
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier
            status: Optional status filter ("success", "failed", "processing")

        Returns:
            List of chunk documents
        """
        try:
            collection = await self._ensure_collection()

            filter_query = {
                "meeting_id": meeting_id,
                "tenant_id": tenant_id
            }

            if status:
                filter_query["status"] = status

            cursor = collection.find(filter_query).sort("chunk_id", ASCENDING)
            chunks = await cursor.to_list(length=None)

            logger.info(
                f"Retrieved {len(chunks)} chunks for meeting={meeting_id}"
                + (f", status={status}" if status else "")
            )

            return chunks

        except Exception as exc:
            logger.exception(f"Failed to get chunks for meeting={meeting_id}: {exc}")
            raise TranscriptionChunkRepositoryError(f"Failed to get chunks: {exc}") from exc

    async def get_completed_chunk_ids(
        self,
        meeting_id: str,
        tenant_id: str
    ) -> List[int]:
        """
        Get list of successfully completed chunk IDs for a meeting.

        Args:
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier

        Returns:
            List of chunk IDs with status="success"
        """
        try:
            chunks = await self.get_chunks(meeting_id, tenant_id, status="success")
            chunk_ids = [chunk["chunk_id"] for chunk in chunks]

            logger.info(f"Found {len(chunk_ids)} completed chunks for meeting={meeting_id}")
            return chunk_ids

        except Exception as exc:
            logger.exception(f"Failed to get completed chunk IDs for meeting={meeting_id}: {exc}")
            raise TranscriptionChunkRepositoryError(f"Failed to get completed chunk IDs: {exc}") from exc

    async def delete_chunks(
        self,
        meeting_id: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """
        Delete all chunks for a meeting.

        Args:
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier

        Returns:
            Dictionary with deletion result
        """
        try:
            collection = await self._ensure_collection()

            filter_query = {
                "meeting_id": meeting_id,
                "tenant_id": tenant_id
            }

            result = await collection.delete_many(filter_query)

            logger.info(
                f"Deleted {result.deleted_count} chunks for meeting={meeting_id}, tenant={tenant_id}"
            )

            return {
                "meeting_id": meeting_id,
                "tenant_id": tenant_id,
                "deleted_count": result.deleted_count
            }

        except Exception as exc:
            logger.exception(f"Failed to delete chunks for meeting={meeting_id}: {exc}")
            raise TranscriptionChunkRepositoryError(f"Failed to delete chunks: {exc}") from exc

    async def get_chunk_stats(
        self,
        meeting_id: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """
        Get statistics about chunks for a meeting.

        Args:
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier

        Returns:
            Dictionary with chunk statistics
        """
        try:
            collection = await self._ensure_collection()

            pipeline = [
                {
                    "$match": {
                        "meeting_id": meeting_id,
                        "tenant_id": tenant_id
                    }
                },
                {
                    "$group": {
                        "_id": "$status",
                        "count": {"$sum": 1}
                    }
                }
            ]

            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=None)

            stats = {
                "total": 0,
                "success": 0,
                "failed": 0,
                "processing": 0
            }

            for result in results:
                status = result["_id"]
                count = result["count"]
                stats["total"] += count
                if status in stats:
                    stats[status] = count

            logger.info(f"Chunk stats for meeting={meeting_id}: {stats}")
            return stats

        except Exception as exc:
            logger.exception(f"Failed to get chunk stats for meeting={meeting_id}: {exc}")
            raise TranscriptionChunkRepositoryError(f"Failed to get chunk stats: {exc}") from exc
