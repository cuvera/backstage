from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from app.schemas.meeting_analysis import MeetingPrepPack
from app.services.agents.meeting_prep_agent import MeetingPrepAgent, MeetingPrepAgentError

logger = logging.getLogger(__name__)


class MeetingPrepServiceError(Exception):
    """Raised when the meeting prep service cannot complete its task."""


class MeetingPrepService:
    """
    Service layer for meeting preparation packs.
    Orchestrates MeetingPrepAgent and provides MongoDB persistence.
    """

    COLLECTION = "meeting_preparations"

    def __init__(
        self,
        *,
        db: Optional[AsyncIOMotorDatabase] = None,
        collection_name: str = COLLECTION,
    ) -> None:
        self._db = db
        self._collection_name = collection_name
        self._collection: Optional[AsyncIOMotorCollection] = None
        self._agent = MeetingPrepAgent()

    @classmethod
    async def from_default(cls, collection_name: str = COLLECTION) -> "MeetingPrepService":
        from app.db.mongodb import get_database  # lazy import to avoid circular deps

        

        db = await get_database()
        service = cls(db=db, collection_name=collection_name)
        await service.ensure_indexes()
        return service

    async def ensure_indexes(self) -> None:
        """Create MongoDB indexes for optimal query performance."""
        collection = await self._ensure_collection()
        
        # Unique index on tenant_id + recurring_meeting_id for upserts
        await collection.create_index(
            [("tenant_id", ASCENDING), ("recurring_meeting_id", ASCENDING)],
            unique=True,
            name="ux_tenant_recurring_meeting",
        )
        
        # Index for fetching by tenant and meeting_id (for lookup by individual meeting)
        await collection.create_index(
            [("tenant_id", ASCENDING), ("meeting_id", ASCENDING)],
            name="ix_tenant_meeting",
        )
        
        # Index for time-based queries
        await collection.create_index(
            [("created_at", DESCENDING)],
            name="ix_created_at",
        )

    async def generate_and_save_prep_pack(
        self,
        meeting_id: str,
        *,
        recurring_meeting_id: Optional[str] = None,
        previous_meeting_counts: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a meeting prep pack using the agent and save it to MongoDB.
        
        Args:
            meeting_id: ID of the upcoming meeting
            recurring_meeting_id: Optional override for recurring meeting ID
            previous_meeting_counts: Number of previous meetings to analyze
            context: Additional context for processing
            
        Returns:
            Dictionary with save result and prep pack data
        """
        try:
            # Generate prep pack using agent
            prep_pack = self._agent.generate_prep_pack(
                meeting_id=meeting_id,
                recurring_meeting_id=recurring_meeting_id,
                previous_meeting_counts=previous_meeting_counts,
                context=context,
            )
            
            # Save to MongoDB
            save_result = await self.save_prep_pack(prep_pack, meeting_id)
            
            logger.info(
                "[MeetingPrepService] Generated and saved prep pack for meeting=%s recurring=%s",
                meeting_id,
                prep_pack.recurring_meeting_id,
            )
            
            return {
                "prep_pack": prep_pack.dict(),
                "save_result": save_result,
            }
            
        except MeetingPrepAgentError as exc:
            logger.error("[MeetingPrepService] Agent error: %s", exc)
            raise MeetingPrepServiceError(f"Failed to generate prep pack: {exc}") from exc
        except Exception as exc:
            logger.exception("[MeetingPrepService] Unexpected error: %s", exc)
            raise MeetingPrepServiceError(f"Unexpected error: {exc}") from exc

    async def save_prep_pack(self, prep_pack: MeetingPrepPack, meeting_id: str) -> Dict[str, Any]:
        """
        Save a meeting prep pack to MongoDB.
        
        Args:
            prep_pack: The prep pack to save
            meeting_id: The meeting ID this prep pack is for
            
        Returns:
            Dictionary with save operation details
        """
        now = datetime.now(timezone.utc).isoformat()
        doc: Dict[str, Any] = prep_pack.dict(exclude_none=True)
        doc.setdefault("created_at", now)
        doc["updated_at"] = now
        doc["meeting_id"] = meeting_id  # Store the meeting this prep pack is for

        collection = await self._ensure_collection()
        key = {
            "tenant_id": prep_pack.tenant_id,
            "recurring_meeting_id": prep_pack.recurring_meeting_id,
        }

        logger.info(
            "[MeetingPrepService] Upserting prep pack for tenant=%s recurring_meeting=%s",
            prep_pack.tenant_id,
            prep_pack.recurring_meeting_id,
        )

        result = await collection.update_one(key, {"$set": doc}, upsert=True)
        stored = await collection.find_one(key, {"_id": 1})

        mongo_id = stored.get("_id") if stored else result.upserted_id
        return {
            "document_id": str(mongo_id) if mongo_id else None,
            "matched": result.matched_count,
            "upserted": bool(result.upserted_id),
            "collection": self._collection_name,
        }

    async def get_prep_pack_by_meeting_id(
        self, *, tenant_id: str, meeting_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a prep pack by meeting ID.
        
        Args:
            tenant_id: Tenant identifier
            meeting_id: Meeting identifier
            
        Returns:
            Prep pack data or None if not found
        """
        collection = await self._ensure_collection()
        record = await collection.find_one(
            {"tenant_id": tenant_id, "meeting_id": meeting_id},
            {"_id": 0},
        )
        return record

    async def get_prep_pack_by_recurring_meeting_id(
        self, *, tenant_id: str, recurring_meeting_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest prep pack by recurring meeting ID.
        
        Args:
            tenant_id: Tenant identifier
            recurring_meeting_id: Recurring meeting identifier
            
        Returns:
            Latest prep pack data or None if not found
        """
        collection = await self._ensure_collection()
        record = await collection.find_one(
            {"tenant_id": tenant_id, "recurring_meeting_id": recurring_meeting_id},
            {"_id": 0},
        )
        return record

    async def get_prep_packs_by_recurring_meeting_id(
        self, *, tenant_id: str, recurring_meeting_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get multiple prep packs for a recurring meeting series, ordered by creation date.
        
        Args:
            tenant_id: Tenant identifier
            recurring_meeting_id: Recurring meeting identifier
            limit: Maximum number of prep packs to return
            
        Returns:
            List of prep pack data ordered by created_at (newest first)
        """
        collection = await self._ensure_collection()
        cursor = collection.find(
            {"tenant_id": tenant_id, "recurring_meeting_id": recurring_meeting_id},
            {"_id": 0},
        ).sort("created_at", DESCENDING).limit(limit)
        
        records = await cursor.to_list(length=limit)
        return records

    async def delete_prep_pack(
        self, *, tenant_id: str, recurring_meeting_id: str
    ) -> Dict[str, Any]:
        """
        Delete a prep pack by tenant and recurring meeting ID.
        
        Args:
            tenant_id: Tenant identifier
            recurring_meeting_id: Recurring meeting identifier
            
        Returns:
            Dictionary with deletion result
        """
        collection = await self._ensure_collection()
        result = await collection.delete_one({
            "tenant_id": tenant_id,
            "recurring_meeting_id": recurring_meeting_id,
        })
        
        logger.info(
            "[MeetingPrepService] Deleted prep pack for tenant=%s recurring_meeting=%s, deleted_count=%d",
            tenant_id,
            recurring_meeting_id,
            result.deleted_count,
        )
        
        return {
            "deleted_count": result.deleted_count,
            "collection": self._collection_name,
        }

    async def _ensure_collection(self) -> AsyncIOMotorCollection:
        """Ensure the MongoDB collection is available."""
        if self._collection is None:
            if self._db is None:
                from app.db.mongodb import get_database

                self._db = await get_database()
            self._collection = self._db[self._collection_name]
        return self._collection