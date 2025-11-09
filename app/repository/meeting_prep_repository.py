import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING
from bson.objectid import ObjectId

from app.schemas.meeting_analysis import MeetingPrepPack
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class MeetingPrepRepository(BaseRepository):
    """Repository for meeting preparation data persistence."""

    COLLECTION = "meeting_preparations"

    def __init__(
        self,
        *,
        db: Optional[AsyncIOMotorDatabase] = None,
        collection_name: str = COLLECTION,
    ) -> None:
        super().__init__(db=db, collection_name=collection_name)

    @classmethod
    async def from_default(cls, collection_name: str = COLLECTION) -> "MeetingPrepRepository":
        from app.db.mongodb import get_database

        db = await get_database()
        repository = cls(db=db, collection_name=collection_name)
        await repository.ensure_indexes()
        return repository

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
        doc: Dict[str, Any] = prep_pack.model_dump(exclude_none=True)
        doc.setdefault("created_at", now)
        doc["updated_at"] = now
        doc["tenant_id"] = ObjectId(prep_pack.tenant_id)
        doc["meeting_id"] = meeting_id  # Store the meeting this prep pack is for

        collection = await self._ensure_collection()
        key = {
            "tenant_id": ObjectId(prep_pack.tenant_id),
            "recurring_meeting_id": prep_pack.recurring_meeting_id,
        }

        logger.info(
            "[MeetingPrepRepository] Upserting prep pack for tenant=%s recurring_meeting=%s",
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
            "[MeetingPrepRepository] Deleted prep pack for tenant=%s recurring_meeting=%s, deleted_count=%d",
            tenant_id,
            recurring_meeting_id,
            result.deleted_count,
        )
        
        return {
            "deleted_count": result.deleted_count,
            "collection": self._collection_name,
        }

    async def find_immediate_next_meeting(
        self,
        current_meeting_metadata: Dict[str, Any],
        recurring_meeting_id: str,
        platform: str = "google"
    ) -> Optional[str]:
        """
        Find the immediate next scheduled meeting after the current meeting ends.
        
        Args:
            current_meeting_metadata: Already fetched current meeting data
            recurring_meeting_id: Recurring meeting series ID
            platform: Meeting platform
            
        Returns:
            Next meeting ID or None
        """
        if not current_meeting_metadata:
            logger.warning("[MeetingPrepRepository] No current meeting metadata provided")
            return None
        
        # Extract end time from current meeting
        current_end_time = (
            current_meeting_metadata.get("end_time") or 
            current_meeting_metadata.get("end")
        )
        
        if not current_end_time:
            logger.warning("[MeetingPrepRepository] Current meeting has no end time")
            return None
        
        # Query next scheduled meeting
        from app.utils.meeting_metadata import PLATFORM_COLLECTIONS
        collection_name = PLATFORM_COLLECTIONS.get(platform, "google_meetings")
        collection = self._db[collection_name]
        
        try:
            query = {
                "recurringEventId": recurring_meeting_id,
                "status": "scheduled",
                "start": {"$gte": current_end_time},
                "eventId": {"$ne": current_meeting_metadata.get("id")}
            }
            
            # Get immediate next meeting
            cursor = collection.find(query, {"eventId": 1}).sort("start", 1).limit(1)
            docs = await cursor.to_list(length=1)
            
            if docs:
                print(f"Found next meeting: {docs}")
                next_meeting_id = docs[0].get("_id")
                logger.info("[MeetingPrepRepository] Found immediate next meeting: %s", next_meeting_id)
                return next_meeting_id
            
            logger.info("[MeetingPrepRepository] No immediate next scheduled meeting found")
            return None
            
        except Exception as exc:
            logger.error("[MeetingPrepRepository] Error finding next meeting: %s", exc)
            return None