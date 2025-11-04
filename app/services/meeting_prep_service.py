from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from app.schemas.meeting_analysis import MeetingAnalysis, MeetingPrepPack
from app.services.agents.meeting_prep_agent import MeetingPrepAgent, MeetingPrepAgentError
from app.services.meeting_analysis_service import MeetingAnalysisService
from app.utils.meeting_metadata import (
    fetch_meeting_metadata,
    fetch_meetings_by_recurring_id,
    extract_recurring_meeting_id,
    PLATFORM_COLLECTIONS,
)

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
        self._analysis_service = MeetingAnalysisService(db=db)

    @classmethod
    async def from_default(cls, collection_name: str = COLLECTION) -> "MeetingPrepService":
        from app.db.mongodb import get_database  # lazy import to avoid circular deps

        db = await get_database()
        service = cls(db=db, collection_name=collection_name)
        await service.ensure_indexes()
        service._analysis_service = MeetingAnalysisService(db=db)
        await service._analysis_service.ensure_indexes()
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
        platform: str,
        *,
        recurring_meeting_id: Optional[str] = None,
        previous_meeting_counts: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a meeting prep pack using the agent and save it to MongoDB.
        
        Args:
            meeting_id: ID of the upcoming meeting
            platform: Meeting platform (e.g., 'google', 'zoom', 'offline')
            recurring_meeting_id: Optional override for recurring meeting ID
            previous_meeting_counts: Number of previous meetings to analyze
            context: Additional context for processing
            
        Returns:
            Dictionary with save result and prep pack data
        """
        try:
            # Only for online platforms (currently supporting Google's recurring meetings)
            if platform != "google":
                logger.warning("Meeting Prep Service is not implemented for platform=%s", platform)
                return {
                    "status": "skipped",
                    "prep_pack": None,
                    "save_result": None,
                }

            # Fetch meeting metadata based on platform
            meeting_metadata = await self._get_meeting_metadata(meeting_id, platform)
            
            # Resolve recurring meeting ID
            resolved_recurring_id = self._resolve_recurring_meeting_id(
                meeting_metadata, recurring_meeting_id
            )
            
            # Fetch previous meetings and their analyses based on platform
            counts = previous_meeting_counts or self._agent.previous_meeting_counts
            previous_meetings = await self._get_previous_meetings(
                resolved_recurring_id, 
                counts, 
                platform=platform
            )
            previous_analyses = await self._get_meeting_analyses(previous_meetings)
            
            # Generate prep pack using agent
            prep_pack = self._agent.generate_prep_pack(
                meeting_metadata=meeting_metadata,
                previous_analyses=previous_analyses,
                recurring_meeting_id=resolved_recurring_id,
                previous_meetings=previous_meetings,
                context=context,
            )
            
            # Find immediate next meeting using already-fetched current meeting data
            next_meeting_id = await self._find_immediate_next_meeting(
                meeting_metadata,
                resolved_recurring_id,
                platform
            )
            
            # Use next meeting ID if found, otherwise use current meeting_id
            target_meeting_id = next_meeting_id or meeting_id
            
            # Save to MongoDB
            save_result = await self.save_prep_pack(prep_pack, target_meeting_id)
            
            logger.info(
                "[MeetingPrepService] Generated prep pack for meeting=%s, saved for target=%s (next=%s), recurring=%s",
                meeting_id,
                target_meeting_id,
                "found" if next_meeting_id else "not found",
                prep_pack.recurring_meeting_id,
            )
            
            return {
                "prep_pack": prep_pack.model_dump(),
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
        doc: Dict[str, Any] = prep_pack.model_dump(exclude_none=True)
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

    def _resolve_recurring_meeting_id(
        self, meeting_metadata: Dict[str, Any], recurring_meeting_id: Optional[str] = None
    ) -> str:
        """Resolve recurring meeting ID from meeting details if not provided."""
        if recurring_meeting_id:
            return recurring_meeting_id
        
        # Use utility function to extract recurring meeting ID
        resolved_id = extract_recurring_meeting_id(meeting_metadata)
        
        if not resolved_id:
            logger.warning("No recurring_meeting_id found in meeting metadata")
            return None

        return resolved_id

    async def _get_meeting_metadata(self, meeting_id: str, platform: str) -> Dict[str, Any]:
        """
        Fetch meeting metadata from MongoDB collection.
        
        Args:
            meeting_id: Individual meeting identifier
            platform: Meeting platform (e.g., 'google', 'zoom', 'offline')
            
        Returns:
            Meeting metadata dictionary
        """
        # Only fetch from MongoDB for online platforms
        if platform != "offline" and platform == "google":
            logger.info("Fetching meeting metadata from MongoDB for platform=%s, meeting_id=%s", platform, meeting_id)
            meeting_data = await fetch_meeting_metadata(meeting_id, self._db, platform)
            
            if meeting_data:
                return meeting_data
            
            logger.warning("Failed to fetch meeting metadata from MongoDB, using placeholder data")
        
        # Return None for offline or when MongoDB query fails
        logger.info("Using placeholder meeting metadata for platform=%s, meeting_id=%s", platform, meeting_id)
        return None

    async def _get_previous_meetings(
        self, 
        recurring_meeting_id: str, 
        count: int, 
        platform: str = "google",
        from_date: Optional[str] = None, 
        to_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch previous meetings for the recurring meeting series from MongoDB.
        
        Args:
            recurring_meeting_id: Recurring meeting identifier
            count: Number of previous meetings to fetch
            platform: Meeting platform (e.g., 'google', 'zoom', 'offline')
            from_date: Start date filter (ISO format)
            to_date: End date filter (ISO format)
            
        Returns:
            List of previous meeting data
        """        
        meetings_data = await fetch_meetings_by_recurring_id(
            recurring_meeting_id,
            self._db,
            platform=platform,
            limit=count,
            start=from_date,
            end=to_date,
        )
        
        if meetings_data and isinstance(meetings_data, list):
            logger.info("Fetched %d previous meetings from MongoDB", len(meetings_data))
            return meetings_data[:count]  # Ensure we don't exceed requested count
        
        logger.warning("Failed to fetch previous meetings from MongoDB or no meetings found")
        return []

    async def _get_meeting_analyses(self, meetings: List[Dict[str, Any]]) -> List[MeetingAnalysis]:
        """
        Fetch meeting analyses for the given meetings from MongoDB using MeetingAnalysisService.
        
        Args:
            meetings: List of meeting data with session_id fields
            
        Returns:
            List of MeetingAnalysis objects
        """
        if not meetings:
            logger.warning("[MeetingPrepService] No meetings provided for analysis fetch")
            return []
        
        # Extract session IDs and tenant_id from meetings
        session_ids = []
        tenant_id = None
        
        for meeting in meetings:
            session_id = meeting.get("session_id")
            if session_id:
                session_ids.append(session_id)
            
            # Extract tenant_id for security filtering (should be same for all meetings)
            if not tenant_id:
                tenant_id = meeting.get("tenant_id")
        
        if not session_ids:
            logger.warning("[MeetingPrepService] No valid session_ids found in meetings data")
            return []
        
        # Fetch analyses using MeetingAnalysisService
        try:
            analysis_docs = await self._analysis_service.get_analyses_by_session_ids(
                session_ids=session_ids,
                tenant_id=tenant_id
            )
            
            # Convert documents to MeetingAnalysis objects
            analyses = []
            for doc in analysis_docs:
                try:
                    # The document should already be in the correct format for MeetingAnalysis
                    analysis = MeetingAnalysis(**doc)
                    analyses.append(analysis)
                except Exception as e:
                    logger.warning(
                        "[MeetingPrepService] Failed to parse analysis for session %s: %s",
                        doc.get("session_id"),
                        e
                    )
                    continue
            
            logger.info(
                "[MeetingPrepService] Retrieved %d analyses for %d meetings",
                len(analyses),
                len(meetings)
            )
            return analyses
            
        except Exception as exc:
            logger.error(
                "[MeetingPrepService] Error fetching analyses: %s",
                exc
            )
            return []

    async def _find_immediate_next_meeting(
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
            logger.warning("[MeetingPrepService] No current meeting metadata provided")
            return None
        
        # Extract end time from current meeting
        current_end_time = (
            current_meeting_metadata.get("end_time") or 
            current_meeting_metadata.get("end")
        )
        
        if not current_end_time:
            logger.warning("[MeetingPrepService] Current meeting has no end time")
            return None
        
        # Query next scheduled meeting
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
                next_meeting_id = docs[0].get("eventId")
                logger.info("[MeetingPrepService] Found immediate next meeting: %s", next_meeting_id)
                return next_meeting_id
            
            logger.info("[MeetingPrepService] No immediate next scheduled meeting found")
            return None
            
        except Exception as exc:
            logger.error("[MeetingPrepService] Error finding next meeting: %s", exc)
            return None

    async def _ensure_collection(self) -> AsyncIOMotorCollection:
        """Ensure the MongoDB collection is available."""
        if self._collection is None:
            if self._db is None:
                from app.db.mongodb import get_database

                self._db = await get_database()
            self._collection = self._db[self._collection_name]
        return self._collection