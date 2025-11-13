import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from bson.objectid import ObjectId

from app.schemas.meeting_metadata import (
    google_meeting_to_metadata,
    google_meeting_to_previous_meeting,
    MeetingMetadata,
    PreviousMeeting
)
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class MeetingMetadataRepository(BaseRepository):
    """Repository for meeting metadata operations across different platforms."""

    # Platform-specific collection mapping
    PLATFORM_COLLECTIONS = {
        "google": "google_meetings",
    }

    def __init__(
        self,
        *,
        db: Optional[AsyncIOMotorDatabase] = None,
    ) -> None:
        # Note: This repository doesn't use a single collection, so we pass empty collection_name
        super().__init__(db=db, collection_name="")

    @classmethod
    async def from_default(cls) -> "MeetingMetadataRepository":
        from app.db.mongodb import get_database

        db = await get_database()
        repository = cls(db=db)
        return repository

    async def get_speaker_timeframes(
        self,
        meeting_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch speaker timeframes from Google meeting document.
        
        Args:
            meeting_id: Meeting identifier (_id or eventId)
            
        Returns:
            List of speaker timeframes or None if not found
            Format: [{"speakerName": "Name", "start": ms, "end": ms}, ...]
        """
        try:
            collection_name = self.PLATFORM_COLLECTIONS["google"]
            collection = self._db[collection_name]
            
            # Try to find by _id first (ObjectId)
            query = {"_id": ObjectId(meeting_id)}
            meeting_doc = await collection.find_one(query)
            
            # If not found, try by eventId (string)
            if not meeting_doc:
                query = {"eventId": meeting_id}
                meeting_doc = await collection.find_one(query)
            
            if meeting_doc:
                speaker_timeframes = meeting_doc.get("speakerTimeframes", [])
                logger.info(f"Found {len(speaker_timeframes)} speaker timeframes for meeting {meeting_id}")
                return speaker_timeframes
            else:
                logger.warning(f"Google meeting not found for ID: {meeting_id}")
                return None
                
        except Exception as exc:
            logger.error(f"Error fetching Google meeting timeframes for {meeting_id}: {exc}")
            return None

    async def get_meeting_metadata(
        self,
        meeting_id: str,
        platform: str = "google"
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch meeting metadata from MongoDB collection by meeting ID.
        
        Args:
            meeting_id: Individual meeting identifier
            platform: Meeting platform (google, zoom, etc.)
            
        Returns:
            Meeting metadata or None if not found/error
        """
        collection_name = self.PLATFORM_COLLECTIONS.get(platform)
        if not collection_name:
            logger.error("Unsupported platform: %s", platform)
            return None
        
        collection = self._db[collection_name]
        
        try:
            # Query by _id for MongoDB ObjectId
            query_field = "_id"
            doc = await collection.find_one({query_field: ObjectId(meeting_id)})

            if not doc:
                logger.warning("Meeting not found: meeting_id=%s, platform=%s", meeting_id, platform)
                return None

            # Convert to standardized format
            if platform == "google":
                metadata = google_meeting_to_metadata(doc)
                return metadata.model_dump()
            
            logger.info("Fetched meeting metadata for meeting_id=%s, platform=%s", meeting_id, platform)
            return doc  # Return raw doc for other platforms
            
        except Exception as exc:
            logger.error("Error fetching meeting %s from %s: %s", meeting_id, platform, exc)
            return None

    async def get_meetings_by_recurring_id(
        self,
        recurring_meeting_id: str,
        *,
        platform: str = "google",
        limit: Optional[int] = None,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch meetings for a recurring meeting series with optional filters.
        
        Args:
            recurring_meeting_id: Recurring meeting identifier
            platform: Meeting platform
            limit: Maximum number of meetings to return
            start: Start date filter (ISO format)
            end: End date filter (ISO format)
            
        Returns:
            List of meetings or None if not found/error
        """
        collection_name = self.PLATFORM_COLLECTIONS.get(platform)
        if not collection_name:
            logger.error("Unsupported platform: %s", platform)
            return None
        
        collection = self._db[collection_name]
        
        try:
            # Build query
            query = {"recurringEventId": recurring_meeting_id}
            
            # Add date filters
            if start or end:
                date_filter = {}
                if start:
                    date_filter["$gte"] = datetime.fromisoformat(start.replace('Z', '+00:00'))
                if end:
                    date_filter["$lte"] = datetime.fromisoformat(end.replace('Z', '+00:00'))
                query["start"] = date_filter
            
            # Execute query with sorting and limit
            cursor = collection.find(query).sort("start", 1)  # Sort by start time ascending
            if limit:
                cursor = cursor.limit(limit)
            
            docs = await cursor.to_list(length=limit or 1000)
            
            if not docs:
                logger.warning("No meetings found for recurring_meeting_id=%s", recurring_meeting_id)
                return []
            
            # Convert to standardized format
            if platform == "google":
                meetings = [google_meeting_to_previous_meeting(doc).model_dump() for doc in docs]
            else:
                meetings = docs  # Raw docs for other platforms
            
            logger.info("Fetched %d meetings for recurring_meeting_id=%s", len(meetings), recurring_meeting_id)
            return meetings
            
        except Exception as exc:
            logger.error("Error fetching recurring meetings %s: %s", recurring_meeting_id, exc)
            return None

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
            logger.warning("[MeetingMetadataRepository] No current meeting metadata provided")
            return None
        
        # Extract end time from current meeting
        current_end_time = (
            current_meeting_metadata.get("end_time") or 
            current_meeting_metadata.get("end")
        )
        
        if not current_end_time:
            logger.warning("[MeetingMetadataRepository] Current meeting has no end time")
            return None
        
        # Query next scheduled meeting
        collection_name = self.PLATFORM_COLLECTIONS.get(platform, "google_meetings")
        collection = self._db[collection_name]
        
        try:
            query = {
                "recurringEventId": recurring_meeting_id,
                # "status": "scheduled",
                "start": {"$gte": current_end_time},
                "eventId": {"$ne": current_meeting_metadata.get("id")}
            }
            

            print("Next meeting query: ", query)


            # Get immediate next meeting
            cursor = collection.find(query, {"eventId": 1}).sort("start", 1).limit(1)
            docs = await cursor.to_list(length=1)
            
            if docs:
                print(f"Found next meeting: {docs}")
                next_meeting_id = docs[0].get("_id")
                logger.info("[MeetingMetadataRepository] Found immediate next meeting: %s", next_meeting_id)
                return next_meeting_id
            
            logger.info("[MeetingMetadataRepository] No immediate next scheduled meeting found")
            return None
            
        except Exception as exc:
            logger.error("[MeetingMetadataRepository] Error finding next meeting: %s", exc)
            return None

    async def update_meeting_status(
        self,
        meeting_id: str,
        platform: str,
        status: str
    ) -> bool:
        """
        Update meeting status in the platform-specific collection.
        
        Args:
            meeting_id: Meeting identifier
            platform: Meeting platform
            status: New status value ('completed', 'failed', etc.)
            
        Returns:
            True if update was successful, False otherwise
        """
        collection_name = self.PLATFORM_COLLECTIONS.get(platform)
        if not collection_name:
            logger.error("[MeetingMetadataRepository] Unsupported platform: %s", platform)
            return False
        
        collection = self._db[collection_name]
        
        try:
            # Update by _id (ObjectId)
            result = await collection.update_one(
                {"_id": ObjectId(meeting_id)},
                {"$set": {"status": status}}
            )
            
            if result.matched_count > 0:
                logger.info(
                    "[MeetingMetadataRepository] Updated meeting %s status to %s", 
                    meeting_id, status
                )
                return True
            else:
                logger.warning(
                    "[MeetingMetadataRepository] Meeting %s not found for status update", 
                    meeting_id
                )
                return False
                
        except Exception as exc:
            logger.error(
                "[MeetingMetadataRepository] Error updating meeting %s status: %s", 
                meeting_id, exc
            )
            return False