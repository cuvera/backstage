from motor.motor_asyncio import AsyncIOMotorDatabase
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from bson.objectid import ObjectId

from app.schemas.meeting_metadata import (
    google_meeting_to_metadata,
    google_meeting_to_previous_meeting,
    MeetingMetadata,
    PreviousMeeting
)

logger = logging.getLogger(__name__)

# Platform-specific collection mapping
PLATFORM_COLLECTIONS = {
    "google": "google_meetings",
}

async def fetch_meeting_metadata(
    meeting_id: str, 
    db: AsyncIOMotorDatabase,
    platform: str = "google"
) -> Optional[Dict[str, Any]]:
    """
    Fetch meeting metadata from MongoDB collection by meeting ID.
    
    Args:
        meeting_id: Individual meeting identifier
        db: MongoDB database instance
        platform: Meeting platform (google, zoom, etc.)
        
    Returns:
        Meeting metadata or None if not found/error
    """
    collection_name = PLATFORM_COLLECTIONS.get(platform)
    if not collection_name:
        logger.error("Unsupported platform: %s", platform)
        return None
    
    collection = db[collection_name]
    
    try:
        # Query by eventId for Google meetings
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


async def fetch_meetings_by_recurring_id(
    recurring_meeting_id: str,
    db: AsyncIOMotorDatabase,
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
        db: MongoDB database instance
        platform: Meeting platform
        limit: Maximum number of meetings to return
        start: Start date filter (ISO format)
        end: End date filter (ISO format)
        
    Returns:
        List of meetings or None if not found/error
    """
    collection_name = PLATFORM_COLLECTIONS.get(platform)
    if not collection_name:
        logger.error("Unsupported platform: %s", platform)
        return None
    
    collection = db[collection_name]
    
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


def extract_tenant_id(meeting_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract tenant_id from meeting metadata.
    
    Args:
        meeting_data: Meeting metadata from database
        
    Returns:
        Tenant ID or None if not found
    """
    return meeting_data.get("tenant_id") or meeting_data.get("tenantId")


def extract_recurring_meeting_id(meeting_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract recurring_meeting_id from meeting metadata.
    
    Args:
        meeting_data: Meeting metadata from database
        
    Returns:
        Recurring meeting ID or None if not found
    """
    return meeting_data.get("recurring_meeting_id") or meeting_data.get("recurringEventId")