import httpx
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

BASE_URL = "https://api-dev.cuvera.ai/meeting-bot-service/api/v1/meetings"

async def fetch_meeting_metadata(meeting_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch meeting metadata from external API by meeting ID.
    
    Args:
        meeting_id: Individual meeting identifier
        
    Returns:
        Meeting metadata or None if not found/error
    """
    url = f"{BASE_URL}/{meeting_id}"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            logger.info("Fetched meeting metadata for meeting_id=%s", meeting_id)
            return data
        except httpx.RequestError as exc:
            logger.error("Request error fetching meeting %s: %s", meeting_id, exc)
            return None
        except httpx.HTTPStatusError as exc:
            logger.error("HTTP error fetching meeting %s: %s", meeting_id, exc)
            return None


async def fetch_meetings_by_recurring_id(
    recurring_meeting_id: str,
    *,
    limit: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch meetings for a recurring meeting series with optional filters.
    
    Args:
        recurring_meeting_id: Recurring meeting identifier
        limit: Maximum number of meetings to return
        start: Start date filter (ISO format)
        end: End date filter (ISO format)
        
    Returns:
        List of meetings or None if not found/error
    """
    url = f"{BASE_URL}/recurring/{recurring_meeting_id}"
    
    # Build query parameters
    params = {}
    if limit is not None:
        params["limit"] = limit
    if start is not None:
        params["start"] = start
    if end is not None:
        params["end"] = end
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            logger.info("Fetched %d meetings for recurring_meeting_id=%s with filters: limit=%s, start=%s, end=%s", 
                       len(data) if isinstance(data, list) else 1, recurring_meeting_id, limit, start, end)
            return data
        except httpx.RequestError as exc:
            logger.error("Request error fetching recurring meetings %s: %s", recurring_meeting_id, exc)
            return None
        except httpx.HTTPStatusError as exc:
            logger.error("HTTP error fetching recurring meetings %s: %s", recurring_meeting_id, exc)
            return None


def extract_tenant_id(meeting_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract tenant_id from meeting metadata.
    
    Args:
        meeting_data: Meeting metadata from API
        
    Returns:
        Tenant ID or None if not found
    """
    return meeting_data.get("tenant_id") or meeting_data.get("tenantId")


def extract_recurring_meeting_id(meeting_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract recurring_meeting_id from meeting metadata.
    
    Args:
        meeting_data: Meeting metadata from API
        
    Returns:
        Recurring meeting ID or None if not found
    """
    return meeting_data.get("recurringEventId")