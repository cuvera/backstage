from typing import Dict, Any, Optional


# Platform-specific collection mapping (moved to MeetingMetadataRepository)
PLATFORM_COLLECTIONS = {
    "google": "google_meetings",
}


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