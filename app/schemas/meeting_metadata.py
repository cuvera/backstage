from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class SpeakerTimeframe(BaseModel):
    """Speaker timeframe within a meeting."""
    speaker_name: str = Field(alias="speakerName")
    start: int
    end: int


class GoogleMeetingDocument(BaseModel):
    """MongoDB document structure for google_meetings collection."""
    event_id: str = Field(alias="eventId")
    summary: str
    start: datetime
    end: datetime
    hangout_link: Optional[str] = Field(alias="hangoutLink", default=None)
    file_url: Optional[str] = Field(alias="fileUrl", default=None)
    tenant_id: str = Field(alias="tenantId")
    status: str
    speaker_timeframes: Optional[List[SpeakerTimeframe]] = Field(alias="speakerTimeframes", default=None)
    organizer: str
    recurring_event_id: Optional[str] = Field(alias="recurringEventId", default=None)
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    bot_ended_at: Optional[datetime] = Field(alias="botEndedAt", default=None)
    
    class Config:
        allow_population_by_field_name = True
        populate_by_name = True


class MeetingMetadata(BaseModel):
    """Standardized meeting metadata response model."""
    id: str
    tenant_id: str
    recurring_meeting_id: Optional[str] = None
    summary: str
    start_time: datetime
    end_time: datetime
    organizer: str
    status: str
    hangout_link: Optional[str] = None
    file_url: Optional[str] = None
    attendees: List[str] = []
    speaker_timeframes: Optional[List[SpeakerTimeframe]] = None
    created_at: datetime
    updated_at: datetime


class PreviousMeeting(BaseModel):
    """Previous meeting data structure for prep pack generation."""
    id: str
    recurring_meeting_id: str
    datetime: str | datetime
    session_id: str


def google_meeting_to_metadata(doc: Dict[str, Any]) -> MeetingMetadata:
    """Convert Google meeting MongoDB document to standardized metadata."""
    return MeetingMetadata(
        id=str(doc.get("_id")),
        tenant_id=doc.get("tenantId"),
        recurring_meeting_id=doc.get("recurringEventId"),
        summary=doc.get("summary", ""),
        start_time=doc.get("start"),
        end_time=doc.get("end"),
        organizer=doc.get("organizer", ""),
        status=doc.get("status", ""),
        hangout_link=doc.get("hangoutLink"),
        file_url=doc.get("fileUrl"),
        attendees=doc.get("attendees", []),
        speaker_timeframes=[
            SpeakerTimeframe(**tf) for tf in doc.get("speakerTimeframes", [])
        ] if doc.get("speakerTimeframes") else None,
        created_at=doc.get("createdAt"),
        updated_at=doc.get("updatedAt")
    )


def google_meeting_to_previous_meeting(doc: Dict[str, Any]) -> PreviousMeeting:
    """Convert Google meeting MongoDB document to previous meeting structure."""
    return PreviousMeeting(
        id=str(doc.get("_id")),
        recurring_meeting_id=doc.get("recurringEventId", ""),
        datetime=str(doc.get("start")),
        session_id=doc.get("eventId")  # Using eventId as session_id for now
    )