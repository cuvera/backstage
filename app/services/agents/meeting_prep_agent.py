from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.core.prompts import MEETING_PREP_SUGGESTION_PROMPT

from pydantic import ValidationError

from app.schemas.meeting_analysis import (
    BlockingItem,
    BlockingItemStatus,
    ExpectedOutcome,
    MeetingAnalysis,
    MeetingPrepPack,
    OutcomeType,
    PreviousMeetingReference,
    SeverityLevel,
)
from app.core.openai_client import llm_client

logger = logging.getLogger(__name__)


class MeetingPrepAgentError(Exception):
    """Raised when the meeting prep agent cannot complete its task."""


class MeetingPrepAgent:
    """
    Meeting preparation agent that generates executive prep packs by analyzing
    previous meeting data for recurring meetings.
    
    - Fetches meeting metadata and historical meeting analyses
    - Uses LLM to synthesize insights and generate structured prep recommendations
    - Validates and returns structured MeetingPrepPack objects
    """

    DEFAULT_PREVIOUS_MEETING_COUNTS = 3

    def __init__(self, llm=None, previous_meeting_counts: int = None) -> None:
        self.client = llm or llm_client
        self.model = "gemini-2.5-pro"
        self.previous_meeting_counts = previous_meeting_counts or self.DEFAULT_PREVIOUS_MEETING_COUNTS

    async def generate_prep_pack(
        self,
        meeting_metadata: Dict[str, Any],
        previous_analyses: List[MeetingAnalysis],
        recurring_meeting_id: str,
        *,
        previous_meetings: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MeetingPrepPack:
        """
        Generate an executive prep pack for an upcoming meeting.
        
        Args:
            meeting_metadata: Meeting metadata dict
            previous_analyses: List of previous meeting analyses
            recurring_meeting_id: Recurring meeting identifier
            previous_meetings: Optional list of previous meeting data for references
            context: Additional context for processing
            
        Returns:
            MeetingPrepPack with synthesized insights and recommendations
        """
        context_parts = []

        meeting_info = (
            f"Title: {meeting_metadata.get('title', 'Unknown')}\n"
            f"Scheduled: {meeting_metadata.get('scheduled_datetime', 'Unknown')}\n"
            f"Attendees: {', '.join(meeting_metadata.get('attendees', []))}\n"
        )
        context_parts.append(meeting_info)
        
        previous_meetings_context = ""
        if previous_analyses:
            previous_meetings_context = "PREVIOUS MEETINGS DATA:\n"
            for i, analysis in enumerate(previous_analyses, 1):
                previous_meetings_context += f"\nMeeting {i} ({analysis.created_at}):\n"
                previous_meetings_context += f"Summary: {analysis.summary}\n"
                previous_meetings_context += f"Key Points: {', '.join(analysis.key_points)}\n"
                previous_meetings_context += f"Action Items: {', '.join([item.task for item in analysis.action_items])}\n"

            context_parts.append(previous_meetings_context)

        context_message = "\n".join(context_parts)

        # Make API call to Gemini (copied from orchestrator lines 211-237)
        response = await self.client.chat.completions.create(
            model=self.model,
            reasoning_effort="low",
            messages=[
                {
                    "role": "system",
                    "content": MEETING_PREP_SUGGESTION_PROMPT
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": context_message
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        # Build and validate prep pack
        prep_pack = self._build_prep_pack(
            parsed_response=json.loads(response.choices[0].message.content),
            meeting_metadata=meeting_metadata,
            recurring_meeting_id=recurring_meeting_id,
            previous_meetings=previous_meetings,
        )
        
        return prep_pack

    def _strip_code_fence(self, text: str) -> str:
        """Remove code fence markers from text."""
        if text.startswith("```") and text.endswith("```"):
            inner = text.strip("`")
            parts = inner.split("\n", 1)
            return parts[1] if len(parts) == 2 else parts[0]
        return text

    def _build_prep_pack(
        self,
        *,
        parsed_response: Dict[str, Any],
        meeting_metadata: Dict[str, Any],
        recurring_meeting_id: str,
        previous_meetings: List[Dict[str, Any]],
    ) -> MeetingPrepPack:
        """Build and validate the MeetingPrepPack from parsed response."""
        
        now = datetime.now(timezone.utc).isoformat()
        
        # Build previous meeting references
        previous_refs = []
        for meeting in previous_meetings:
            meeting_id = meeting.get("id")
            analysis_id = meeting.get("session_id")
            datetime_str = meeting.get("start_date") or meeting.get("start") or meeting.get("datetime") or "unknown"
            
            # Only include meetings that have valid required fields
            if meeting_id and analysis_id and datetime_str and len(datetime_str) > 0:
                previous_refs.append(
                    PreviousMeetingReference(
                        meeting_id=meeting_id,
                        analysis_id=analysis_id,
                        datetime=datetime_str,
                    )
                )
        
        # Extract and validate data from LLM response
        prep_pack_data = {
            "title": parsed_response.get("title", meeting_metadata.get("title", "Meeting Prep Pack")),
            "tenant_id": meeting_metadata.get("tenant_id", ""),
            "recurring_meeting_id": recurring_meeting_id,
            "purpose": parsed_response.get("purpose", ""),
            "expected_outcomes": self._build_expected_outcomes(parsed_response.get("expected_outcomes", [])),
            "blocking_items": self._build_blocking_items(parsed_response.get("blocking_items", [])),
            "key_points": self._ensure_str_list(parsed_response.get("key_points", [])),
            "open_questions": self._ensure_str_list(parsed_response.get("open_questions", [])),
            "previous_meetings_ref": previous_refs,
            "created_at": now,
            "updated_at": now,
        }
        
        try:
            return MeetingPrepPack(**prep_pack_data)
        except ValidationError as exc:
            logger.exception("[MeetingPrepAgent] Prep pack validation failed: %s", exc)
            raise MeetingPrepAgentError(f"prep pack validation failed: {exc}") from exc

    def _build_expected_outcomes(self, outcomes_data: List[Dict[str, Any]]) -> List[ExpectedOutcome]:
        """Build expected outcomes from parsed data."""
        outcomes = []
        if not isinstance(outcomes_data, list):
            return outcomes
        
        for outcome_dict in outcomes_data:
            if not isinstance(outcome_dict, dict):
                continue
            
            description = outcome_dict.get("description", "").strip()
            if not description:
                continue
            
            outcome_type = outcome_dict.get("type", "decision").strip().lower()
            try:
                type_enum = OutcomeType(outcome_type)
            except ValueError:
                type_enum = OutcomeType.decision
            
            outcome = ExpectedOutcome(
                description=description,
                owner=outcome_dict.get("owner", "").strip(),
                type=type_enum,
            )
            outcomes.append(outcome)
        
        return outcomes

    def _build_blocking_items(self, blocking_data: List[Dict[str, Any]]) -> List[BlockingItem]:
        """Build blocking items from parsed data."""
        items = []
        if not isinstance(blocking_data, list):
            return items
        
        for item_dict in blocking_data:
            if not isinstance(item_dict, dict):
                continue
            
            title = item_dict.get("title", "").strip()
            if not title:
                continue
            
            severity_str = item_dict.get("severity", "medium").strip().lower()
            try:
                severity = SeverityLevel(severity_str)
            except ValueError:
                severity = SeverityLevel.medium
            
            status_str = item_dict.get("status", "open").strip().lower()
            try:
                status = BlockingItemStatus(status_str)
            except ValueError:
                status = BlockingItemStatus.open
            
            item = BlockingItem(
                title=title,
                owner=item_dict.get("owner", "").strip(),
                eta=item_dict.get("eta", "").strip(),
                impact=item_dict.get("impact", "").strip(),
                severity=severity,
                status=status,
            )
            items.append(item)
        
        return items

    def _ensure_str_list(self, value: Any) -> List[str]:
        """Ensure value is a list of non-empty strings."""
        items = []
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                items.append(candidate)
        elif isinstance(value, list):
            for entry in value:
                if isinstance(entry, str):
                    candidate = entry.strip()
                    if candidate:
                        items.append(candidate)
        return items
