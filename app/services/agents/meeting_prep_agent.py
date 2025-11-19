from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
from app.services.llm.factory import get_llm
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
        self.llm = llm or get_llm()
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
        ctx = context or {}
        prev_meetings = previous_meetings or []
        
        # Generate prep pack using LLM
        prompt = self._build_prompt(meeting_metadata, previous_analyses, ctx)
        raw_response = await self._call_llm(prompt)
        parsed_response = self._parse_response(raw_response)
        
        # Build and validate prep pack
        prep_pack = self._build_prep_pack(
            parsed_response=parsed_response,
            meeting_metadata=meeting_metadata,
            recurring_meeting_id=recurring_meeting_id,
            previous_meetings=prev_meetings,
            context=ctx,
        )
        
        return prep_pack


    def _build_prompt(
        self,
        meeting_metadata: Dict[str, Any],
        previous_analyses: List[MeetingAnalysis],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the LLM prompt for generating the prep pack."""
        
        meeting_info = (
            f"Title: {meeting_metadata.get('title', 'Unknown')}\n"
            f"Scheduled: {meeting_metadata.get('scheduled_datetime', 'Unknown')}\n"
            f"Attendees: {', '.join(meeting_metadata.get('attendees', []))}\n"
        )
        
        previous_meetings_context = ""
        if previous_analyses:
            previous_meetings_context = "PREVIOUS MEETINGS DATA:\n"
            for i, analysis in enumerate(previous_analyses, 1):
                previous_meetings_context += f"\nMeeting {i} ({analysis.created_at}):\n"
                previous_meetings_context += f"Summary: {analysis.summary}\n"
                previous_meetings_context += f"Key Points: {', '.join(analysis.key_points)}\n"
                previous_meetings_context += f"Action Items: {', '.join([item.task for item in analysis.action_items])}\n"
                # previous_meetings_context += f"Open Questions: {', '.join(analysis.open_questions)}\n" # Removed as per request to simplify
                # previous_meetings_context += f"Topics: {', '.join(analysis.topics)}\n" # Removed as per request to simplify
        else:
            previous_meetings_context = "No previous meeting data available.\n"

        prompt = f"""You are an expert meeting operations assistant. Produce an Executive Prep Pack (one-pager) for an upcoming online, recurring meeting. Your output must be a single valid JSON object only (no prose), follow the schema below, reference at least 1 previous meetings.

        INPUTS
        You will receive a JSON object with:
            - meeting: upcoming meeting metadata.
            - attendees: tentative
            - signals: machine-extracted summaries from ≥3 prior meetings (per-meeting summaries, topics, decisions, action items).

        STYLE GUIDE (CRITICAL)
        - **Tone:** Executive, concise, action-oriented. No corporate fluff.
        - **Format:** Use "Status: Context" for bullets (e.g., "Frontend: 50% Complete (On Track)").
        - **Quantify:** Use numbers, dates, and percentages where possible.
        - **Voice:** Active voice. "Bob to fix bug" instead of "Bug should be fixed by Bob".

        WHAT TO DO
        - Synthesize purpose and today_outcomes (decision/approval/alignment) from inputs and last 1 or more meetings.
        - Identify blocking_items (owner, ETA, impact) that must be cleared to decide today.
        - **Key Points:** Generate 3-5 high-impact bullet points. Focus strictly on **DELTAS** (what changed since last time) and **PROGRESS**.
        - **Open Questions:** Generate exactly 3 critical strategic questions that need to be answered in this meeting to unblock progress.

        IMPORTANT
        - Do not come up with data on your own
        - If any data is not present like id, owner, email, name etc. Keep it empty

        OUTPUT FORMAT (must be valid JSON; no comments):
        {{
        "title": "string (The exact title of the meeting)",
        "purpose":"string (A single, punchy sentence stating the STRATEGIC GOAL of this specific session. E.g., 'Unblock Auth Service to maintain Demo Timeline.')",
        "expected_outcomes": [{{
            "description": "string (Specific decision, approval, or alignment point required)",
            "owner": "email (Who is responsible for this outcome)",
            "type": "decision|approval|alignment"
        }}],
        "blocking_items": [
            {{
            "title": "string (Critical blocker that this meeting must resolve)",
            "owner": "email (Who owns the blocker)",
            "eta": "YYYY-MM-DD (Estimated resolution date)",
            "impact": "string (Business impact: 'Delays Demo', 'Blocks QA', etc.)",
            "severity": "low|medium|high",
            "status": "open|mitigating|cleared"
            }}
        ],
        "key_points": ["string (Format: '**Topic:** Status/Delta'. E.g., '**Frontend:** 50% Complete (On Track)')"],
        "open_questions": ["string (Strategic question. E.g., 'What is the specific remediation plan for the Auth bug?')"]
        }}

        MEETING INFORMATION:
        {meeting_info}

        {previous_meetings_context}

        Generate the Executive Prep Pack as a single JSON object:"""

        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the prepared prompt."""
        try:
            llm_response = await llm_client.chat.completions.create(
                model="gemini-2.5-pro",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                reasoning_effort="low",
            )

            response = llm_response.choices[0].message.content
            # response = self.llm.complete(prompt)
        except Exception as exc:
            logger.exception("[MeetingPrepAgent] LLM call failed: %s", exc)
            raise MeetingPrepAgentError(f"llm call failed: {exc}") from exc

        text = str(response).strip()
        if not text:
            raise MeetingPrepAgentError("llm returned empty response.")
        return text

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse and validate the LLM response."""
        cleaned = self._strip_code_fence(raw.strip())
        try:
            print('#'*80)
            print(cleaned)
            print('#'*80)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end > start:
                snippet = cleaned[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    pass
            logger.error("[MeetingPrepAgent] Invalid JSON from LLM: %s", raw)
            raise MeetingPrepAgentError("llm response was not valid JSON.")

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
        context: Optional[Dict[str, Any]] = None,
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
