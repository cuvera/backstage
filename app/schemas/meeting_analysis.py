from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class SentimentLabel(str, Enum):
    negative = "negative"
    neutral = "neutral"
    positive = "positive"
    mixed = "mixed"


class ConfidenceLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class OutcomeType(str, Enum):
    decision = "decision"
    approval = "approval"
    alignment = "alignment"


class SeverityLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class BlockingItemStatus(str, Enum):
    open = "open"
    mitigating = "mitigating"
    cleared = "cleared"


class ExpectedOutcome(BaseModel):
    description: str = Field(..., min_length=1)
    owner: str = Field(default="")
    type: OutcomeType


class BlockingItem(BaseModel):
    title: str = Field(..., min_length=1)
    owner: str = Field(default="")
    eta: str = Field(default="")
    impact: str = Field(default="")
    severity: SeverityLevel = SeverityLevel.medium
    status: BlockingItemStatus = BlockingItemStatus.open


class DecisionQueueItem(BaseModel):
    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    needs: List[str] = Field(default_factory=list)
    owner: str = Field(default="")


class PreviousMeetingReference(BaseModel):
    meeting_id: str = Field(..., min_length=1)
    analysis_id: str = Field(..., min_length=1)
    datetime: str = Field(..., min_length=1)


class Turn(BaseModel):
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., gt=0.0)
    speaker: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    duration: Optional[float] = Field(
        None,
        gt=0.0,
        description="Duration in seconds. Auto-derived when omitted.",
    )

    @validator("end_time")
    def _end_after_start(cls, v, values):
        start = values.get("start_time", 0.0)
        if v <= start:
            raise ValueError("end_time must be greater than start_time")
        return v

    @validator("duration", always=True)
    def _ensure_duration(cls, v, values):
        start = float(values.get("start_time", 0.0))
        end = float(values.get("end_time", 0.0))
        duration = float(v) if v is not None else max(0.0, end - start)
        if duration <= 0.0:
            raise ValueError("duration must be positive")
        return round(duration, 2)


class Participant(BaseModel):
    name: str = Field(..., min_length=1)
    user_id: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None


class MeetingTranscript(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    participants: List[Participant] = Field(default_factory=list)
    conversation: List[Turn] = Field(default_factory=list)
    started_at_sec: Optional[float] = Field(None, ge=0.0)
    ended_at_sec: Optional[float] = Field(None, ge=0.0)
    duration_sec: Optional[float] = Field(None, ge=0.0)
    language: Optional[str] = Field(None, description="BCP-47 code, e.g. 'en-US'.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("conversation")
    def _ensure_sorted(cls, turns: List[Turn]):
        return sorted(turns, key=lambda t: (t.start_time, t.end_time))

    @validator("duration_sec", always=True)
    def _derive_duration(cls, value, values):
        if value is not None:
            return value
        conversation: List[Turn] = values.get("conversation", [])
        if not conversation:
            return 0.0
        start = conversation[0].start_time
        end = conversation[-1].end_time
        return round(max(0.0, end - start), 2)


class SentimentOverview(BaseModel):
    overall: SentimentLabel = SentimentLabel.neutral
    per_speaker: Dict[str, SentimentLabel] = Field(default_factory=dict)


class TalkTimeStat(BaseModel):
    speaker: str
    total_seconds: float = Field(..., ge=0.0)
    share_percent: float = Field(..., ge=0.0, le=100.0)
    turns: int = Field(..., ge=0)


class Decision(BaseModel):
    title: str = Field(..., min_length=1)
    owner: Optional[str] = None
    due_date: Optional[str] = Field(
        None, description="ISO date if known (YYYY-MM-DD)."
    )
    references: List[int] = Field(default_factory=list)


class ActionItem(BaseModel):
    task: str = Field(..., min_length=1)
    owner: Optional[str] = None
    due_date: Optional[str] = None
    priority: Optional[str] = None
    references: List[int] = Field(default_factory=list)


class RiskIssue(BaseModel):
    description: str = Field(..., min_length=1)
    mitigation: Optional[str] = None
    severity: Optional[str] = None
    references: List[int] = Field(default_factory=list)


class Deviation(BaseModel):
    """Represents a deviation from meeting standards or expectations."""
    type: str = Field(..., min_length=1, description="Type of deviation (e.g., 'missing_owner', 'vague_action', 'unclear_decision')")
    description: str = Field(..., min_length=1, description="Clear description of the deviation")
    reference: Dict[str, str] = Field(..., description="Timestamp reference with 'start' and 'end' keys (e.g., {'start': '00:00:05', 'end': '00:00:25'})")


class ScoringReason(BaseModel):
    """Represents a reason contributing to the meeting score."""
    reason: str = Field(..., min_length=1, description="Brief explanation of scoring factor")
    reference: Dict[str, str] = Field(..., description="Timestamp reference with 'start' and 'end' keys")


class CallScoring(BaseModel):
    """Overall meeting quality score with detailed breakdown."""
    score: float = Field(..., ge=0.0, le=10.0, description="Final meeting score from 0-10")
    grade: str = Field(..., description="Letter grade: A, B, C, D, F")
    reasons: List[ScoringReason] = Field(default_factory=list, description="List of reasons supporting the score")
    summary: str = Field(..., min_length=1, description="Brief summary of overall meeting quality")
    
    action_item_completeness_score: float = Field(..., ge=0.0, le=10.0, description="Score for action item quality (0-10)")
    owner_clarity_score: float = Field(..., ge=0.0, le=10.0, description="Score for owner assignment clarity (0-10)")
    due_date_quality_score: float = Field(..., ge=0.0, le=10.0, description="Score for due date quality (0-10)")
    meeting_structure_score: float = Field(..., ge=0.0, le=10.0, description="Score for meeting structure & decision flow (0-10)")
    signal_noise_ratio_score: float = Field(..., ge=0.0, le=10.0, description="Score for signal vs noise ratio (0-10)")


class TimelineHighlight(BaseModel):
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., gt=0.0)
    label: str = Field(..., min_length=1)


class MeetingAnalysis(BaseModel):
    tenant_id: str
    session_id: str
    summary: str = Field(..., min_length=1)
    key_points: List[str] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    sentiment_overview: SentimentOverview = Field(default_factory=SentimentOverview)
    talk_time_stats: List[TalkTimeStat] = Field(default_factory=list)
    call_scoring: Optional[CallScoring] = Field(None, description="Overall meeting quality score")
    
    created_at: Optional[str] = Field(
        None, description="ISO timestamp when analysis was created."
    )
    transcript_language: Optional[str] = None
    duration_sec: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MeetingPrepPack(BaseModel):
    title: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    timezone: str = Field(default="")
    locale: str = Field(default="en-US")
    recurring_meeting_id: str = Field(..., min_length=1)
    purpose: str = Field(default="")
    confidence: ConfidenceLevel = ConfidenceLevel.medium
    expected_outcomes: List[ExpectedOutcome] = Field(default_factory=list)
    blocking_items: List[BlockingItem] = Field(default_factory=list)
    decision_queue: List[DecisionQueueItem] = Field(default_factory=list)
    key_points: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    risks_issues: List[str] = Field(default_factory=list)
    leadership_asks: List[str] = Field(default_factory=list)
    previous_meetings_ref: List[PreviousMeetingReference] = Field(default_factory=list)
    created_at: str = Field(..., min_length=1)
    updated_at: str = Field(..., min_length=1)