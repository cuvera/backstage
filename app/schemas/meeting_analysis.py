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
    risks_issues: List[RiskIssue] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    sentiment_overview: SentimentOverview = Field(default_factory=SentimentOverview)
    talk_time_stats: List[TalkTimeStat] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.medium
    created_at: Optional[str] = Field(
        None, description="ISO timestamp when analysis was created."
    )
    transcript_language: Optional[str] = None
    duration_sec: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
