from __future__ import annotations
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class PainPointEnriched(BaseModel):
    """
    The LLM-produced, structured enrichment of a raw user message.
    Keep this small and analytics-friendly.
    """
    # Short, human-readable summary for list views and reports
    title: str = Field(..., description="Short, human-readable summary (<= ~80 chars)")

    # Your taxonomy label for grouping (UX, Pricing, Bug, Support, etc.)
    category: str = Field(..., description="Taxonomy label for grouping")

    # Impact/urgency level. Keep to a closed set for dashboards.
    severity: Literal["low", "medium", "high", "critical"] = Field(
        "low", description="Impact/urgency of the pain point"
    )

    # Optional enrichments to help triage and slice data later
    persona: Optional[str] = Field(None, description="User persona (optional)")
    product_area: Optional[str] = Field(None, description="Impacted product area (optional)")
    tags: List[str] = Field(default_factory=list, description="Free-form labels")
    root_causes: List[str] = Field(default_factory=list, description="Suspected root causes")
    impacted_flows: List[str] = Field(default_factory=list, description="Impacted user flows")
    suggested_actions: List[str] = Field(default_factory=list, description="Proposed next steps")


class PainPointBase(BaseModel):
    """
    Base shape for a pain point. This is the logical record we want to store.
    """
    # Tenancy & identity
    tenant_id: str
    user_id: str

    # Where did this come from (chat/meeting/voice/form/etc.)?
    source: Literal["chat", "meeting", "voice", "form", "other"] = "chat"

    # The raw user message that triggered this pain point
    raw_text: str

    # Optional context we may receive from the producer
    session_id: Optional[str] = None

    # OPTIONAL department in user context (as requested)
    department: Optional[str] = Field(
        None, description="User's department if available"
    )

    # LLM-enriched structured payload
    enriched: PainPointEnriched

    # Arbitrary extra context (locale, channel, appVersion, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PainPointCreate(PainPointBase):
    """
    Shape used by the service layer when creating a record.
    Includes message_id for idempotency (one DB row per message).
    """
    message_id: str = Field(..., description="Event id for idempotent writes")


class PainPoint(PainPointBase):
    """
    Read/back model. Mirrors what you'll have in Mongo (normalized).
    `_id` is exposed as `id` via alias for convenience in Python.
    """
    id: str | None = Field(None, alias="_id")
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)
