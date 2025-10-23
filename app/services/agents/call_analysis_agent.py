from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from pydantic import ValidationError

from app.schemas.meeting_analysis import (
    ActionItem,
    ConfidenceLevel,
    Decision,
    MeetingAnalysis,
    MeetingTranscript,
    RiskIssue,
    SentimentLabel,
    SentimentOverview,
    TalkTimeStat,
)
from app.services.llm.factory import get_llm

logger = logging.getLogger(__name__)


class CallAnalysisAgentError(Exception):
    """Raised when the call analysis agent cannot complete its task."""


class CallAnalysisAgent:
    """
    Minimal call analysis agent inspired by PainPointAgent:
      - accepts a raw transcript payload and context
      - builds a compact prompt with light-weight analytics
      - requests STRICT JSON from the LLM
      - parses/validates into MeetingAnalysis
    """

    MAX_TURNS_IN_PROMPT = 500

    def __init__(self, llm=None) -> None:
        self.llm = llm or get_llm()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        transcript_payload: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> MeetingAnalysis:
        ctx = context or {}
        
        tenant_id = (ctx.get("tenant_id") or transcript_payload.get("tenant_id") or "").strip()
        session_id = (ctx.get("session_id") or transcript_payload.get("session_id") or "").strip()
        if not tenant_id or not session_id:
            raise CallAnalysisAgentError("tenant_id and session_id are required.")

        transcript = self._to_transcript(transcript_payload, tenant_id, session_id)
        talk_time_stats = self._compute_talk_time(transcript)
        prompt = self._build_prompt(transcript, talk_time_stats)
        raw = self._call_llm(prompt)
        payload = self._parse_response(raw)

        sentiment = self._neutral_sentiment(transcript)
        analysis = self._build_analysis(
            payload=payload,
            transcript=transcript,
            talk_time_stats=talk_time_stats,
            sentiment=sentiment,
            context=ctx,
        )
        return analysis

    # ------------------------------------------------------------------
    # Transcript helpers
    # ------------------------------------------------------------------
    def _to_transcript(
        self,
        payload: Dict[str, Any],
        tenant_id: str,
        session_id: str,
    ) -> MeetingTranscript:
        candidate = dict(payload or {})
        candidate["tenant_id"] = tenant_id
        candidate["session_id"] = session_id
        if not candidate.get("conversation"):
            raise CallAnalysisAgentError("conversation must contain at least one turn.")

        try:
            return MeetingTranscript(**candidate)
        except ValidationError as exc:
            logger.exception("[CallAnalysisAgent] Transcript validation failed: %s", exc)
            raise CallAnalysisAgentError(f"transcript validation failed: {exc}") from exc

    def _compute_talk_time(self, transcript: MeetingTranscript) -> List[TalkTimeStat]:
        totals: Dict[str, float] = defaultdict(float)
        turns: Dict[str, int] = defaultdict(int)

        for turn in transcript.conversation:
            totals[turn.speaker] += turn.duration
            turns[turn.speaker] += 1

        total_duration = sum(totals.values()) or 1.0
        stats: List[TalkTimeStat] = []
        for speaker, seconds in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
            share = round((seconds / total_duration) * 100.0, 2)
            stats.append(
                TalkTimeStat(
                    speaker=speaker,
                    total_seconds=round(seconds, 2),
                    share_percent=share,
                    turns=turns[speaker],
                )
            )
        return stats

    # ------------------------------------------------------------------
    # Prompt / LLM
    # ------------------------------------------------------------------
    def _build_prompt(self, transcript: MeetingTranscript, stats: List[TalkTimeStat]) -> str:
        participants = ", ".join(p.name for p in transcript.participants) or "Unknown participants"
        talk_time_lines = "\n".join(
            f"{s.speaker}: {s.total_seconds}s ({s.share_percent}%), turns={s.turns}"
            for s in stats
        )

        header = (
            "You are a senior meeting analyst. Read the full transcript and produce a single JSON object.\n"
            "STRICT OUTPUT CONTRACT (one object, no comments, no markdown):\n"
            "{\n"
            '  "summary": string,\n'
            '  "key_points": [string],\n'
            '  "decisions": [{"title": string, "owner": string|null, "due_date": string|null, "references": [int]}],\n'
            '  "action_items": [{"task": string, "owner": string|null, "due_date": string|null, "priority": string|null, "references": [int]}],\n'
            '  "risks_issues": [{"description": string, "mitigation": string|null, "severity": string|null, "references": [int]}],\n'
            '  "open_questions": [string],\n'
            '  "topics": [string],\n'
            '  "confidence": "low" | "medium" | "high"\n'
            "}\n"
            "QUALITY RULES:\n"
            "1. Ground every statement in the transcript; never invent people, dates, or workstreams.\n"
            "2. Keep summary to 5-8 crisp sentences that cover goals, outcomes, follow-ups, and risks.\n"
            "3. Key points are bullet-style highlights of material progress, blockers, or metrics.\n"
            "4. Decisions/action_items/risks must include references (0-based turn indices) whenever evidence exists; otherwise use an empty list.\n"
            "5. Use null (not empty string) for unknown owner/due_date/priority fields.\n"
            "6. Use [] for empty arrays; never use null or omit required keys.\n"
            "7. Confidence guidance: high when transcript is explicit and consistent, medium for partial clarity, low when outcomes are uncertain or contradictory.\n"
            "8. Output MUST be valid JSON — no code fences, trailing commas, extra keys, or explanatory text.\n"
        )

        metadata = (
            f"Tenant: {transcript.tenant_id}\n"
            f"Session: {transcript.session_id}\n"
            f"Participants: {participants}\n"
            f"Talk time:\n{talk_time_lines or 'No speech recorded.'}\n"
        )

        convo_lines: List[str] = []
        for idx, turn in enumerate(transcript.conversation[: self.MAX_TURNS_IN_PROMPT]):
            convo_lines.append(f"[{idx}] {turn.speaker}: {turn.text}")

        transcript_block = "\n".join(convo_lines)
        return f"{header}\n{metadata}\nTranscript:\n{transcript_block}"

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.llm.complete(prompt)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.exception("[CallAnalysisAgent] LLM call failed: %s", exc)
            raise CallAnalysisAgentError(f"llm call failed: {exc}") from exc

        text = str(response).strip()
        if not text:
            raise CallAnalysisAgentError("llm returned empty response.")
        return text

    # ------------------------------------------------------------------
    # Parsing / normalisation
    # ------------------------------------------------------------------
    def _parse_response(self, raw: str) -> Dict[str, Any]:
        cleaned = self._strip_code_fence(raw.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end > start:
                snippet = cleaned[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    pass
            logger.error("[CallAnalysisAgent] Invalid JSON from LLM: %s", raw)
            raise CallAnalysisAgentError("llm response was not valid JSON.")

    def _strip_code_fence(self, text: str) -> str:
        if text.startswith("```") and text.endswith("```"):
            inner = text.strip("`")
            # Handles ```json ... ``` by splitting once.
            parts = inner.split("\n", 1)
            return parts[1] if len(parts) == 2 else parts[0]
        return text

    def _build_analysis(
        self,
        *,
        payload: Dict[str, Any],
        transcript: MeetingTranscript,
        talk_time_stats: List[TalkTimeStat],
        sentiment: SentimentOverview,
        context: Dict[str, Any],
    ) -> MeetingAnalysis:
        summary = (payload.get("summary") or "").strip() or "Summary not available."
        analysis = MeetingAnalysis(
            tenant_id=transcript.tenant_id,
            session_id=transcript.session_id,
            summary=summary,
            key_points=self._ensure_str_list(payload.get("key_points")),
            decisions=self._build_decisions(payload.get("decisions")),
            action_items=self._build_action_items(payload.get("action_items")),
            risks_issues=self._build_risks(payload.get("risks_issues")),
            open_questions=self._ensure_str_list(payload.get("open_questions")),
            sentiment_overview=sentiment,
            talk_time_stats=talk_time_stats,
            topics=self._ensure_str_list(payload.get("topics")),
            confidence=self._safe_confidence(payload.get("confidence")),
            created_at=datetime.utcnow().isoformat(),
            transcript_language=transcript.language,
            duration_sec=transcript.duration_sec,
            metadata=self._build_metadata(payload.get("metadata"), context),
        )
        return analysis

    def _neutral_sentiment(self, transcript: MeetingTranscript) -> SentimentOverview:
        return SentimentOverview(
            overall=SentimentLabel.neutral,
            per_speaker={turn.speaker: SentimentLabel.neutral for turn in transcript.conversation},
        )

    def _build_metadata(
        self,
        payload_meta: Any,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if isinstance(payload_meta, dict):
            metadata.update(payload_meta)
        if context.get("user_id"):
            metadata.setdefault("requested_by", context["user_id"])
        if context.get("source"):
            metadata.setdefault("source", context["source"])
        return metadata

    def _ensure_str_list(self, value: Any) -> List[str]:
        items: List[str] = []
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                items.append(candidate)
        elif isinstance(value, Iterable):
            for entry in value:
                if isinstance(entry, str):
                    candidate = entry.strip()
                    if candidate:
                        items.append(candidate)
        return items

    def _ensure_int_list(self, value: Any) -> List[int]:
        out: List[int] = []
        if isinstance(value, Iterable):
            for item in value:
                try:
                    out.append(int(item))
                except (TypeError, ValueError):
                    continue
        return out

    def _build_decisions(self, value: Any) -> List[Decision]:
        decisions: List[Decision] = []
        if not isinstance(value, Iterable):
            return decisions
        for entry in value:
            if not isinstance(entry, dict):
                continue
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            decision = Decision(
                title=title,
                owner=self._clean_optional(entry.get("owner")),
                due_date=self._clean_optional(entry.get("due_date")),
                references=self._ensure_int_list(entry.get("references")),
            )
            decisions.append(decision)
        return decisions

    def _build_action_items(self, value: Any) -> List[ActionItem]:
        items: List[ActionItem] = []
        if not isinstance(value, Iterable):
            return items
        for entry in value:
            if not isinstance(entry, dict):
                continue
            task = (entry.get("task") or "").strip()
            if not task:
                continue
            item = ActionItem(
                task=task,
                owner=self._clean_optional(entry.get("owner")),
                due_date=self._clean_optional(entry.get("due_date")),
                priority=self._clean_optional(entry.get("priority")),
                references=self._ensure_int_list(entry.get("references")),
            )
            items.append(item)
        return items

    def _build_risks(self, value: Any) -> List[RiskIssue]:
        risks: List[RiskIssue] = []
        if not isinstance(value, Iterable):
            return risks
        for entry in value:
            if not isinstance(entry, dict):
                continue
            description = (entry.get("description") or "").strip()
            if not description:
                continue
            risk = RiskIssue(
                description=description,
                mitigation=self._clean_optional(entry.get("mitigation")),
                severity=self._clean_optional(entry.get("severity")),
                references=self._ensure_int_list(entry.get("references")),
            )
            risks.append(risk)
        return risks

    def _safe_confidence(self, value: Any) -> ConfidenceLevel:
        if isinstance(value, ConfidenceLevel):
            return value
        if isinstance(value, str):
            norm = value.strip().lower()
            if norm in ConfidenceLevel.__members__:
                return ConfidenceLevel[norm]  # type: ignore[index]
            try:
                return ConfidenceLevel(norm)
            except ValueError:
                pass
        return ConfidenceLevel.medium

    def _clean_optional(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return str(value)
