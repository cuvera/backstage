from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from app.core.openai_client import llm_client # Assuming this is correctly imported

from pydantic import ValidationError

from app.schemas.meeting_analysis import (
    ActionItem,
    CallScoring,
    Decision,
    MeetingAnalysis,
    MeetingTranscript,
    ScoringReason,
    SentimentLabel,
    SentimentOverview,
    TalkTimeStat,
    TimeReference,
)

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
        self.llm = llm or llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def analyze(
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
        
        # FIX 1: Process sentiment data from the transcript payload
        sentiment = self._process_sentiment(transcript_payload)
        
        prompt = self._build_prompt(transcript, talk_time_stats)
        raw = await self._call_llm(prompt)
        payload = self._parse_response(raw)

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
        """Transform input payload to MeetingTranscript.
        
        Handles new 'transcriptions' format with MM:SS times.
        """
        candidate = dict(payload or {})
        candidate["tenant_id"] = tenant_id
        candidate["session_id"] = session_id
        
        # Transform new 'transcriptions' format to 'conversation'
        if "transcriptions" in candidate and "conversation" not in candidate:
            transcriptions = candidate.get("transcriptions", [])
            conversation = []
            for t in transcriptions:
                if len(str(t.get("transcription", ""))) > 0:
                    start_mmss = t.get("start", "00:00")
                    end_mmss = t.get("end", "00:00")
                    # Convert MM:SS to HH:MM:SS
                    start_hhmmss = self._mmss_to_hhmmss(start_mmss)
                    end_hhmmss = self._mmss_to_hhmmss(end_mmss)
                    conversation.append({
                        "start_time": start_hhmmss,
                        "end_time": end_hhmmss,
                        "speaker": t.get("speaker", "Unknown"),
                        "text": t.get("transcription", ""),
                    })
            candidate["conversation"] = conversation
        
        if not candidate.get("conversation"):
            raise CallAnalysisAgentError("conversation must contain at least one turn.")

        try:
            return MeetingTranscript(**candidate)
        except ValidationError as exc:
            logger.exception("[CallAnalysisAgent] Transcript validation failed: %s", exc)
            raise CallAnalysisAgentError(f"transcript validation failed: {exc}") from exc
    
    def _mmss_to_hhmmss(self, mmss: str) -> str:
        """Convert MM:SS format to HH:MM:SS format.
        
        Handles cases like '65:32' → '01:05:32'
        """
        try:
            parts = mmss.strip().split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                hours = minutes // 60
                remaining_minutes = minutes % 60
                return f"{hours:02d}:{remaining_minutes:02d}:{seconds:02d}"
            elif len(parts) == 3:
                # Already in HH:MM:SS format
                return mmss
            else:
                return "00:00:00"
        except (ValueError, IndexError):
            return "00:00:00"
    
    def _hhmmss_to_seconds(self, hhmmss: str) -> float:
        """Convert HH:MM:SS format to seconds."""
        try:
            parts = hhmmss.strip().split(":")
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = int(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
            else:
                return 0.0
        except (ValueError, IndexError):
            return 0.0
    
    def _seconds_to_hhmmss(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        s = int(seconds)
        hours = s // 3600
        minutes = (s % 3600) // 60
        sec = s % 60
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"

    def _compute_talk_time(self, transcript: MeetingTranscript) -> List[TalkTimeStat]:
        """Compute talk time statistics for each speaker.
        
        Converts HH:MM:SS times to seconds for calculation, then formats output as HH:MM:SS.
        """
        totals: Dict[str, float] = defaultdict(float)
        turns: Dict[str, int] = defaultdict(int)

        for turn in transcript.conversation:
            # Parse start and end times to calculate duration
            start_seconds = self._hhmmss_to_seconds(turn.start_time)
            end_seconds = self._hhmmss_to_seconds(turn.end_time)
            duration = max(0.0, end_seconds - start_seconds)
            totals[turn.speaker] += duration
            turns[turn.speaker] += 1

        total_duration = sum(totals.values()) or 1.0
        stats: List[TalkTimeStat] = []
        for speaker, seconds in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
            share = round((seconds / total_duration) * 100.0, 2)
            stats.append(
                TalkTimeStat(
                    speaker=speaker,
                    total_duration=self._seconds_to_hhmmss(seconds),
                    share_percent=share,
                    turns=turns[speaker],
                )
            )
        return stats

    # ------------------------------------------------------------------
    # Sentiment processing
    # ------------------------------------------------------------------
    def _process_sentiment(self, payload: Dict[str, Any]) -> SentimentOverview:
        """Processes sentiment data from the raw transcript payload."""
        sentiments = payload.get("sentiments", {})
        
        # 1. Get overall sentiment
        overall = SentimentLabel.neutral
        try:
            overall = SentimentLabel(sentiments.get("overall", "neutral"))
        except ValueError:
            logger.warning(f"[CallAnalysisAgent] Invalid overall sentiment label: {sentiments.get('overall')}")
        
        # 2. Get per-speaker sentiment from multiple possible sources
        per_speaker: Dict[str, SentimentLabel] = {}
        
        # First, try the 'participant' list from sentiments
        participants = sentiments.get("participant", [])
        logger.info(f"[CallAnalysisAgent] Found {len(participants)} participants in sentiments.participant")
        for p in participants:
            name = (p.get("name") or "").strip()
            sentiment_str = (p.get("sentiment") or "neutral").strip()
            if name:
                try:
                    per_speaker[name] = SentimentLabel(sentiment_str)
                    logger.info(f"[CallAnalysisAgent] Added sentiment from participant: {name} = {sentiment_str}")
                except ValueError:
                    logger.warning(f"[CallAnalysisAgent] Invalid sentiment label for {name}: {sentiment_str}")
                    per_speaker[name] = SentimentLabel.neutral
        
        # Second, try the 'speakers' array from the payload root (new format)
        speakers = payload.get("speakers", [])
        logger.info(f"[CallAnalysisAgent] Found {len(speakers)} speakers in payload.speakers")
        for speaker_data in speakers:
            name = (speaker_data.get("speaker") or "").strip()
            sentiment_str = (speaker_data.get("sentiment") or "neutral").strip()
            if name:
                try:
                    per_speaker[name] = SentimentLabel(sentiment_str)
                    logger.info(f"[CallAnalysisAgent] Added sentiment from speaker: {name} = {sentiment_str}")
                except ValueError:
                    logger.warning(f"[CallAnalysisAgent] Invalid sentiment label for {name}: {sentiment_str}")
                    per_speaker[name] = SentimentLabel.neutral

        logger.info(f"[CallAnalysisAgent] Final per_speaker sentiment count: {len(per_speaker)}")
        return SentimentOverview(overall=overall, per_speaker=per_speaker)


    # ------------------------------------------------------------------
    # Prompt / LLM
    # ------------------------------------------------------------------
    def _build_prompt(self, transcript: MeetingTranscript, stats: List[TalkTimeStat]) -> str:
        participants = ", ".join(p.name for p in transcript.participants) or "Unknown participants"
        talk_time_lines = "\n".join(
            f"{s.speaker}: {s.total_duration} ({s.share_percent}%), turns={s.turns}"
            for s in stats
        )

        header = (
            "You are a senior meeting analyst. Read the full transcript and produce a single JSON object.\n"
            "STRICT OUTPUT CONTRACT (one object, no comments, no markdown):\n"
            "{\n"
            '  "summary": string,\n'
            '  "key_points": [string],\n'
            '  "decisions": [{"title": string, "owner": string|null, "due_date": string|null, "references": [{"start": "HH:MM:SS", "end": "HH:MM:SS"}]}],\n'
            '  "action_items": [{"task": string, "owner": string|null, "due_date": string|null, "priority": string|null, "references": [{"start": "HH:MM:SS", "end": "HH:MM:SS"}]}],\n'
            '  "call_scoring": {\n'
            '    "score": number (0-10),\n'
            '    "grade": string (A, B, C, D, F),\n'
            '    "reasons": [{"reason": string, "reference": {"start": "HH:MM:SS", "end": "HH:MM:SS"}}],\n'
            '    "summary": string\n'
            '  }\n'
            "}\n\n"
            "QUALITY RULES AND FIELD DEFINITIONS:\n"
            "1. **MUTUALLY EXCLUSIVE:** An item CANNOT be both a Decision and an Action Item. You must choose the single best category.\n"
            "2. **DECISION:** A final **outcome**, **approval**, **agreement**, or **policy change** made during the meeting. It is a STATE CHANGE. (e.g., 'Budget Approved', 'Hiring Freeze Lifted', 'Vendor X Selected').\n"
            "2. **ACTION ITEM:** A specific **task**, **to-do**, or **deliverable** assigned for FUTURE execution. Any task or actionable follow-up that mentions an owner name or a time/due-date must be captured as an action item.\n"
            "3. **Summary:** A comprehensive, high-level overview (5-8 crisp sentences) covering the main **purpose/goals**, the final **outcomes**, and any major **follow-up** steps.\n"
            "4. **Key Points:** Specific, factual, bullet-style highlights (strings) of significant **progress**, **blockers**, **metrics**, or **updates** discussed.\n"
            "5. **REFERENCES:** All decisions and action_items MUST include time-based references in HH:MM:SS format (e.g., {\"start\": \"00:05:32\", \"end\": \"00:06:15\"}). Use timestamps from the transcript directly.\n"
            "6. Use null (not empty string) for unknown owner/due_date/priority fields.\n"
            "7. Use [] for empty arrays; never use null or omit required keys.\n"
            "8. Output MUST be valid JSON — no code fences, trailing commas, extra keys, or explanatory text.\n\n"
            
            "CALL SCORING (PRODUCTION-GRADE CALCULATION):\n"
            "The final `score` (0-10) and `grade` must be rigorously calculated based on the completeness and quality of the extracted Decisions and Action Items.\n\n"
            
            "STEP 1: Calculate the 5 component scores (0-10) by assessing all *extracted* decisions and action items against these criteria:\n"
            "  - **Action Item Completeness (30% weight):** Are action items clear, specific, and detailed?\n"
            "  - **Owner Assignment Clarity (20% weight):** Are clear owners assigned to all major decisions and actions?\n"
            "  - **Due Date Quality (20% weight):** Are specific, realistic due dates present for actions/decisions?\n"
            "  - **Meeting Structure & Decision Flow (20% weight):** Is the meeting flow productive, leading to clear decisions and not circular discussion?\n"
            "  - **Signal vs Noise Ratio (10% weight):** Was the discussion focused and efficient?\n\n"
            
            "STEP 2: Use the following weighted average formula to determine the final `score`:\n"
            "score = (0.30 * Action Completeness) + (0.20 * Owner Clarity) + \n"
            "      (0.20 * Due Date Quality) + (0.20 * Structure) + \n"
            "      (0.10 * Signal/Noise)\n\n"
            
            "STEP 3: Map the final score to the appropriate, **simplified** `grade`:\n"
            "A: 8.5-10.0 (Excellent),\n"
            "B: 7.0-8.4 (Good),\n"
            "C: 5.5-6.9 (Acceptable),\n"
            "D: 4.0-5.4 (Poor),\n"
            "F: 0.0-3.9 (Unacceptable)\n\n"
            
            "STEP 4: Generate 3-5 high-impact `reasons` supporting the score. **CRITICAL:** Each reason must be a **professional, insight-driven summary** (2-4 words).\n"
            "  - **POSITIVE EXAMPLES:** 'Clear Ownership Assigned', 'Detailed Action Plan', 'Productive Decision Flow', 'Strong Focus'.\n"
            "  - **NEGATIVE EXAMPLES:** 'Vague Timelines', 'Undefined Responsibilities', 'Circular Discussion', 'Missing Follow-up'.\n"
            "  - **AVOID:** Generic phrases like 'Good Structure', 'Bad Score', 'Nice Meeting'.\n"
        )

        metadata = (
            f"Tenant: {transcript.tenant_id}\n"
            f"Session: {transcript.session_id}\n"
            f"Participants: {participants}\n"
            f"Talk time:\n{talk_time_lines or 'No speech recorded.'}\n"
        )

        convo_lines: List[str] = []
        for idx, turn in enumerate(transcript.conversation[: self.MAX_TURNS_IN_PROMPT]):
            start_ts = self._format_timestamp(turn.start_time)
            end_ts = self._format_timestamp(turn.end_time)
            convo_lines.append(f"[{idx}] [{start_ts} - {end_ts}] {turn.speaker}: {turn.text}")

        transcript_block = "\n".join(convo_lines)
        return f"{header}\n{metadata}\nTranscript:\n{transcript_block}"

    async def _call_llm(self, prompt: str) -> str:
        logger.info("[CallAnalysisAgent] Starting LLM call")
        llm_start_time = time.time()
        
        try:
            response = await self.llm.chat.completions.create(
                model="gemini-2.5-pro",
                reasoning_effort="low",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            text = response.choices[0].message.content
            
            llm_duration_ms = round((time.time() - llm_start_time) * 1000, 2)
            logger.info(f"[CallAnalysisAgent] LLM call completed in {llm_duration_ms}ms")
            
        except Exception as exc:
            llm_duration_ms = round((time.time() - llm_start_time) * 1000, 2)
            logger.error(f"[CallAnalysisAgent] LLM call failed after {llm_duration_ms}ms: {exc}")
            raise CallAnalysisAgentError(f"llm call failed: {exc}") from exc

        if not text:
            raise CallAnalysisAgentError("llm returned empty response.")
        return text.strip()

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
        if text.startswith("\`\`\`") and text.endswith("\`\`\`"):
            inner = text.strip("`")
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
        
        call_scoring = self._build_call_scoring(payload.get("call_scoring"))
        
        # Calculate duration from transcript conversation
        duration_hhmmss = "00:00:00"
        if transcript.conversation:
            first_turn = transcript.conversation[0]
            last_turn = transcript.conversation[-1]
            start_sec = self._hhmmss_to_seconds(first_turn.start_time)
            end_sec = self._hhmmss_to_seconds(last_turn.end_time)
            duration_sec = max(0.0, end_sec - start_sec)
            duration_hhmmss = self._seconds_to_hhmmss(duration_sec)
        
        analysis = MeetingAnalysis(
            tenant_id=transcript.tenant_id,
            session_id=transcript.session_id,
            summary=summary,
            key_points=self._ensure_str_list(payload.get("key_points")),
            decisions=self._build_decisions(payload.get("decisions")),
            action_items=self._build_action_items(payload.get("action_items")),
            sentiment_overview=sentiment,
            talk_time_stats=talk_time_stats,
            call_scoring=call_scoring,
            created_at=datetime.utcnow().isoformat(),
            transcript_language=transcript.language,
            duration=duration_hhmmss,
            metadata=self._build_metadata(payload.get("metadata"), context),
        )
        return analysis

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
        """Build decisions list from LLM response with TimeReference objects."""
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
                references=self._build_time_references(entry.get("references")),
            )
            decisions.append(decision)
        return decisions

    def _build_action_items(self, value: Any) -> List[ActionItem]:
        """Build action items list from LLM response with TimeReference objects."""
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
                references=self._build_time_references(entry.get("references")),
            )
            items.append(item)
        return items
    
    def _build_time_references(self, value: Any) -> List[TimeReference]:
        """Build list of TimeReference objects from LLM response."""
        refs: List[TimeReference] = []
        if not isinstance(value, Iterable):
            return refs
        for entry in value:
            if not isinstance(entry, dict):
                continue
            start = (entry.get("start") or "00:00:00").strip()
            end = (entry.get("end") or "00:00:00").strip()
            refs.append(TimeReference(start=start, end=end))
        return refs

    def _clean_optional(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return str(value)

    def _calculate_grade(self, score: float) -> str:
        """Map score to the simplified grade based on prompt definition."""
        if score >= 8.5:
            return "A"
        elif score >= 7.0:
            return "B"
        elif score >= 5.5:
            return "C"
        elif score >= 4.0:
            return "D"
        else:
            return "F"
    
    def _build_scoring_reasons(self, value: Any) -> List[ScoringReason]:
        """Build scoring reasons list, ensuring reference times are extracted."""
        reasons: List[ScoringReason] = []
        if not isinstance(value, Iterable):
            return reasons
        
        for entry in value:
            if not isinstance(entry, dict):
                continue
            reason_text = (entry.get("reason") or "").strip()
            if not reason_text:
                continue
            
            reference = entry.get("reference", {})
            start = self._clean_optional(reference.get("start"))
            end = self._clean_optional(reference.get("end"))
            
            reasons.append(
                ScoringReason(
                    reason=reason_text,
                    reference={
                        "start": start,
                        "end": end,
                    }
                )
            )
        return reasons
    
    def _build_call_scoring(self, value: Any) -> Optional[CallScoring]:
        """
        Build call scoring object from LLM response.
        Uses the LLM's final 'score' and calculates the individual component
        scores as proxies to satisfy the Pydantic CallScoring model.
        """
        if not isinstance(value, dict):
            return None
        
        try:
            raw_score = float(value.get("score", 0.0))
            
            # Use raw_score as a proxy for the individual component scores
            proxy_score = raw_score if 0.0 <= raw_score <= 10.0 else 5.0
            
            action_item_score = proxy_score
            owner_clarity_score = proxy_score
            due_date_score = proxy_score
            structure_score = proxy_score
            signal_noise_score = proxy_score
            
            final_score = proxy_score

            grade = (value.get("grade") or "F").strip()
            summary = (value.get("summary") or "").strip()
            reasons = self._build_scoring_reasons(value.get("reasons", []))
            
            # Recalculate grade based on the simplified scale
            if not grade:
                grade = self._calculate_grade(final_score)
            
            return CallScoring(
                score=round(final_score, 2),
                grade=grade,
                reasons=reasons,
                summary=summary or "Meeting quality assessment completed.",
                action_item_completeness_score=round(action_item_score, 2),
                owner_clarity_score=round(owner_clarity_score, 2),
                due_date_quality_score=round(due_date_score, 2),
                meeting_structure_score=round(structure_score, 2),
                signal_noise_ratio_score=round(signal_noise_score, 2)
            )
        except (TypeError, ValueError, ValidationError) as e:
            logger.warning("[CallAnalysisAgent] Failed to build CallScoring: %s", e)
            return None

    def _format_timestamp(self, time_str: str) -> str:
        """Format timestamp - returns as-is if already in HH:MM:SS format."""
        # Time is already in HH:MM:SS format from input transformation
        return time_str
