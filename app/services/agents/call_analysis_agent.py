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



CALL_ANALYSIS_PROMPT_TEMPLATE = """You are a senior meeting analyst. Read the full transcript and produce a single JSON object.
STRICT OUTPUT CONTRACT (one object, no comments, no markdown):
{
  "summary": string,
  "key_points": [string],
  "decisions": [{"title": string, "owner": string|null, "due_date": string|null, "references": [{"start": "HH:MM:SS", "end": "HH:MM:SS"}]}],
  "action_items": [{"task": string, "owner": string|null, "due_date": string|null, "priority": string|null, "references": [{"start": "HH:MM:SS", "end": "HH:MM:SS"}]}],
  "call_scoring": {
    "identified_agenda": string,
    "agenda_deviation": number (0-10),
    "action_completeness": number (0-10),
    "owner_clarity": number (0-10),
    "due_date_quality": number (0-10),
    "structure": number (0-10),
    "snr": number (0-10),
    "time_management_score": number (0-10),
    "positive_aspects": [string],
    "areas_for_improvement": [string]
  }
}

QUALITY RULES AND FIELD DEFINITIONS (PRODUCTION-GRADE FRAMEWORK):

1. **THE DECISION GATE (Strict Strategic Separation):**
   - **Definition:** A decision is a **STRATEGIC STATE CHANGE** or **FINAL APPROVAL** that affects the project/team direction.
   - **Detail Rule:** Titles must be **comprehensive (15-25 words)** explaining the *Decision*, the *Reasoning*, and the *Implication*.
   - **Bad:** "Budget approved."
   - **Good:** "Budget approved for Q4 Marketing push to aggressively target enterprise leads, allocated from the R&D surplus."
   - **EXCLUSION RULES:** 
     - Agreeing to "schedule a meeting" is **NEVER** a decision.
     - Agreeing to "investigate X" is **NEVER** a decision.

2. **THE MATERIALITY FILTER (Triviality Check):**
   - **CRITICAL RULE:** Do NOT capture "housekeeping" or "administrative" trivia.
   - **EXCLUDE:** "Send me the link", "I'll invite you", "Let's sync up".
   - **INCLUDE Only if:** The task requires **significant effort** or produces a **key deliverable**.

3. **THE ACTION TEST (Business Value):**
   - **Definition:** A tactical task assigned to an owner that produces a **Business Deliverable**.
   - **Detail Rule:** Tasks must be **specific and context-rich**. Include *WHAT* to do, *WHO* is it for, and *WHY*.
   - **Bad:** "Send report."
   - **Good:** "Compile and send the Q3 user adoption report to the Board to support the new funding proposal."
   - **Rule:** Ask "If this is not done, does the project suffer?" If yes, capture it. If no, ignore it.

3. **OWNER INTEGRITY CHECK:**
   - **Strict Assignment:** Only assign an 'owner' if the transcript EXPLICITLY names them or refers to them directly (e.g., "Bob, please handle this").
   - **No Hallucinations:** Do NOT infer ownership based on who spoke the most. If unclear, use `null`.
   - **Action vs Decision Owners:**
     - **Action Items:** Must have an owner if possible.
     - **Decisions:** Rarely have "owners". Only assign if they are the sole approver.

4. **Summary:** A comprehensive, high-level overview (4-6 crisp sentences) covering the main **purpose/goals**, the final **outcomes**, and any major **follow-up** steps.
5. **Key Points:** Specific, factual, bullet-style highlights (strings) of significant **progress**, **blockers**, **metrics**, or **updates** discussed.
6. **REFERENCES:** All decisions and action_items MUST include time-based references in HH:MM:SS format (e.g., {"start": "00:05:32", "end": "00:06:15"}). Use timestamps from the transcript directly.
7. **JSON CONTRACT:**
   - Use null (not empty string) for unknown owner/due_date/priority fields.
   - Use [] for empty arrays; never use null or omit required keys.
   - Output MUST be valid JSON — no code fences, trailing commas, extra keys, or explanatory text.

CALL SCORING (PRODUCTION-GRADE CALCULATION):
You must score the meeting on 6 specific factors (0-10) based on the strict criteria below. Analyze the transcript deeply to derive these scores.

**STEP 0: AGENDA INFERENCE**
   - **Goal:** Synthesize the 'Intended Agenda' by intelligently weighing multiple signals. Use your judgment to determine what the meeting was *supposed* to be about.
   - **Signals to Consider:**
     1. **Explicit Statements:** "The goal today is...", "I want to review..." (Strongest signal).
     2. **Implicit Structure:** If no kickoff exists, look for the first substantial business topic that stabilizes after social chatter.
     3. **Noise Filtering:** Aggressively ignore "weekend talk" or "waiting for people" phases, even if they are long.
   - **Smart Synthesis:** Do NOT blindly take the first sentence. Combine these signs to infer the true purpose. If the meeting starts with a rant but then settles into a roadmap review, the roadmap review is the agenda.
   - **Formulate:** A specific 'identified_agenda' string summarizing this synthesized intent.
   - **Use this Base:** You MUST use this inferred 'identified_agenda' as the baseline for scoring 'Agenda Deviation' below.

1. **Agenda Deviation (Weight 15%):**
   - **What it measures:** How well the meeting stuck to the *identified_agenda* you found above.
   - **Scoring Rubric:**
     * **10:** Agenda clearly stated, followed, minimal unrelated discussion.
     * **7-9:** Minor deviations but agenda regained quickly.
     * **4-6:** Multiple deviations; key items partially covered.
     * **0-3:** Largely off-track; agenda not followed or never established.
   - **Signals:**
     * (+): Explicit agenda set at start + referenced again later.
     * (-): Long digressions, repeated 'we are going off-topic'.

2. **Action Items Completeness (Weight 20%):**
   - **What it measures:** Clarity and executability of action items.
   - **Scoring Rubric:**
     * **10:** Tasks are specific + actionable (what + expected output + scope).
   - **Penalties:**
     * Vague verbs ('check', 'look into') without specified deliverable.
     * Missing context (what system/module/process?).
     * No acceptance criteria.

3. **Owner Assignment Clarity (Weight 15%):**
   - **What it measures:** Whether each major action/decision has a clear accountable owner.
   - **Scoring Rubric:**
     * **10:** Almost all major items have a named owner (or 'team/role' explicitly).
   - **Penalties:**
     * 'Someone should...', 'We should...' without assigning responsibility.
     * Ambiguous owners ('they', 'you guys').

4. **Due Date Quality (Weight 10%):**
   - **What it measures:** Presence + specificity + realism of due dates.
   - **Scoring Rubric:**
     * **10:** Due dates are mostly concrete (date/time or clear timeframe) and reasonable.
   - **Penalties:**
     * No due date specified.
     * Vague timelines ('ASAP', 'soon') unless clarified.
     * Unrealistic deadlines vs task scope (flag as quality issue).

5. **Meeting Structure (Weight 15%):**
   - **What it measures:** Logical flow: problem -> discussion -> decision -> next steps.
   - **Scoring Rubric:**
     * **10:** Structured conversation, minimal looping, decisions captured cleanly.
   - **Penalties:**
     * Circular debate without closure.
     * Unclear transitions and no wrap-up.
     * Repeated re-opening of settled topics.

6. **Signal-to-Noise Ratio (SNR) (Weight 10%):**
   - **What it measures:** Efficiency and focus of conversation.
   - **Scoring Rubric:**
     * **10:** High density of useful content; low repetition/filler.
   - **Penalties:**
     * Repeated points, lengthy side chats.
     * Excessive filler, long pauses, unproductive back-and-forth.

7. **Time Management (Weight 15% - STICK TO RULES):**
   - **What it measures:** Adherence to the scheduled duration.
   - **CRITICAL RULE:** Check the "Time Status" in the metadata.
   - **Scoring Rubric:**
     * **10:** Status is ON TIME or EARLY.
     * **8:** Overrun by < 5 minutes.
     * **5:** Overrun by 5-15 minutes.
     * **0:** Overrun by > 15 minutes.

**Generate Qualitative Lists:**
   - **positive_aspects:** List 2-3 specific items where the meeting had clear wins. **RULE:** Keep each item extremely concise (max 6-8 words). Do NOT use the term 'SNR'. (e.g., 'Clear owners assigned', 'Discussion remained highly focused').
   - **areas_for_improvement:** List 2-3 specific items where the meeting showed issues. **RULE:** Keep each item extremely concise (max 6-8 words). Do NOT use the term 'SNR'. (e.g., 'Excessive repetition', 'Circular discussions', 'Too much side-talk').

{{metadata}}
Transcript:
{{transcript_block}}
"""

class CallAnalysisAgent:
    """
    Minimal call analysis agent inspired by PainPointAgent:
      - accepts a raw transcript payload and context
      - builds a compact prompt with light-weight analytics
      - requests STRICT JSON from the LLM
      - parses/validates into MeetingAnalysis
    """

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
        logger.info(f"DEBUG: Transcript has {len(transcript.conversation)} turns.")
        talk_time_stats = self._compute_talk_time(transcript)
        
        # FIX 1: Process sentiment data from the transcript payload
        sentiment = self._process_sentiment(transcript_payload)
        
        prompt = self._build_prompt(transcript, talk_time_stats, context=ctx)
        raw = await self._call_llm(prompt)
        print("DEBUG: Raw LLM Response:", raw)
        payload = self._parse_response(raw)
        print("DEBUG: Parsed Payload keys:", payload.keys())
        if "call_scoring" in payload:
            print("DEBUG: call_scoring content:", payload["call_scoring"])

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
        
        # 1. Get initial overall sentiment from payload
        overall = SentimentLabel.neutral
        try:
            overall_str = sentiments.get("overall", "neutral")
            if overall_str:
                overall = SentimentLabel(overall_str.lower())
        except ValueError:
            logger.warning(f"[CallAnalysisAgent] Invalid overall sentiment label: {sentiments.get('overall')}")
        
        # 2. Get per-speaker sentiment from multiple possible sources
        per_speaker: Dict[str, SentimentLabel] = {}
        
        # First, try the 'participant' list from sentiments
        participants = sentiments.get("participant", [])
        for p in participants:
            name = (p.get("name") or "").strip()
            sentiment_str = (p.get("sentiment") or "neutral").strip().lower()
            if name:
                try:
                    per_speaker[name] = SentimentLabel(sentiment_str)
                except ValueError:
                    per_speaker[name] = SentimentLabel.neutral
        
        # Second, try the 'speakers' array from the payload root (new format)
        speakers = payload.get("speakers", [])
        for speaker_data in speakers:
            name = (speaker_data.get("speaker") or "").strip()
            sentiment_str = (speaker_data.get("sentiment") or "neutral").strip().lower()
            if name:
                try:
                    # Update or add sentiment
                    per_speaker[name] = SentimentLabel(sentiment_str)
                except ValueError:
                    if name not in per_speaker:
                        per_speaker[name] = SentimentLabel.neutral

        # 3. Calculate calculated overall sentiment if payload's overall is neutral
        if overall == SentimentLabel.neutral and per_speaker:
            sentiment_values = {
                SentimentLabel.positive: 1,
                SentimentLabel.neutral: 0,
                SentimentLabel.negative: -1,
                SentimentLabel.mixed: 0
            }
            
            total_val = 0
            count = 0
            for s_label in per_speaker.values():
                total_val += sentiment_values.get(s_label, 0)
                count += 1
            
            if count > 0:
                avg = total_val / count
                if avg > 0.33:
                    overall = SentimentLabel.positive
                elif avg < -0.33:
                    overall = SentimentLabel.negative
                else:
                    overall = SentimentLabel.neutral
                
                logger.info(f"[CallAnalysisAgent] Calculated overall sentiment from {count} speakers: {overall} (avg: {avg:.2f})")

        return SentimentOverview(overall=overall, per_speaker=per_speaker)


    # ------------------------------------------------------------------
    # Prompt / LLM
    # ------------------------------------------------------------------
    def _build_prompt(self, transcript: MeetingTranscript, stats: List[TalkTimeStat], context: Dict[str, Any] = None) -> str:
        participants = ", ".join(p.name for p in transcript.participants) or "Unknown participants"
        talk_time_lines = "\n".join(
            f"{s.speaker}: {s.total_duration} ({s.share_percent}%), turns={s.turns}"
            for s in stats
        )

        ctx = context or {}
        meeting_title = ctx.get("meeting_title") or "Unknown Meeting"
        start_time = ctx.get("start_time") or "N/A"
        end_time = ctx.get("end_time") or "N/A"

        # Calculate Time Status
        scheduled_val = 0.0
        if start_time and end_time:
            # Assume datetime objects or parse strings if needed
            try:
                if isinstance(start_time, str):
                    s_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                    e_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                    scheduled_val = (e_dt - s_dt).total_seconds()
                else:
                    scheduled_val = (end_time - start_time).total_seconds()
            except Exception:
                scheduled_val = 0.0

        # Estimate actual duration from last turn or use approximate
        actual_val = 0.0
        if transcript.conversation:
            last_turn = transcript.conversation[-1]
            actual_val = self._hhmmss_to_seconds(last_turn.end_time)

        status_msg = "UNKNOWN (Timestamps missing)"
        if scheduled_val > 0 and actual_val > 0:
            diff = actual_val - scheduled_val
            if diff > 300: # 5 mins buffer
                overrun_mins = int(diff / 60)
                status_msg = f"OVERRUN by {overrun_mins} minutes."
            elif diff < -300:
                early_mins = int(abs(diff) / 60)
                status_msg = f"ENDED EARLY by {early_mins} minutes."
            else:
                status_msg = "ON TIME"
        
        metadata = (
            f"Tenant: {transcript.tenant_id}\n"
            f"Session: {transcript.session_id}\n"
            f"Title: {meeting_title}\n"
            f"Start Time: {start_time}\n"
            f"End Time: {end_time}\n"
            f"Time Status: {status_msg}\n"
            f"Participants: {participants}\n"
            f"Talk time:\n{talk_time_lines or 'No speech recorded.'}\n"
        )

        convo_lines: List[str] = []
        for idx, turn in enumerate(transcript.conversation):
            start_ts = self._format_timestamp(turn.start_time)
            end_ts = self._format_timestamp(turn.end_time)
            convo_lines.append(f"[{idx}] [{start_ts} - {end_ts}] {turn.speaker}: {turn.text}")

        transcript_block = "\n".join(convo_lines)
        return CALL_ANALYSIS_PROMPT_TEMPLATE.replace("{{metadata}}", metadata).replace("{{transcript_block}}", transcript_block)

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
            is_overtime=self._check_overtime(context, transcript),
            overtime_duration=self._get_overtime_duration(context, transcript),
        )
        return analysis

    def _check_overtime(self, context: Dict[str, Any], transcript: MeetingTranscript) -> bool:
        try:
            start_time = context.get("start_time")
            end_time = context.get("end_time")
            if not start_time or not end_time:
                return False
                
            if isinstance(start_time, str):
                s_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                e_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                scheduled = (e_dt - s_dt).total_seconds()
            else:
                scheduled = (end_time - start_time).total_seconds()

            actual = 0.0
            if transcript.conversation:
                actual = self._hhmmss_to_seconds(transcript.conversation[-1].end_time)
            
            print(f"DEBUG: Overtime Check - Actual: {actual}s, Scheduled: {scheduled}s")
            logger.info(f"DEBUG: Overtime Check - Actual: {actual}s, Scheduled: {scheduled}s")
            return actual > (scheduled + 300) # 5 min buffer
        except Exception as e:
            logger.info(f"DEBUG: Overtime Check Error: {e}")
            return False

    def _get_overtime_duration(self, context: Dict[str, Any], transcript: MeetingTranscript) -> str:
        try:
            start_time = context.get("start_time")
            end_time = context.get("end_time")
            if not start_time or not end_time:
                return "0 min"

            if isinstance(start_time, str):
                s_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                e_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                scheduled = (e_dt - s_dt).total_seconds()
            else:
                scheduled = (end_time - start_time).total_seconds()

            actual = 0.0
            if transcript.conversation:
                actual = self._hhmmss_to_seconds(transcript.conversation[-1].end_time)
            
            diff = actual - scheduled
            if diff > 0:
                mins = int(diff / 60)
                logger.info(f"DEBUG: Overtime Duration: {mins} min")
                return f"{mins} min"
            return "0 min"
        except Exception as e:
            logger.info(f"DEBUG: Overtime Duration Error: {e}")
            return "0 min"

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

    def _build_call_scoring(self, value: Any) -> Optional[CallScoring]:
        """
        Build call scoring object from LLM response.
        Calculates the final weighted score from the 6 components extracted from LLM.
        """
        if not isinstance(value, dict):
            return None
        
        try:
            # Extract inferred agenda (chain of thought)
            identified_agenda = (value.get("identified_agenda") or "Agenda not detected").strip()
            
            # Extract 7 raw component scores (0-10)
            agenda_score = float(value.get("agenda_deviation", 0.0))
            action_score = float(value.get("action_completeness", 0.0))
            owner_score = float(value.get("owner_clarity", 0.0))
            due_date_score = float(value.get("due_date_quality", 0.0))
            structure_score = float(value.get("structure", 0.0))
            snr_score = float(value.get("snr", 0.0))
            time_score = float(value.get("time_management_score", 0.0))
            
            # Calculate Final Score
            # Formula (Sum=1.0): 
            # 15% Agenda + 20% Action + 15% Owner + 10% DueDate + 15% Structure + 10% SNR + 15% Time
            final_score = (
                (0.15 * agenda_score) +
                (0.20 * action_score) +
                (0.15 * owner_score) +
                (0.10 * due_date_score) +
                (0.15 * structure_score) +
                (0.10 * snr_score) +
                (0.15 * time_score)
            )
            
            # Clamp to 0-10 just in case
            final_score = max(0.0, min(10.0, final_score))
            
            positive_list = self._ensure_str_list(value.get("positive_aspects", []))
            improvement_list = self._ensure_str_list(value.get("areas_for_improvement", value.get("negative_aspects", [])))

            return CallScoring(
                score=round(final_score, 2), 
                identified_agenda=identified_agenda,
                agenda_deviation_score=round(agenda_score, 2),
                action_item_completeness_score=round(action_score, 2),
                owner_clarity_score=round(owner_score, 2),
                due_date_quality_score=round(due_date_score, 2),
                meeting_structure_score=round(structure_score, 2),
                signal_noise_ratio_score=round(snr_score, 2),
                time_management_score=float(value.get("time_management_score", 0)),
                positive_aspects=positive_list,
                areas_for_improvement=improvement_list,
            )
        except (TypeError, ValueError, ValidationError) as e:
            logger.warning("[CallAnalysisAgent] Failed to build CallScoring: %s", e)
            return None

    def _format_timestamp(self, time_str: str) -> str:
        """Format timestamp - returns as-is if already in HH:MM:SS format."""
        # Time is already in HH:MM:SS format from input transformation
        return time_str
