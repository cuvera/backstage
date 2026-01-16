from typing import Any, Dict, List, Optional
from .base import BaseAnalysisAgent

SCORING_SYSTEM_INSTRUCTION = """You are a Senior Meeting Quality Auditor. Your task is to perform a nuanced, 7-factor audit of a meeting's effectiveness.

### THE AUDIT FRAMEWORK:
- **Agenda Deviation**: How strictly did the team stick to the 'Identified Agenda'?
- **Action Item Completeness**: Do tasks have high clarity (WHAT, WHO, WHY)?
- **Owner Clarity**: Is there zero ambiguity about who owns which takeaway?
- **Due Date Quality**: Were specific dates or firm timeframes established?
- **Meeting Structure**: Was there a logical flow (Kickoff -> Discussion -> Closing)?
- **Signal-to-Noise Ratio**: How much of the discussion was high-value vs social/trivia?
- **Time Management**: Did the meeting achieve its agenda within the implied or scheduled time?

### FEEDBACK RULES:
- **Positive Aspects**: Highlight specific moments of high collaboration or clarity.
- **Improvements**: Provide actionable, professional advice for the next session.
- **Output**: Valid JSON."""

SCORING_TASK_PROMPT = """Analyze the meeting transcript and context to generate the quality audit.

### INPUT DATA:
Identified Agenda: {{identified_agenda}}
Metadata: {{metadata}}
Transcript: {{transcript_block}}

### OUTPUT FORMAT:
Output ONLY a JSON object:
{
  "score": float (0.0 to 10.0),
  "agenda_deviation_score": float (0.0 to 10.0),
  "action_item_completeness_score": float (0.0 to 10.0),
  "owner_clarity_score": float (0.0 to 10.0),
  "due_date_quality_score": float (0.0 to 10.0),
  "meeting_structure_score": float (0.0 to 10.0),
  "signal_noise_ratio_score": float (0.0 to 10.0),
  "time_management_score": float (0.0 to 10.0),
  "positive_aspects": ["string 1", "string 2"],
  "areas_for_improvement": ["string 1", "string 2"]
}
"""

class CallScoringAgent(BaseAnalysisAgent):
    async def score_call(self, transcript_block: str, metadata: str, identified_agenda: str) -> Dict[str, Any]:
        prompt = SCORING_TASK_PROMPT.replace("{{metadata}}", metadata)
        prompt = prompt.replace("{{transcript_block}}", transcript_block)
        prompt = prompt.replace("{{identified_agenda}}", identified_agenda)
        
        raw = await self._call_llm(prompt, system_instruction=SCORING_SYSTEM_INSTRUCTION)
        payload = self._parse_json(raw)
        return payload
