from typing import Any, Dict, List, Optional
from .base import BaseAnalysisAgent

SCORING_SYSTEM_INSTRUCTION = """You are a Senior Quality Auditor. Your task is to objectively score meeting effectiveness and provide razor-sharp feedback.

### CORE OPERATING PRINCIPLES:
1. **Impact-Driven Feedback**: Positive aspects and improvement areas MUST be short, punchy, "impact" sentences (max 10-12 words).
2. **Brutal Honesty**: No fluff. No professional jargon. Be direct.
3. **Metric Alignment**: Ensure scores (0-10) directly reflect the identified gaps.
4. **Contextual Awareness**: Use the identified agenda as the benchmark."""

SCORING_TASK_PROMPT = """Analyze the meeting effectiveness based on the transcript and inferred agenda.

### AUDIT CRITERIA:
- **Agenda Adherence**: Did the team stick to the intent: {{identified_agenda}}?
- **Action Item Quality**: Are owners/tasks specific?
- **Owner Clarity**: Is it clear who is doing what?
- **Due Date Presence**: Are commitments time-bound?
- **Structure**: Is there a logical flow from problem to solution?
- **Signal-to-Noise**: Is the meeting efficient?
- **Time Management**: Did the meeting end with clear next steps?

### INPUT DATA:
Metadata: {{metadata}}
Transcript: {{transcript_block}}
Inferred Agenda: {{identified_agenda}}

### OUTPUT RULES:
- **positive_aspects**: Short, energetic wins.
- **areas_for_improvement**: Sharp, actionable fixes.
- **Output**: Valid JSON only.

### OUTPUT FORMAT:
{
  "score": 0.0,
  "agenda_deviation_score": 0.0,
  "action_item_completeness_score": 0.0,
  "owner_clarity_score": 0.0,
  "due_date_quality_score": 0.0,
  "meeting_structure_score": 0.0,
  "signal_noise_ratio_score": 0.0,
  "time_management_score": 0.0,
  "positive_aspects": ["Concise win 1", "Concise win 2"],
  "areas_for_improvement": ["Sharp fix 1", "Sharp fix 2"]
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
