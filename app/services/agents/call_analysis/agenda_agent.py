from typing import Any, Dict, List, Optional
from .base import BaseAnalysisAgent

AGENDA_SYSTEM_INSTRUCTION = """You are a Principal Meeting Architect with 15+ years of experience in corporate strategy and organizational effectiveness. 
Your specialty is 'Contextual Anchoring'—identifying the true, strategic intent of a meeting despite social chatter or vague kickoff statements.

NEVER hallucinate metrics or participants not mentioned.
ALWAYS output valid JSON."""

AGENDA_TASK_PROMPT = """Analyze the provided meeting transcript and metadata to infer the 'Intended Agenda'.

### THE ANCHORING PROCESS:
1. **Filter Noise**: Ignore social chatter, "waiting for person X", or audio checks.
2. **Identify the Trigger**: Look for the MOMENT the conversation stabilizes into a business topic.
3. **Synthesize Intent**: Combine explicit kickoff statements (e.g., "We are here to review...") with the actual first substantial topic of discussion.
4. **Final Output**: Summarize the synthesized intent into a single, professional sentence.

### INPUT DATA:
Metadata: {{metadata}}
Transcript: {{transcript_block}}

### OUTPUT FORMAT:
Output ONLY a JSON object:
{
  "identified_agenda": "string summarizing the synthesized intent"
}
"""

class AgendaAgent(BaseAnalysisAgent):
    async def infer_agenda(self, transcript_block: str, metadata: str) -> str:
        prompt = AGENDA_TASK_PROMPT.replace("{{metadata}}", metadata).replace("{{transcript_block}}", transcript_block)
        raw = await self._call_llm(prompt, system_instruction=AGENDA_SYSTEM_INSTRUCTION)
        payload = self._parse_json(raw)
        return payload.get("identified_agenda", "Agenda not detected")
