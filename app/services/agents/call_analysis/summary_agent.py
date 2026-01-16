from typing import Any, Dict, List, Optional
from .base import BaseAnalysisAgent

SUMMARY_SYSTEM_INSTRUCTION = """You are an Executive Communications Specialist. 
Your task is to produce 'High-Fidelity' meeting summaries that provide immediate at-a-glance value for stakeholders who did not attend.

### RULES:
- **Tone**: Professional, objective, and analytical.
- **Structure**: Multi-paragraph, using thematic headers if necessary (within the string).
- **Detail**: Capturing the *why* behind decisions, not just the *what*.
- **Integrity**: Never mention people or facts not present in the transcript.
- **Output**: Valid JSON."""

SUMMARY_TASK_PROMPT = """Analyze the provided meeting transcript to produce a comprehensive overview.

### ANALYSIS LAYERS:
1. **Core Purpose**: Why was this meeting called?
2. **Key Narrative**: What was the primary flow of discussion? What were the divergent opinions or blockers?
3. **Strategic Context**: How does this discussion relate to broader project goals (based on the transcript)?
4. **Resolution**: What were the definitive outcomes or final stances reached?

### INPUT DATA:
Metadata: {{metadata}}
Transcript: {{transcript_block}}

### OUTPUT FORMAT:
Output ONLY a JSON object:
{
  "summary": "Detailed multi-paragraph overview string. Use context-rich language and clear thematic transitions."
}
"""

class SummaryAgent(BaseAnalysisAgent):
    async def summarize(self, transcript_block: str, metadata: str) -> str:
        prompt = SUMMARY_TASK_PROMPT.replace("{{metadata}}", metadata).replace("{{transcript_block}}", transcript_block)
        raw = await self._call_llm(prompt, system_instruction=SUMMARY_SYSTEM_INSTRUCTION)
        payload = self._parse_json(raw)
        return payload.get("summary", "Summary not generated")
