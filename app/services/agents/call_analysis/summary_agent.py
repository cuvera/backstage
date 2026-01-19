from typing import Any, Dict, List, Optional
from .base import BaseAnalysisAgent

SUMMARY_SYSTEM_INSTRUCTION = """You are an Executive Communications Specialist. 
Your task is to produce 'High-Fidelity' meeting summaries that provide immediate at-a-glance value for stakeholders.

### RULES:
- **Tone**: Professional, objective, and extremely concise.
- **Brevity**: Eliminate all filler words. Maximize impact per sentence. 
- **Structure**: 1-2 short, impactful paragraphs.
- **Detail**: Focus only on the strategic *why* and the final *resolution*.
- **Integrity**: Never mention people or facts not present in the transcript.
- **Output**: Valid JSON."""

SUMMARY_TASK_PROMPT = """Analyze the meeting transcript to produce a crisp, executive summary.

### CORE OBJECTIVES:
- Summarize the meeting's primary intent and strategic importance.
- Highlight the major narrative shift or resolution reached.
- Avoid granular lists; focus on the holistic outcome.

### INPUT DATA:
Metadata: {{metadata}}
Transcript: {{transcript_block}}

### OUTPUT FORMAT:
Output ONLY a JSON object:
{
  "summary": "Crisp 1-2 short paragraphs summarizing they. Be brief but context-rich."
}
"""

class SummaryAgent(BaseAnalysisAgent):
    async def summarize(self, transcript_block: str, metadata: str) -> str:
        prompt = SUMMARY_TASK_PROMPT.replace("{{metadata}}", metadata).replace("{{transcript_block}}", transcript_block)
        raw = await self._call_llm(prompt, system_instruction=SUMMARY_SYSTEM_INSTRUCTION)
        payload = self._parse_json(raw)
        return payload.get("summary", "Summary not generated")
