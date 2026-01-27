from typing import Any, Dict, List, Optional
from .base import BaseAnalysisAgent

SUMMARY_SYSTEM_INSTRUCTION = """You are an Executive Communications Specialist. 
Your task is to produce 'High-Fidelity' meeting summaries that provide immediate at-a-glance value for stakeholders.

### RULES:
- **Tone**: Professional and objective.
- **Detail**: Provide a detailed account of the meeting, focusing on the strategic *why*, final *resolution*, and key context. 
- **Highlighting**: Use standard Markdown bold syntax (`**keyword**`) to **highlight** key words, topics, and critical findings. In the user interface, these will appear as **colored text** for emphasis. Do NOT use bolding for any other purpose.
- **Integrity**: Never mention people or facts not present in the transcript.
- **Scope**: Cover the entire meeting data comprehensively.
- **Flexibility**: Autonomously decide the appropriate length of the summary based on the meeting's complexity and the amount of significant information discussed.
- **Output**: Output must be strictly valid JSON"""

SUMMARY_TASK_PROMPT = """Analyze the meeting transcript to produce a detailed, high-fidelity executive summary.

### CORE OBJECTIVES:
- Summarize the meeting's primary intent, strategic importance, and overall arc in detail.
- Highlight the major narrative shifts, key discussions, or resolutions reached.
- Use **Markdown highlighing** (wrapping key topics, terms, and outcomes in `**`) to ensure they standby. These will be rendered as colored highlights in the UI.
- Ensure the summary accurately reflects the depth and breadth of the entire meeting.
- Avoid granular lists; focus on the holistic outcome and progression while maintaining detail.

### INPUT DATA:
Metadata: {{metadata}}
Transcript: {{transcript_block}}

### OUTPUT FORMAT:
Output ONLY a JSON object:
{
  "summary": "Detailed summary in Markdown. Wrap key topics/terms in `**` for colored highlighting in the UI. The length should be determined by the richness and complexity of the meeting content."
}
"""

class SummaryAgent(BaseAnalysisAgent):
    async def summarize(self, transcript_block: str, metadata: str) -> str:
        prompt = SUMMARY_TASK_PROMPT.replace("{{metadata}}", metadata).replace("{{transcript_block}}", transcript_block)
        raw = await self._call_llm(prompt, system_instruction=SUMMARY_SYSTEM_INSTRUCTION, response_format={"type": "json_object"})
        payload = self._parse_json(raw)
        return payload.get("summary", "Summary not generated") if payload else "Summary not generated"
