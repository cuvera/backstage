from typing import Any, Dict, List, Optional
from .base import BaseAnalysisAgent


KEY_POINTS_SYSTEM_INSTRUCTION = """You are a Strategic Data Analyst. Your task is to extract unique, high-impact 'Key Highlights' from a meeting.

### CORE OPERATING PRINCIPLES:
- **Zero Redundancy**: Semantically similar points MUST be merged. Each point must provide unique value.
- **Significance over Volume**: Target 5-8 high-impact highlights. Avoid granular detail.
- **Topic-Wise Merging**: Use the thematic headers to group related insights into single, powerful statements.
- **Precision**: Focus on metrics, milestones, blockers, and strategic shifts.
- **Rich String Formatting**: Each point must be: "**Topic Name**: Insight description [Start HH:MM:SS - End HH:MM:SS]".
- **Output**: Valid JSON."""

KEY_POINTS_TASK_PROMPT = """Analyze the meeting transcript to extract the most significant, non-redundant highlights.

### EXTRACTION RULES:
1. **Strategic Synthesis**: Do not just list facts. Synthesize the core takeaway of each major discussion thread.
2. **Aggressive Deduplication**: If multiple segments discuss the same root issue, merge them into one comprehensive point referencing the full time range.
3. **Draft the Rich String**: Format as "**Topic Name**: Precise highlight [Start HH:MM:SS - End HH:MM:SS]". 
4. **Context Integrity**: Ensure the description explains the *impact* or *outcome* of the point.

### INPUT DATA:
Metadata: {{metadata}}
Transcript: {{transcript_block}}

### OUTPUT FORMAT:
Output ONLY a JSON object:
{
  "key_points": [
    "**Topic Name**: Impactful highlight description [00:01:23 - 00:05:45]",
    "**Topic Name**: Unique insight statement [00:08:10 - 00:10:20]"
  ]
}
"""

class KeyPointsAgent(BaseAnalysisAgent):
    async def extract_key_points(self, transcript_block: str, metadata: str) -> List[str]:
        prompt = KEY_POINTS_TASK_PROMPT.replace("{{metadata}}", metadata).replace("{{transcript_block}}", transcript_block)
        raw = await self._call_llm(prompt, system_instruction=KEY_POINTS_SYSTEM_INSTRUCTION)
        payload = self._parse_json(raw)
        return payload.get("key_points", [])
