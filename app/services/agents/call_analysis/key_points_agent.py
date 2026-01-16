from typing import Any, Dict, List, Optional
from .base import BaseAnalysisAgent

KEY_POINTS_SYSTEM_INSTRUCTION = """You are a Strategic Data Analyst. Your task is to extract high-impact, factual 'Key Points' from a meeting.

### CORE OPERATING PRINCIPLES:
- **Topic-Wise Segmentation**: Group insights by the provided V2 thematic headers.
- **Precision**: Focus on metrics, milestones, blockers, and progress shifts.
- **Noise Suppression**: Ignore "housekeeping," social chatter, or administrative overhead.
- **Rich String Formatting**: Each point must be a string in the following format: "**Topic Name**: Detailed description [Start HH:MM:SS - End HH:MM:SS]".
- **Output**: Valid JSON."""

KEY_POINTS_TASK_PROMPT = """Analyze the provided meeting transcript segments to extract factual key discussion points.

### INPUT FORMAT:
The transcript is organized into V2 blocks:
`--- TOPIC: [Name] (Type: [actionable_item/decision/key_insight/discussion/question]) ---`

### EXTRACTION RULES:
1. **Identify High-Value Blocks**: Prioritize blocks of type 'key_insight' and 'discussion'.
2. **Synthesize Factoid**: For each significant topic, extract the most important factual takeaway.
3. **Draft the Rich String**: Format the point as "**Topic Name**: Point description [Start HH:MM:SS - End HH:MM:SS]". Ensure there is only ONE string per significant insight.
4. **Context Check**: Ensure each point contains enough context to be understood independently.

### INPUT DATA:
Metadata: {{metadata}}
Transcript: {{transcript_block}}

### OUTPUT FORMAT:
Output ONLY a JSON object:
{
  "key_points": [
    "**Topic Name**: Point description [00:01:23 - 00:01:45]",
    "**Topic Name**: Another point [00:05:10 - 00:05:55]"
  ]
}
"""

class KeyPointsAgent(BaseAnalysisAgent):
    async def extract_key_points(self, transcript_block: str, metadata: str) -> List[str]:
        prompt = KEY_POINTS_TASK_PROMPT.replace("{{metadata}}", metadata).replace("{{transcript_block}}", transcript_block)
        raw = await self._call_llm(prompt, system_instruction=KEY_POINTS_SYSTEM_INSTRUCTION)
        payload = self._parse_json(raw)
        return payload.get("key_points", [])
