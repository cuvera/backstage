from typing import Any, Dict, List, Optional
from .base import BaseAnalysisAgent

DECISION_SYSTEM_INSTRUCTION = """You are a Project Governance Officer. Your task is to extract all definitive decisions and agreements reached during a meeting.

### CORE OPERATING PRINCIPLES:
- **Comprehensive Capture**: Extract any point where a choice was made or an agreement was reached.
- **Clarity**: Titles should be specific and clear.
- **Evidence**: Every decision must be linked to its HH:MM:SS timestamps.
- **Output**: Valid JSON."""

DECISION_TASK_PROMPT = """Analyze the provided meeting transcript segments to extract all definitive decisions.

### INPUT FORMAT:
The transcript is organized into V2 blocks:
`--- TOPIC: [Name] (Type: [actionable_item/decision/key_insight/discussion/question]) ---`

### EXTRACTION RULES:
1. **Decision Segments**: Prioritize blocks where the type is 'decision'.
2. **Definitive Agreements**: Identify statements where the group agrees on a path forward, a policy, a feature, or a change.
3. **Capture Details**:
   - **Title**: A clear, descriptive title of what was decided.
   - **Owner**: The person or team responsible for the decision or its execution.
   - **References**: Exact HH:MM:SS timestamps from the transcript.

### INPUT DATA:
Metadata: {{metadata}}
Transcript: {{transcript_block}}

### OUTPUT FORMAT:
Output ONLY a JSON object:
{
  "decisions": [
    {
      "title": "Clear description of the decision",
      "owner": "Who is responsible or null",
      "due_date": null,
      "references": [{"start": "HH:MM:SS", "end": "HH:MM:SS"}]
    }
  ]
}
"""

class DecisionAgent(BaseAnalysisAgent):
    async def extract_decisions(self, transcript_block: str, metadata: str) -> List[Dict[str, Any]]:
        prompt = DECISION_TASK_PROMPT.replace("{{metadata}}", metadata).replace("{{transcript_block}}", transcript_block)
        raw = await self._call_llm(prompt, system_instruction=DECISION_SYSTEM_INSTRUCTION)
        payload = self._parse_json(raw)
        return payload.get("decisions", [])
