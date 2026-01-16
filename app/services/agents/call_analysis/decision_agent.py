from typing import Any, Dict, List, Optional
from .base import BaseAnalysisAgent

DECISION_SYSTEM_INSTRUCTION = """You are a Corporate Governance Auditor. Your task is to extract 'Strategic Decisions' from meeting transcripts.

### THE STRATEGIC GATE:
- **DEFINITION**: A decision is a commitment to a specific course of action, a strategic state change, or a final approval.
- **EXCLUSIONS**: Non-decisions include: "Let's meet again," "We should look into X," or "I'll think about it."
- **TITLE REQUIREMENT**: Titles must be self-contained (15-25 words), explaining the decision AND its intended impact.
- **OUTPUT**: Valid JSON."""

DECISION_TASK_PROMPT = """Analyze the provided meeting transcript segments to identify strategic decisions.

### ANALYSIS PROCESS:
1. **Source Check**: The input may be pre-filtered for clusters of type 'decision'. Verify if these content blocks contain actual commitments.
2. **Impact Assessment**: Only capture items that change the project's direction or resolve a previously open question.
3. **Draft Title**: Create a comprehensive title (e.g., "Approval of [Component X] architecture to mitigate [Risk Y] and ensure [Goal Z]").
4. **Reference Alignment**: Map the decision to the exact HH:MM:SS timestamps from the transcript.

### INPUT DATA:
Metadata: {{metadata}}
Transcript: {{transcript_block}}

### OUTPUT FORMAT:
Output ONLY a JSON object:
{
  "decisions": [
    {
      "title": "Comprehensive 15-25 word strategic title",
      "owner": "Directly named owner or null",
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
