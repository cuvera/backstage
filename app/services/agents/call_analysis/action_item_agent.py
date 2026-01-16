from typing import Any, Dict, List, Optional
from .base import BaseAnalysisAgent

ACTION_ITEM_SYSTEM_INSTRUCTION = """You are a Project Management Specialist. Your task is to extract 'Material Action Items' from meeting transcripts.

### THE MATERIALITY FILTER:
- **DEFINITION**: An action item is a specific, assigned task that requires effort and produces a deliverable.
- **EXCLUSIONS**: Ignore "housekeeping" (e.g., "I'll send the link") or social commitments (e.g., "Let's grab coffee").
- **IDENTITY**: Only assign an owner if explicitly named or clearly identifiable from the context. Use 'null' if uncertain.
- **OUTPUT**: Valid JSON."""

ACTION_ITEM_TASK_PROMPT = """Analyze the provided meeting transcript segments to extract actionable tasks.

### EXTRACTION PROCESS:
1. **Source Check**: The input may be pre-filtered for clusters of type 'actionable_item'. Verify if these content blocks contain actual business deliverables.
2. **Contextual Enrichment**: For each task, extract the WHO (owner), WHAT (task details), WHY (purpose/context), and WHEN (due date).
3. **Reference Mapping**: Associate each action item with the exact HH:MM:SS timestamps where it was discussed.

### INPUT DATA:
Metadata: {{metadata}}
Transcript: {{transcript_block}}

### OUTPUT FORMAT:
Output ONLY a JSON object:
{
  "action_items": [
    {
      "task": "Detailed description containing context (WHAT & WHY)",
      "owner": "Specific name or null",
      "due_date": "Specific date/timeframe or null",
      "priority": "High/Medium/Low or null",
      "references": [{"start": "HH:MM:SS", "end": "HH:MM:SS"}]
    }
  ]
}
"""

class ActionItemAgent(BaseAnalysisAgent):
    async def extract_action_items(self, transcript_block: str, metadata: str) -> List[Dict[str, Any]]:
        prompt = ACTION_ITEM_TASK_PROMPT.replace("{{metadata}}", metadata).replace("{{transcript_block}}", transcript_block)
        raw = await self._call_llm(prompt, system_instruction=ACTION_ITEM_SYSTEM_INSTRUCTION)
        payload = self._parse_json(raw)
        return payload.get("action_items", [])
