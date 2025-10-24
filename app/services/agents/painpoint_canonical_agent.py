from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from app.services.llm.factory import get_llm  # type: ignore

logger = logging.getLogger(__name__)


class PainPointCanonicalAgent:
    """
    Uses the configured LLM to canonicalise a pain-point record so that
    downstream aggregations can deduplicate similar issues.
    """

    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def canonicalise(self, record: Dict[str, Any]) -> Dict[str, Any]:
        raw_text = (record.get("raw_text") or "").strip()
        enriched = record.get("enriched") or {}

        payload = {
            "raw_text": raw_text,
            "title": enriched.get("title"),
            "category": enriched.get("category"),
            "severity": enriched.get("severity"),
            "product_area": enriched.get("product_area"),
            "tags": enriched.get("tags") or [],
            "root_causes": enriched.get("root_causes") or [],
            "impacted_flows": enriched.get("impacted_flows") or [],
            "suggested_actions": enriched.get("suggested_actions") or [],
        }

        prompt = f"""
You are normalising noisy pain-point records so that analytics can group identical issues.

Given the JSON payload below, return STRICT JSON with this schema:
{{
  "canonical_key": "<lowercase hyphen slug capturing the core issue>",
  "title": "<clean concise title>",
  "category": "<Process|Communication|Tools/Systems|Workload|Policy|Environment|Coordination|Management|Scheduling|Other>",
  "severity": "<low|medium|high|critical>",
  "product_area": "<short name or null>",
  "tags": ["tag", ...],                        # <= 10, lower-case, unique, ordered by relevance
  "root_causes": ["cause", ...],               # <= 10
  "impacted_flows": ["flow", ...],             # <= 10
  "suggested_actions": ["action", ...]         # <= 10
}}

Rules:
- Derive canonical_key as 3-8 words, lower-case, hyphen-separated (e.g., "shift-schedule-delays").
- Prefer existing structured fields when sensible; only infer when strongly implied.
- Default missing arrays to [] and scalar fields to null.
- Output JSON ONLY, no explanations.

Payload:
```json
{json.dumps(payload, ensure_ascii=False)}
```
"""
        try:
            response = self.llm.complete(prompt)
            data = self._safe_json(response)
            if not data or "canonical_key" not in data:
                raise ValueError("Invalid canonicalisation JSON")
            return self._apply_defaults(data)
        except Exception as exc:
            logger.exception("[PainPointCanonicalAgent] Canonicalisation failed: %s", exc)
            return self._fallback(record)

    @staticmethod
    def _apply_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
        def _lower_slug(value: Optional[str]) -> str:
            if not value:
                return ""
            return (
                value.strip()
                .lower()
                .replace("/", " ")
                .replace("_", " ")
                .replace("  ", " ")
                .replace(" ", "-")
            )

        canonical_key = _lower_slug(data.get("canonical_key")) or _lower_slug(data.get("title"))
        tags = PainPointCanonicalAgent._dedupe_list(data.get("tags"))
        root_causes = PainPointCanonicalAgent._dedupe_list(data.get("root_causes"))
        flows = PainPointCanonicalAgent._dedupe_list(data.get("impacted_flows"))
        actions = PainPointCanonicalAgent._dedupe_list(data.get("suggested_actions"))

        return {
            "canonical_key": canonical_key or "uncategorized",
            "title": data.get("title"),
            "category": data.get("category"),
            "severity": data.get("severity"),
            "product_area": data.get("product_area"),
            "tags": tags,
            "root_causes": root_causes,
            "impacted_flows": flows,
            "suggested_actions": actions,
        }

    @staticmethod
    def _safe_json(text: str) -> Optional[Dict[str, Any]]:
        if not isinstance(text, str):
            return None
        try:
            return json.loads(text)
        except Exception:
            try:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(text[start : end + 1])
            except Exception:
                pass
        return None

    @staticmethod
    def _fallback(record: Dict[str, Any]) -> Dict[str, Any]:
        enriched = record.get("enriched") or {}
        title = (enriched.get("title") or record.get("raw_text") or "").strip()
        canonical_key = (
            title.lower().replace("/", " ").replace("_", " ").replace("  ", " ").replace(" ", "-")[:80]
            or "uncategorized"
        )
        return {
            "canonical_key": canonical_key,
            "title": title[:120] or None,
            "category": enriched.get("category"),
            "severity": enriched.get("severity"),
            "product_area": enriched.get("product_area"),
            "tags": PainPointCanonicalAgent._dedupe_list(enriched.get("tags")),
            "root_causes": PainPointCanonicalAgent._dedupe_list(enriched.get("root_causes")),
            "impacted_flows": PainPointCanonicalAgent._dedupe_list(enriched.get("impacted_flows")),
            "suggested_actions": PainPointCanonicalAgent._dedupe_list(enriched.get("suggested_actions")),
        }

    @staticmethod
    def _dedupe_list(values: Optional[Any], limit: int = 10) -> list[str]:
        if not values:
            return []
        seen = []
        for item in values:
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            if not normalized:
                continue
            if normalized not in seen:
                seen.append(normalized)
            if len(seen) >= limit:
                break
        return seen
