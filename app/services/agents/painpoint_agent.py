from __future__ import annotations
import json
import logging
from typing import Any, Dict, Optional

from pydantic import ValidationError

from app.schemas.painpoint import PainPointCreate, PainPointEnriched

logger = logging.getLogger(__name__)


class PainPointAgent:
    """
    Two-phase agent:
      - barrier(raw_text, context) -> {is_painpoint: bool, confidence: float, category?: str, reason?: str}
      - enrich(raw_text, context) -> PainPointCreate

    The idea: avoid enriching noise. Only enrich AFTER the barrier says it's worthy.
    """

    BARRIER_CONFIDENCE_THRESHOLD = 0.85

    def __init__(self, llm=None):
        if llm is None:
            from app.services.llm.factory import get_llm  # type: ignore
            self.llm = get_llm()
        else:
            self.llm = llm

    def barrier(self, raw_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        text = (raw_text or "").strip()

        if len(text) < 12:
            logger.info("[PainPointAgent] Barrier: reject (too short)")
            return {"is_painpoint": False, "confidence": 0.20, "reason": "too_short"}

        smalltalk = {
            "hi", "hello", "hey", "thanks", "thank you", "ok", "k",
            "good morning", "good night", "gm", "gn", "yo", "sup"
        }
        if text.lower() in smalltalk:
            logger.info("[PainPointAgent] Barrier: reject (small talk)")
            return {"is_painpoint": False, "confidence": 0.20, "reason": "small_talk"}

        prompt = f"""
Classify whether the message describes a WORKPLACE PAIN POINT.

Return ONLY strict JSON on one line:
{{"is_painpoint": <bool>, "confidence": <number 0..1>, "category": "<string>"}}

Decision rule (BE CONSERVATIVE):
- Default to false unless the message explicitly states a concrete, current work issue/blocker
  tied to process, communication, tools/systems, workload, policy, environment, coordination,
  management, or scheduling—AND shows impact/friction.
- Exclude: requests for info/help (“can you do X”, “send me Y”), planning/brainstorming,
  generic tasks/commands, greetings/thanks, HR recruiting/training questions, CAPA/report asks,
  generic “learn more” queries, personal issues unrelated to work execution.

Categories (pick one; else "Other"):
[Process, Communication, Tools/Systems, Workload, Policy, Environment, Coordination, Management, Scheduling, Other]

Confidence guidance:
0.95–0.85: explicit work problem/blocker with clear impact
0.70–0.84: likely pain point but missing detail (treat as leaning false)
<0.70: not a pain point or too ambiguous

Output rules:
- JSON only. No prose/markdown/comments.
- Booleans in lowercase.
- confidence is a decimal 0..1.

Few-shot negatives (do NOT echo):
Input: "give me the CAPA analysis of what last we discussed"
Output: {{"is_painpoint": false, "confidence": 0.06, "category": "Other"}}
Input: "hii can u tell me more about how we can have a best candidate related to hiring"
Output: {{"is_painpoint": false, "confidence": 0.08, "category": "Other"}}
Input: "can you send the meeting link?"
Output: {{"is_painpoint": false, "confidence": 0.10, "category": "Other"}}
Input: "please prepare the weekly report"
Output: {{"is_painpoint": false, "confidence": 0.12, "category": "Other"}}
Input: "what’s the status of line A?"
Output: {{"is_painpoint": false, "confidence": 0.15, "category": "Other"}}
Input: "good morning team"
Output: {{"is_painpoint": false, "confidence": 0.05, "category": "Other"}}

Few-shot positives (do NOT echo):
Input: "the wifi on the shop floor keeps dropping; scanners disconnect and we have to rescan items"
Output: {{"is_painpoint": true, "confidence": 0.93, "category": "Tools/Systems"}}
Input: "the new approval flow holds orders for hours every morning; we miss truck cutoffs"
Output: {{"is_painpoint": true, "confidence": 0.91, "category": "Process"}}
Input: "shift plan is still missing for tomorrow, half the team doesn't know their slot"
Output: {{"is_painpoint": true, "confidence": 0.89, "category": "Scheduling"}}

Message: ```{text}```
"""
        logger.debug("[PainPointAgent] Barrier prompt => %s", prompt)

        try:
            reply = self.llm.complete(prompt)
        except Exception as e:
            logger.exception("[PainPointAgent] Barrier: LLM error => %s", e)
            return {"is_painpoint": False, "confidence": 0.0, "reason": "llm_error"}

        parsed = self._safe_json(
            reply,
            fallback={"is_painpoint": False, "confidence": 0.0, "category": "Other"}
        )

        is_pp = bool(parsed.get("is_painpoint", False))
        try:
            conf = float(parsed.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(conf, 1.0))
        category = (parsed.get("category") or "Other").strip() or "Other"

        if is_pp and conf >= self.BARRIER_CONFIDENCE_THRESHOLD:
            logger.info("[PainPointAgent] Barrier: accept (conf=%.2f, category=%s)", conf, category)
            return {"is_painpoint": True, "confidence": conf, "category": category}

        logger.info("[PainPointAgent] Barrier: reject (is_pp=%s, conf=%.2f, category=%s)", is_pp, conf, category)
        return {"is_painpoint": False, "confidence": conf, "category": category, "reason": "below_threshold_or_negative"}

    def enrich(self, raw_text: str, context: Dict[str, Any]) -> PainPointCreate:
        base_context = {
            "tenant_id": context["tenant_id"],
            "user_id": context["user_id"],
            "source": context.get("source", "chat"),
            "raw_text": raw_text,
            "session_id": context.get("session_id"),
            "department": context.get("department"),
            "metadata": context.get("metadata", {}),
            "message_id": context["message_id"],
        }

        schema_hint = (
            "{"
            "\"enriched\": {"
            "\"title\": string, \"category\": string, \"severity\": one_of[low,medium,high,critical], "
            "\"persona\": string|null, \"product_area\": string|null, \"tags\": [string], \"root_causes\": [string], "
            "\"impacted_flows\": [string], \"suggested_actions\": [string]"
            "}"
            "}"
        )

        prompt = f"""
Extract a STRICT JSON object describing a WORKPLACE pain point.
Return ONLY valid JSON matching exactly this schema:
{schema_hint}

Field rules:
- title: <= 80 chars, concise, human-readable.
- category: one of [Process, Communication, Tools/Systems, Workload, Policy, Environment, Coordination, Management, Scheduling, Other].
- severity: low|medium|high|critical (by operational impact, not emotion).
  • critical: blocks core work for many or involves safety/security risk
  • high: major disruption or repeated failures; hard workaround
  • medium: noticeable friction; clear workaround exists
  • low: minor annoyance or cosmetic issue
- persona: short role if obvious (e.g., "operator", "supervisor", "analyst", "technician", "agent"); else null.
- product_area: brief process/tool/module name if implied (e.g., "Procurement", "Shift Planning", "Wi-Fi", "ERP"); else null.
- tags/root_causes/impacted_flows/suggested_actions: arrays; use [] when unknown.
- Prefer [] or null over guessing; do not invent facts not stated or safely implied.

Output rules:
- JSON only. No trailing commas. No extra text. Single object.
- Strings must be plain text (no backticks or markdown).

Mini example (do not echo):
Input: "wifi drops on the shop floor every hour; scanners disconnect"
Output: {{"enriched":{{"title":"Shop-floor Wi-Fi drops causing scanner disconnects","category":"Tools/Systems","severity":"high","persona":"operator","product_area":"Wi-Fi","tags":["network","scanner"],"root_causes":[],"impacted_flows":["inventory scanning"],"suggested_actions":["check access points","add signal repeaters","escalate to IT"]}}}}

Message: ```{raw_text}```
"""
        logger.debug("[PainPointAgent] Enrich prompt => %s", prompt)

        try:
            reply = self.llm.complete(prompt)
            data = self._safe_json(reply)
        except Exception as e:
            logger.exception("[PainPointAgent] Enrich: LLM error => %s", e)
            raise

        if not data or "enriched" not in data:
            logger.error("[PainPointAgent] Enrich: invalid JSON returned")
            raise ValueError("Invalid enrichment JSON returned by LLM")

        try:
            enriched = PainPointEnriched(**data["enriched"])
        except ValidationError as ve:
            logger.exception("[PainPointAgent] Enrich: validation error => %s", ve)
            raise

        payload = PainPointCreate(enriched=enriched, **base_context)
        logger.info(
            "[PainPointAgent] Enriched payload | title='%s' category=%s severity=%s",
            payload.enriched.title,
            payload.enriched.category,
            payload.enriched.severity,
        )
        return payload

    @staticmethod
    def _safe_json(text: str, fallback: Optional[dict] = None) -> Optional[dict]:
        if not isinstance(text, str):
            return fallback
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
        return fallback
