import json
import logging
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from app.core.openai_client import llm_client
from app.core.config import settings

logger = logging.getLogger(__name__)

class BaseAnalysisAgent:
    """Base class for all specialized meeting analysis agents."""

    def __init__(self, llm=None):
        self.llm = llm or llm_client

    async def _call_llm(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None,
        model: str = "gemini-2.0-flash", 
        temperature: float = 0.1,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """Helper to call LLM with production-grade instruction sets."""
        start_time = time.time()
        
        # Prepare messages
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # Use structured output if the model supports it and format is provided
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = await self.llm.chat.completions.create(**kwargs)
            text = response.choices[0].message.content
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"[{self.__class__.__name__}] LLM call completed in {duration_ms}ms")
            return text.strip() if text else ""
        except Exception as exc:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.error(f"[{self.__class__.__name__}] LLM call failed after {duration_ms}ms: {exc}")
            raise

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """Strip code fences, handle nested JSON, and provide fallback for production stability."""
        if not raw:
            return {}
            
        cleaned = self._strip_code_fence(raw.strip())
        
        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
            
        # Try extracting the first valid JSON object
        try:
            # Look for the outermost {}
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end > start:
                snippet = cleaned[start : end + 1]
                return json.loads(snippet)
        except json.JSONDecodeError as e:
            logger.error(f"[{self.__class__.__name__}] JSON Parse Error: {e} | Raw snippet: {cleaned[:200]}...")
            # If parsing fails, we return an empty dict to prevent total failure
            # In a real production system, you might want to retry or use a fallback LLM
            return {}

    def _strip_code_fence(self, text: str) -> str:
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return text

    def _mmss_to_hhmmss(self, mmss: str) -> str:
        try:
            parts = mmss.strip().split(":")
            if len(parts) == 2:
                minutes, seconds = int(parts[0]), int(parts[1])
                return f"{minutes // 60:02d}:{minutes % 60:02d}:{seconds:02d}"
            return mmss if len(parts) == 3 else "00:00:00"
        except:
            return "00:00:00"

    def _hhmmss_to_seconds(self, hhmmss: str) -> float:
        try:
            parts = hhmmss.strip().split(":")
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            return 0.0
        except:
            return 0.0
