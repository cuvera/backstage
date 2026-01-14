# =============================
# FILE: app/services/processors/classification_processor.py
# PURPOSE:
#   LLM-based processor for classifying normalized transcription segments.
#   Identifies topics and types (actionable_item, decision, key_insight, question).
#   Maintains strict chronological ordering.
# =============================

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.core.llm_client import llm_client
from app.core.prompts import SEGMENT_CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)


class SegmentClassificationProcessor:
    """
    LLM-based classification processor for normalized transcription segments.
    Uses enhanced LLM client with multi-provider fallback support.
    """

    # Target segment types (general_discussion is classified but filtered out)
    TARGET_TYPES = {
        "actionable_item",
        "decision",
        "key_insight",
        "question",
    }

    ALL_TYPES = TARGET_TYPES | {"general_discussion"}

    def __init__(self):
        self.llm_client = llm_client

    async def classify_segments(
        self,
        normalized_transcription: Dict[str, Any],
        options: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Classify normalized segments into topics and types.

        Args:
            normalized_transcription: Normalized transcription with 'transcriptions' list
            options: Configuration options for classification

        Returns:
            List of cluster definitions with segment_ids, topic, and type
        """
        options = options or {}

        segments = normalized_transcription.get("transcriptions", [])

        if not segments:
            logger.warning("[Classification] No segments to classify")
            return []

        logger.info(f"[Classification] Starting classification for {len(segments)} segments")

        # Call LLM to classify all segments
        cluster_definitions = await self._call_llm(segments, options)

        # Filter to target types only
        filtered_clusters = [
            cluster for cluster in cluster_definitions
            if cluster.get("type") in self.TARGET_TYPES
        ]

        logger.info(
            f"[Classification] Classified into {len(cluster_definitions)} clusters, "
            f"{len(filtered_clusters)} target clusters after filtering"
        )

        return filtered_clusters

    async def _call_llm(
        self,
        segments: List[Dict],
        options: Dict
    ) -> List[Dict[str, Any]]:
        """
        Call LLM to classify segments with fallback support.

        Args:
            segments: List of normalized segments
            options: Classification options including provider_chain

        Returns:
            List of cluster definitions
        """
        # Build segment text with references
        segment_list = self._build_segment_list(segments)

        # Build classification prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(segment_list)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Get provider chain from options (defaults to config chain)
        provider_chain = options.get("provider_chain")
        temperature = options.get("temperature", 0.0)
        max_tokens = options.get("max_tokens", 20480)

        try:
            # Use enhanced LLM client with fallback support
            response_text = await self.llm_client.chat_completion(
                messages=messages,
                provider_chain=provider_chain,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )

            # Parse JSON response
            response_data = json.loads(response_text)
            clusters = response_data.get("clusters", [])

            logger.info(f"[Classification] LLM returned {len(clusters)} clusters")

            return clusters

        except json.JSONDecodeError as e:
            logger.error(f"[Classification] Failed to parse LLM response: {e}")
            return []

        except Exception as e:
            logger.error(f"[Classification] LLM call failed: {e}")
            return []

    def _build_segment_list(self, segments: List[Dict]) -> str:
        """
        Build formatted segment list for LLM prompt.
        """
        lines = []

        for i, seg in enumerate(segments, 1):
            segment_id = seg.get("segment_id", f"seg_{i}")
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "")
            start = seg.get("start", 0)

            lines.append(
                f"{i}. [{segment_id}] ({start:.1f}s) {speaker}: {text}"
            )

        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        """
        Build system prompt for classification.
        """
        return SEGMENT_CLASSIFICATION_PROMPT

    def _build_user_prompt(self, segment_list: str) -> str:
        """
        Build user prompt with segment list.
        """
        return f"""Analyze the following meeting segments and classify them into clusters.
        **Meeting Segments:**
        {segment_list}
        """
