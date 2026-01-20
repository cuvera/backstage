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

    # Target segment types
    TARGET_TYPES = {
        "actionable_item",
        "decision",
        "key_insight",
        "question",
        "general_discussion",
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
        # Log available segment_ids for debugging
        segment_ids_available = [seg.get("segment_id") for seg in segments]
        logger.info(f"[Classification] Available segment_ids: {len(segment_ids_available)} segments")
        logger.info(f"[Classification] Available segment_ids (first 20): {sorted(segment_ids_available)[:20]}")
        logger.debug(f"[Classification] Segment IDs: {segment_ids_available[:20]}... (showing first 20)")

        # Build classification prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(segments)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Get provider chain from options (defaults to config chain)
        temperature = options.get("temperature", 0.0)

        provider_chain = options.get("provider_chain")
        if provider_chain is None:
            provider_chain = self.llm_client._default_provider_chain()

        try:
            # Use enhanced LLM client with fallback support
            response_text = await self.llm_client.chat_completion(
                messages=messages,
                provider_chain=provider_chain,
                temperature=temperature,
                response_format={"type": "json_object"}
            )

            logger.info(f"[Classification] Received LLM response: {len(response_text)} characters")

            # Parse JSON response
            response_data = json.loads(response_text)
            clusters = response_data.get("clusters", [])

            logger.info(f"[Classification] LLM returned {len(clusters)} clusters")

            # Log segment_ids returned by LLM for debugging
            llm_segment_ids = set()
            for cluster in clusters:
                seg_ids = cluster.get("segment_ids", [])
                llm_segment_ids.update(seg_ids)

            logger.info(f"[Classification] LLM returned segment_ids (first 30): {sorted(llm_segment_ids)[:30]}")
            logger.debug(f"[Classification] LLM returned segment_ids: {sorted(llm_segment_ids)[:30]}... (showing first 30)")

            # Check for segment_ids not in available set
            invalid_ids = llm_segment_ids - set(segment_ids_available)
            if invalid_ids:
                logger.warning(
                    f"[Classification] LLM returned {len(invalid_ids)} invalid segment_ids not in normalized transcript: "
                    f"{sorted(invalid_ids)[:20]}. Available IDs: {sorted(segment_ids_available)[:20]}"
                )

            return clusters

        except json.JSONDecodeError as e:
            logger.error(
                f"[Classification] Failed to parse LLM response: {e} | "
                f"response_length={len(response_text)} characters"
            )

            # Log response excerpts for debugging (no file writes)
            logger.error(
                f"[Classification] Response preview (first 500 chars): "
                f"{response_text[:500]}"
            )
            logger.error(
                f"[Classification] Response preview (last 500 chars): "
                f"{response_text[-500:]}"
            )
            logger.debug(
                f"[Classification] Full response: {response_text}"
            )

            return []

        except Exception as e:
            logger.error(f"[Classification] LLM call failed: {e}")
            return []

    def _build_system_prompt(self) -> str:
        """
        Build system prompt for classification.
        """
        return SEGMENT_CLASSIFICATION_PROMPT

    def _build_user_prompt(self, segments: List[Dict]) -> str:
        """
        Build user prompt with segments as JSON.
        """
        return f"""Analyze the following meeting segments and classify them into clusters.

**Meeting Segments:**
```json
{json.dumps(segments, indent=2)}
```

Use the segment_id values from the JSON above in your cluster definitions."""
