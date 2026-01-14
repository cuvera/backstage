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
        temperature = options.get("temperature", 0.3)
        max_tokens = options.get("max_tokens", 8192)

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
        return """You are an expert meeting analyst specializing in extracting structured insights from meeting transcriptions.

Your task is to analyze meeting segments and group them into meaningful clusters based on BOTH topic and type.

**Segment Types:**
1. **actionable_item**: Tasks, action items, assignments, TODOs
2. **decision**: Decisions made, conclusions reached, agreements
3. **key_insight**: Important insights, discoveries, realizations, key points
4. **question**: Questions asked (whether answered or not)
5. **general_discussion**: General conversation that doesn't fit other categories

**Critical Requirements:**
1. **STRICT CHRONOLOGICAL ORDER**: Clusters MUST be ordered by the timestamp of their first segment. NEVER reorder by type or topic.
2. **COMPREHENSIVE GROUPING**: Every segment must be included in exactly one cluster.
3. **DUAL TAGGING**: Each cluster has both a descriptive topic and a specific type.
4. **LOGICAL GROUPING**: Group consecutive segments about the same topic, even if speakers change.
5. **TYPE ACCURACY**: Classify type based on content, not just keywords.

**Output Format:**
Return a JSON object with a "clusters" array. Each cluster has:
- segment_ids: Array of segment IDs in this cluster (in order)
- topic: Brief descriptive topic (3-8 words)
- type: One of the 5 types listed above

Example:
{
  "clusters": [
    {
      "segment_ids": ["seg_1", "seg_2"],
      "topic": "Project timeline discussion",
      "type": "general_discussion"
    },
    {
      "segment_ids": ["seg_3"],
      "topic": "Launch date decision",
      "type": "decision"
    },
    {
      "segment_ids": ["seg_4", "seg_5"],
      "topic": "UI design tasks",
      "type": "actionable_item"
    }
  ]
}

Remember: Maintain chronological order and include all segments!"""

    def _build_user_prompt(self, segment_list: str) -> str:
        """
        Build user prompt with segment list.
        """
        return f"""Analyze the following meeting segments and classify them into clusters.

**Meeting Segments:**
{segment_list}

**Instructions:**
1. Read through all segments carefully
2. Group consecutive segments discussing the same topic
3. Assign each cluster a descriptive topic and appropriate type
4. Maintain strict chronological order (by segment timestamp)
5. Ensure ALL segments are included exactly once

Return the clusters in JSON format as specified."""
