# =============================
# FILE: app/services/processors/normalization_processor.py
# PURPOSE:
#   Algorithmic processor for normalizing V1 transcription output.
#   Handles duplicate removal, overlap resolution, fragment merging,
#   speaker normalization, text cleanup, and timestamp validation.
# =============================

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NormalizationProcessor:
    """
    Algorithmic normalization processor for V1 transcription output.
    No LLM usage - pure algorithmic approach.
    """

    def normalize(
        self,
        v1_transcription: Dict[str, Any],
        options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Main normalization pipeline.

        Args:
            v1_transcription: V1 transcription output with 'transcriptions' list
            options: Configuration options for normalization

        Returns:
            Normalized transcription with same structure as input
        """
        options = options or {}

        segments = v1_transcription.get("transcriptions", [])
        original_count = len(segments)

        logger.info(f"[Normalization] Starting normalization with {original_count} segments")

        # Step 1: Remove duplicates
        segments = self._remove_duplicates(segments, options)
        logger.info(f"[Normalization] After duplicate removal: {len(segments)} segments")

        # Step 2: Resolve overlaps
        segments = self._resolve_overlaps(segments)
        logger.info(f"[Normalization] After overlap resolution: {len(segments)} segments")

        # Step 3: Merge fragments
        segments = self._merge_fragments(segments, options)
        logger.info(f"[Normalization] After fragment merging: {len(segments)} segments")

        # Step 4: Normalize speaker names
        segments = self._normalize_speaker_names(segments)
        logger.info(f"[Normalization] After speaker normalization: {len(segments)} segments")

        # Step 5: Cleanup text
        segments = self._cleanup_text(segments, options)
        logger.info(f"[Normalization] After text cleanup: {len(segments)} segments")

        # Step 6: Reorder by timestamp
        segments = self._reorder_segments(segments)

        # Step 7: Validate timestamps
        segments = self._validate_timestamps(segments)

        logger.info(
            f"[Normalization] Completed: {original_count} → {len(segments)} segments "
            f"({original_count - len(segments)} removed)"
        )

        return {
            "transcriptions": segments,
            "metadata": {
                "original_segment_count": original_count,
                "normalized_segment_count": len(segments),
                "segments_removed": original_count - len(segments),
            }
        }

    def _remove_duplicates(
        self,
        segments: List[Dict],
        options: Dict
    ) -> List[Dict]:
        """
        Remove duplicate segments based on text similarity and timestamp overlap.
        """
        similarity_threshold = options.get("duplicate_similarity_threshold", 0.85)

        if not segments:
            return []

        unique_segments = []

        for segment in segments:
            is_duplicate = False

            for existing in unique_segments:
                # Check text similarity
                similarity = SequenceMatcher(
                    None,
                    segment.get("text", "").lower().strip(),
                    existing.get("text", "").lower().strip()
                ).ratio()

                # Check timestamp overlap
                start1, end1 = segment.get("start", 0), segment.get("end", 0)
                start2, end2 = existing.get("start", 0), existing.get("end", 0)

                overlap = max(0, min(end1, end2) - max(start1, start2))
                duration1 = end1 - start1
                duration2 = end2 - start2

                overlap_ratio = 0
                if duration1 > 0 and duration2 > 0:
                    overlap_ratio = overlap / min(duration1, duration2)

                # Mark as duplicate if high similarity and significant overlap
                if similarity >= similarity_threshold and overlap_ratio > 0.5:
                    is_duplicate = True
                    logger.debug(
                        f"[Normalization] Duplicate found: similarity={similarity:.2f}, "
                        f"overlap={overlap_ratio:.2f}"
                    )
                    break

            if not is_duplicate:
                unique_segments.append(segment)

        return unique_segments

    def _resolve_overlaps(self, segments: List[Dict]) -> List[Dict]:
        """
        Resolve timestamp overlaps by adjusting boundaries.
        """
        if len(segments) <= 1:
            return segments

        # Sort by start time
        sorted_segments = sorted(segments, key=lambda s: s.get("start", 0))
        resolved = []

        for i, segment in enumerate(sorted_segments):
            if i == 0:
                resolved.append(segment.copy())
                continue

            prev = resolved[-1]
            current = segment.copy()

            # Check for overlap
            if current["start"] < prev["end"]:
                # Adjust boundary at midpoint
                midpoint = (prev["end"] + current["start"]) / 2
                prev["end"] = midpoint
                current["start"] = midpoint

                logger.debug(
                    f"[Normalization] Resolved overlap between segments at {midpoint:.2f}s"
                )

            resolved.append(current)

        return resolved

    def _merge_fragments(
        self,
        segments: List[Dict],
        options: Dict
    ) -> List[Dict]:
        """
        Merge short fragments from the same speaker into coherent segments.
        """
        min_fragment_duration = options.get("min_fragment_duration", 2.0)
        max_gap = options.get("max_merge_gap", 1.0)

        if not segments:
            return []

        # Sort by start time
        sorted_segments = sorted(segments, key=lambda s: s.get("start", 0))
        merged = []

        i = 0
        while i < len(sorted_segments):
            current = sorted_segments[i].copy()
            duration = current.get("end", 0) - current.get("start", 0)

            # If current segment is short and not ending with punctuation
            if (duration < min_fragment_duration and
                not self._ends_with_punctuation(current.get("text", ""))):

                # Try to merge with next segment
                if i + 1 < len(sorted_segments):
                    next_seg = sorted_segments[i + 1]

                    # Check if same speaker and close enough
                    gap = next_seg.get("start", 0) - current.get("end", 0)
                    same_speaker = (
                        current.get("speaker", "") == next_seg.get("speaker", "")
                    )

                    if same_speaker and gap <= max_gap:
                        # Merge segments
                        current["text"] = f"{current['text']} {next_seg['text']}"
                        current["end"] = next_seg["end"]

                        logger.debug(
                            f"[Normalization] Merged fragments: "
                            f"{current.get('segment_id', '')} + {next_seg.get('segment_id', '')}"
                        )

                        i += 2  # Skip next segment as it's merged
                        merged.append(current)
                        continue

            merged.append(current)
            i += 1

        return merged

    def _normalize_speaker_names(self, segments: List[Dict]) -> List[Dict]:
        """
        Normalize speaker names (trim, capitalize, handle variations).
        """
        # Build speaker mapping (variations → canonical)
        speaker_variations = {}

        for segment in segments:
            speaker = segment.get("speaker", "").strip()
            if not speaker:
                continue

            # Normalize to title case
            canonical = speaker.title()

            # Track variations
            if canonical not in speaker_variations:
                speaker_variations[canonical] = []
            if speaker not in speaker_variations[canonical]:
                speaker_variations[canonical].append(speaker)

        # Apply normalization
        normalized = []
        for segment in segments:
            seg_copy = segment.copy()
            speaker = seg_copy.get("speaker", "").strip()

            if speaker:
                # Find canonical form
                canonical = speaker.title()
                seg_copy["speaker"] = canonical

            normalized.append(seg_copy)

        return normalized

    def _cleanup_text(self, segments: List[Dict], options: Dict) -> List[Dict]:
        """
        Clean up text: remove extra whitespace, fix common issues.
        """
        remove_filler_words = options.get("remove_filler_words", False)

        filler_patterns = [
            r'\b(um|uh|ah|er|like|you know|actually|basically)\b',
        ] if remove_filler_words else []

        cleaned = []

        for segment in segments:
            seg_copy = segment.copy()
            text = seg_copy.get("text", "")

            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # Remove filler words if enabled
            for pattern in filler_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                text = re.sub(r'\s+', ' ', text).strip()

            seg_copy["text"] = text

            # Only keep non-empty segments
            if text:
                cleaned.append(seg_copy)

        return cleaned

    def _reorder_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Ensure segments are in chronological order by start time.
        """
        return sorted(segments, key=lambda s: s.get("start", 0))

    def _validate_timestamps(self, segments: List[Dict]) -> List[Dict]:
        """
        Validate timestamp integrity (start < end, no negative times).
        """
        valid = []

        for segment in segments:
            start = segment.get("start", 0)
            end = segment.get("end", 0)

            # Skip invalid timestamps
            if start < 0 or end < 0 or start >= end:
                logger.warning(
                    f"[Normalization] Invalid timestamps in segment {segment.get('segment_id', '')}: "
                    f"start={start}, end={end}"
                )
                continue

            valid.append(segment)

        return valid

    @staticmethod
    def _ends_with_punctuation(text: str) -> bool:
        """Check if text ends with sentence-ending punctuation."""
        return bool(re.search(r'[.!?]\s*$', text.strip()))
