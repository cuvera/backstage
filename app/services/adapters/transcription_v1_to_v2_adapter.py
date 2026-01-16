# =============================
# FILE: app/services/adapters/transcription_v1_to_v2_adapter.py
# PURPOSE:
#   Adapter to transform V1 transcription format to V2 expected format.
#   Handles field renaming, type conversions, and timestamp transformations.
#
# HOW TO USE:
#   from app.services.adapters.transcription_v1_to_v2_adapter import TranscriptionV1ToV2Adapter
#
#   adapter = TranscriptionV1ToV2Adapter()
#   v2_format = adapter.transform(v1_transcription)
# =============================

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TranscriptionV1ToV2Adapter:
    """
    Adapter to transform V1 transcription format to V2 expected format.

    V1 Format (from TranscriptionV1Repository):
    {
        "tenant_id": "...",
        "transcriptions": [
            {
                "segment_id": 1,                    # Integer
                "start": "00:25",                   # MM:SS string
                "end": "00:32",                     # MM:SS string
                "transcription": "text...",         # Key name
                "sentiment": "neutral",
                "speaker": "John Doe"
            }
        ],
        "speakers": [...],
        "metadata": {...},
        ...
    }

    V2 Format (expected by normalization processor):
    {
        "transcriptions": [
            {
                "segment_id": "seg_1",              # String
                "start": 25.0,                      # Float (seconds)
                "end": 32.0,                        # Float (seconds)
                "text": "text...",                  # Renamed key
                "sentiment": "neutral",
                "speaker": "John Doe"
            }
        ]
    }
    """

    def transform(self, v1_transcription: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform V1 transcription to V2 format.

        Args:
            v1_transcription: V1 transcription dict from repository

        Returns:
            V2 formatted dict ready for normalization processor
        """
        v1_segments = v1_transcription.get("transcriptions", [])

        logger.info(
            f"[V1→V2 Adapter] Transforming {len(v1_segments)} segments from V1 to V2 format"
        )

        v2_segments = []
        for segment in v1_segments:
            try:
                v2_segment = self._transform_segment(segment)
                v2_segments.append(v2_segment)
            except Exception as e:
                logger.warning(
                    f"[V1→V2 Adapter] Failed to transform segment {segment.get('segment_id')}: {e}"
                )
                # Skip problematic segments
                continue

        logger.info(
            f"[V1→V2 Adapter] Successfully transformed {len(v2_segments)}/{len(v1_segments)} segments"
        )

        return {
            "transcriptions": v2_segments
        }

    def _transform_segment(self, v1_segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single segment from V1 to V2 format.

        Args:
            v1_segment: V1 segment dict

        Returns:
            V2 formatted segment dict
        """
        # Convert segment_id to string
        segment_id = v1_segment.get("segment_id")
        if not isinstance(segment_id, int):
            segment_id =  int(segment_id)
        # else:
        #     segment_id_str = str(segment_id) if segment_id else "seg_unknown"

        # Convert timestamps from MM:SS to float seconds
        start_str = v1_segment.get("start", "00:00")
        end_str = v1_segment.get("end", "00:00")

        start_seconds = self._parse_timestamp(start_str)
        end_seconds = self._parse_timestamp(end_str)

        # Debug: log segment 373 to investigate parsing issue
        if segment_id == 373:
            logger.info(
                f"[V1→V2 Adapter] DEBUG SEGMENT 373: "
                f"start_str='{start_str}' → {start_seconds}s, "
                f"end_str='{end_str}' → {end_seconds}s, "
                f"raw_segment={v1_segment}"
            )

        # Debug invalid timestamps
        if start_seconds > end_seconds:
            logger.warning(
                f"[V1→V2 Adapter] Invalid timestamp order in segment {segment_id}: "
                f"start_str='{start_str}' ({start_seconds}s) > end_str='{end_str}' ({end_seconds}s)"
            )

        # Build V2 segment
        v2_segment = {
            "segment_id": segment_id,
            "start": start_seconds,
            "end": end_seconds,
            "text": v1_segment.get("transcription", ""),  # Rename transcription → text
            "speaker": v1_segment.get("speaker", "Unknown"),
        }

        # Optional fields
        if "sentiment" in v1_segment:
            v2_segment["sentiment"] = v1_segment["sentiment"]

        return v2_segment

    def _parse_timestamp(self, timestamp_str: str) -> float:
        """
        Parse timestamp from MM:SS or HH:MM:SS format to float seconds.

        Supported formats:
        - "00:25" → 25.0
        - "11:23" → 683.0
        - "01:30:45" → 5445.0

        Args:
            timestamp_str: Timestamp string in MM:SS or HH:MM:SS format

        Returns:
            Timestamp in seconds as float

        Raises:
            ValueError: If timestamp format is invalid
        """
        if not timestamp_str:
            return 0.0

        # Handle numeric strings (already in seconds)
        if timestamp_str.replace(".", "").isdigit():
            return float(timestamp_str)

        # Parse MM:SS or HH:MM:SS format
        parts = timestamp_str.strip().split(":")

        if len(parts) == 2:
            # MM:SS format
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)

        elif len(parts) == 3:
            # HH:MM:SS format
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

        else:
            logger.warning(
                f"[V1→V2 Adapter] Invalid timestamp format: {timestamp_str}, defaulting to 0.0"
            )
            return 0.0


# Singleton instance
transcription_v1_to_v2_adapter = TranscriptionV1ToV2Adapter()
