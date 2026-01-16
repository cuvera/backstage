# =============================
# FILE: app/services/processors/cluster_builder.py
# PURPOSE:
#   Builds final cluster structure from classification output.
#   Fetches segment details, calculates metadata, maintains chronological order.
# =============================

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ClusterBuilder:
    """
    Builds final segment classification structure from LLM cluster definitions.
    """

    @staticmethod
    def _seconds_to_mmss(seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def build_clusters(
        self,
        cluster_definitions: List[Dict[str, Any]],
        normalized_transcription: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build final cluster structure with segment details and metadata.

        Args:
            cluster_definitions: List of clusters from classification processor
            normalized_transcription: Normalized transcription with segment details

        Returns:
            Final segment_classifications structure
        """
        segments = normalized_transcription.get("transcriptions", [])

        # Build segment lookup map
        segment_map = {
            seg.get("segment_id"): seg
            for seg in segments
        }

        logger.info(
            f"[ClusterBuilder] Building {len(cluster_definitions)} clusters "
            f"from {len(segments)} segments"
        )

        # Build clusters with full details
        clusters = []

        for idx, cluster_def in enumerate(cluster_definitions, 1):
            cluster = self._build_single_cluster(cluster_def, segment_map, idx)
            if cluster:
                clusters.append(cluster)

        logger.info(f"[ClusterBuilder] Built {len(clusters)} clusters successfully")

        return {
            "segments": clusters,
            "metadata": {
                "total_clusters": len(clusters),
                "total_segments": len(segments),
                "clusters_by_type": self._count_by_type(clusters)
            }
        }

    def _build_single_cluster(
        self,
        cluster_def: Dict[str, Any],
        segment_map: Dict[str, Dict],
        cluster_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Build a single cluster with full segment details.

        Args:
            cluster_def: Cluster definition from LLM
            segment_map: Map of segment_id to segment data
            cluster_id: Sequential cluster ID

        Returns:
            Cluster with full details or None if invalid
        """
        # Handle segment_ids - should be list
        segment_ids_raw = cluster_def.get("segment_ids", [])
        if isinstance(segment_ids_raw, str):
            # LLM might return single string instead of list
            segment_ids = [segment_ids_raw]
            logger.debug(f"[ClusterBuilder] Converted segment_ids string to list: {segment_ids_raw}")
        elif isinstance(segment_ids_raw, list):
            segment_ids = segment_ids_raw
        else:
            logger.warning(f"[ClusterBuilder] Invalid segment_ids type: {type(segment_ids_raw)}")
            segment_ids = []

        # Handle topic - should be list
        topic_raw = cluster_def.get("topic", [])
        if isinstance(topic_raw, list):
            topic = topic_raw
        else:
            # Convert string to list
            topic = [str(topic_raw).strip()] if topic_raw else []
            logger.debug(f"[ClusterBuilder] Converted topic string to list: {topic_raw} → {topic}")

        # Handle type - could be string or list
        type_raw = cluster_def.get("type", "")
        if isinstance(type_raw, list):
            cluster_type = type_raw[0] if type_raw else ""
            logger.debug(f"[ClusterBuilder] Extracted type from list: {type_raw} → {cluster_type}")
        else:
            cluster_type = str(type_raw).strip() if type_raw else ""

        if not segment_ids or not topic or not cluster_type:
            logger.warning(
                f"[ClusterBuilder] Skipping invalid cluster definition: {cluster_def}"
            )
            return None

        # Fetch segment details
        segments_data = []
        for seg_id in segment_ids:
            if seg_id in segment_map:
                segments_data.append(segment_map[seg_id])
            else:
                logger.warning(
                    f"[ClusterBuilder] Segment {seg_id} not found in normalized transcription"
                )

        if not segments_data:
            logger.warning(
                f"[ClusterBuilder] No valid segments found for cluster: {topic}"
            )
            return None

        # Calculate cluster metadata
        start_time = min(seg.get("start", 0) for seg in segments_data)
        end_time = max(seg.get("end", 0) for seg in segments_data)
        duration = end_time - start_time

        # Extract speakers
        speakers = list(set(
            seg.get("speaker", "Unknown")
            for seg in segments_data
        ))

        # Format transcriptions with MM:SS times
        transcriptions = []
        for seg in segments_data:
            seg_copy = seg.copy()
            seg_copy["start"] = self._seconds_to_mmss(seg.get("start", 0))
            seg_copy["end"] = self._seconds_to_mmss(seg.get("end", 0))
            transcriptions.append(seg_copy)

        # Build cluster structure (flattened with segment_cluster_id)
        cluster = {
            "segment_cluster_id": cluster_id,
            "topic": topic,
            "type": cluster_type,
            "start_time": self._seconds_to_mmss(start_time),
            "end_time": self._seconds_to_mmss(end_time),
            "duration": self._seconds_to_mmss(duration),
            "speakers": speakers,
            "segment_count": len(segments_data),
            "transcriptions": transcriptions
        }

        return cluster

    def _count_by_type(self, clusters: List[Dict]) -> Dict[str, int]:
        """
        Count clusters by type for metadata.
        """
        counts = {}

        for cluster in clusters:
            cluster_type = cluster.get("type", "unknown")
            counts[cluster_type] = counts.get(cluster_type, 0) + 1

        return counts
