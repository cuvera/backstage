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

        for cluster_def in cluster_definitions:
            cluster = self._build_single_cluster(cluster_def, segment_map)
            if cluster:
                clusters.append(cluster)

        logger.info(f"[ClusterBuilder] Built {len(clusters)} clusters successfully")

        return {
            "clusters": clusters,
            "metadata": {
                "total_clusters": len(clusters),
                "total_segments": len(segments),
                "clusters_by_type": self._count_by_type(clusters)
            }
        }

    def _build_single_cluster(
        self,
        cluster_def: Dict[str, Any],
        segment_map: Dict[str, Dict]
    ) -> Optional[Dict[str, Any]]:
        """
        Build a single cluster with full segment details.

        Args:
            cluster_def: Cluster definition from LLM
            segment_map: Map of segment_id to segment data

        Returns:
            Cluster with full details or None if invalid
        """
        segment_ids = cluster_def.get("segment_ids", [])
        topic = cluster_def.get("topic", "").strip()
        cluster_type = cluster_def.get("type", "").strip()

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

        # Build cluster structure
        cluster = {
            "topic": topic,
            "type": cluster_type,
            "segment_ids": segment_ids,
            "segments": segments_data,
            "metadata": {
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "segment_count": len(segments_data),
                "speakers": speakers
            }
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
