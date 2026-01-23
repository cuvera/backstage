# =============================
# FILE: app/services/transcription_v2_service.py
# PURPOSE:
#   Main Transcription V2 service orchestrating the enhancement pipeline.
#   Takes V1 transcription output and produces enhanced V2 output with:
#   - Normalized transcription (duplicate removal, overlap resolution, etc.)
#   - Segment classification (topics and types)
#
#   V2 is stateless - does not store in database, only publishes to RabbitMQ.
#
# HOW TO USE:
#   1) Instantiate TranscriptionV2Service
#   2) Call process_and_publish() with V1 transcription data
#   3) Service will normalize, classify, and publish to RabbitMQ
# =============================

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.services.processors.normalization_processor import NormalizationProcessor
from app.services.processors.classification_processor import SegmentClassificationProcessor
from app.services.processors.cluster_builder import ClusterBuilder
from app.messaging.producers.transcription_v2_producer import send_transcription_v2_ready

logger = logging.getLogger(__name__)


class TranscriptionV2Service:
    """
    Transcription V2 enhancement service.

    Pipeline:
    1. Normalization (algorithmic) - remove duplicates, resolve overlaps, merge fragments
    2. Classification (LLM) - identify topics and segment types
    3. Build Clusters - construct final structure with metadata
    4. Publish to RabbitMQ - send to downstream consumers

    This service is stateless and does not persist to database.
    """

    def __init__(self):
        self.normalization_processor = NormalizationProcessor()
        self.classification_processor = SegmentClassificationProcessor()
        self.cluster_builder = ClusterBuilder()

    async def process(
        self,
        id: str,
        tenant_id: str,
        v1_transcription: Dict[str, Any],
        type: Optional[str] = "",
        options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process V1 transcription through V2 pipeline and publish to RabbitMQ.

        Args:
            v1_transcription: V1 transcription output with 'transcriptions' list
            id: Unique audio identifier
            tenant_id: Tenant identifier
            platform: Platform type (default: "offline")
            options: Processing options for normalization, classification, etc.

        Returns:
            Processing result summary with message_id and stats

        Raises:
            Exception: If processing or publishing fails
        """
        options = options or {}

        logger.info(
            f"[TranscriptionV2] Starting V2 processing | audio_id={id} tenant={tenant_id} type={type}"
        )

        try:
            # Step 1: Normalization (algorithmic)
            logger.info(f"[TranscriptionV2] Step 1: Normalization")
            normalized_transcription = self.normalization_processor.normalize(
                v1_transcription,
                options.get("normalization", {})
            )

            normalization_stats = normalized_transcription.get("metadata", {})
            logger.info(
                f"[TranscriptionV2] Normalization complete: "
                f"{normalization_stats.get('normalized_segment_count', 0)} segments"
            )

            # Step 2: Classification (LLM)
            logger.info(f"[TranscriptionV2] Step 2: Segment Classification")
            cluster_definitions = await self.classification_processor.classify_segments(
                normalized_transcription,
                options.get("classification", {})
            )

            logger.info(
                f"[TranscriptionV2] Classification complete: "
                f"{len(cluster_definitions)} clusters"
            )

            # Step 3: Build Clusters
            logger.info(f"[TranscriptionV2] Step 3: Building Clusters")
            segment_classifications = self.cluster_builder.build_clusters(
                cluster_definitions,
                normalized_transcription
            )

            cluster_stats = segment_classifications.get("metadata", {})
            logger.info(
                f"[TranscriptionV2] Cluster building complete: "
                f"{cluster_stats.get('total_clusters', 0)} clusters"
            )

            # Step 4: Publish to RabbitMQ
            logger.info(f"[TranscriptionV2] Step 4: Publishing to RabbitMQ")

            processing_stats = {
                "normalization": normalization_stats,
                "classification": {
                    "total_clusters": cluster_stats.get("total_clusters", 0),
                    "clusters_by_type": cluster_stats.get("clusters_by_type", {})
                }
            }

            return {
                "status": "completed",
                "transcription_v2": segment_classifications,
                "segments": segment_classifications,
                # "message_id": message_id,
                "processing_stats": processing_stats
            }

        except Exception as e:
            logger.exception(
                f"[TranscriptionV2] V2 processing failed | meeting={meeting_id}: {e}"
            )

# Singleton instance
transcription_v2_service = TranscriptionV2Service()
