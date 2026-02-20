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

import asyncio
import logging
from typing import Any, Dict, Optional

from app.services.processors.normalization_processor import NormalizationProcessor
from app.services.processors.classification_processor import SegmentClassificationProcessor
from app.services.processors.cluster_builder import ClusterBuilder
from app.messaging.producers.transcription_producer import publish_v2

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
        v1_transcription: Dict[str, Any],
        id: str,
        tenant_id: str,
        platform: str,
        mode: str,
        type: Optional[str] = "",
        options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process V1 transcription through V2 pipeline and publish to RabbitMQ.

        Args:
            v1_transcription: V1 transcription output with 'transcriptions' list
            id: Unique transcription identifier
            tenant_id: Tenant identifier
            platform: Platform identifier (google, zoom, etc)
            mode: Mode type ("online" or "offline")
            type: Optional type field
            options: Processing options for normalization, classification, etc.

        Returns:
            Processing result summary with processing stats

        Raises:
            Exception: If processing or publishing fails
        """
        options = options or {}

        logger.info(f"V2 pipeline starting | id={id} tenant={tenant_id}")

        try:
            # Step 1: Normalization
            normalized_transcription = await asyncio.to_thread(
                self.normalization_processor.normalize,
                v1_transcription,
                options.get("normalization", {})
            )
            normalization_stats = normalized_transcription.get("metadata", {})
            logger.info(f"Normalized | {normalization_stats.get('normalized_segment_count', 0)} segments | id={id}")

            # Step 2: Classification (LLM)
            cluster_definitions = await self.classification_processor.classify_segments(
                normalized_transcription,
                options.get("classification", {})
            )
            logger.info(f"Classified | {len(cluster_definitions)} clusters | id={id}")

            # Step 3: Build Clusters
            segment_classifications = self.cluster_builder.build_clusters(
                cluster_definitions,
                normalized_transcription
            )
            cluster_stats = segment_classifications.get("metadata", {})
            logger.info(f"Clusters built | {cluster_stats.get('total_clusters', 0)} clusters | id={id}")

            # Step 4: Publish to RabbitMQ
            await publish_v2(
                meeting_id=id,
                tenant_id=tenant_id,
                platform=platform,
                mode=mode,
                transcription_v2=segment_classifications
            )
            logger.info(f"V2 published | id={id}")

            return {
                "status": "completed",
                "transcription_v2": segment_classifications,
                "segments": segment_classifications,
                "processing_stats": {
                    "normalization": normalization_stats,
                    "classification": {
                        "total_clusters": cluster_stats.get("total_clusters", 0),
                        "clusters_by_type": cluster_stats.get("clusters_by_type", {})
                    }
                }
            }

        except Exception as e:
            logger.exception(f"V2 failed | id={id}: {e}")
            raise

# Singleton instance
transcription_v2_service = TranscriptionV2Service()
