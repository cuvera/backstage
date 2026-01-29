"""
Transcription V1 Service
Complete pipeline for V1 transcription: chunking → transcribing → merging → publishing
"""

import logging
from typing import List, Dict, Any

from app.utils.audio_chunker import create_audio_chunks, get_chunk_output_dir
from app.services.chunk_transcriber import ChunkTranscriber
from app.utils.chunk_merger import ChunkMerger, MergedTranscriptionResult
from app.messaging.producers.transcription_producer import publish_v1

logger = logging.getLogger(__name__)


class TranscriptionV1Service:
    """
    Service for complete V1 transcription pipeline

    Responsibilities:
    - Create audio chunks optimized for Gemini model
    - Transcribe chunks in parallel
    - Merge chunk results into single transcript
    - Publish V1 transcription to RabbitMQ
    """

    def __init__(self, max_concurrent: int = 5):
        """
        Initialize TranscriptionV1Service

        Args:
            max_concurrent: Maximum concurrent transcription calls
        """
        self.chunk_transcriber = ChunkTranscriber(max_concurrent=max_concurrent)
        self.chunk_merger = ChunkMerger()
        logger.info(f"[TranscriptionV1] Initialized with max {max_concurrent} concurrent")

    async def process(
        self,
        audio_file_path: str,
        transcription_id: str,
        tenant_id: str,
        platform: str,
        mode: str,
        speaker_timeframes: List[Dict] = None,
        participants: List[Dict] = None
    ) -> MergedTranscriptionResult:
        """
        Process audio file through complete V1 transcription pipeline

        Args:
            audio_file_path: Path to downloaded audio file
            transcription_id: Unique identifier for this transcription
            tenant_id: Tenant identifier
            platform: Platform identifier (google, zoom, etc)
            mode: "online" (with speaker timeframes) or "offline"
            speaker_timeframes: Optional speaker timeframe data
            participants: Optional participant data

        Returns:
            MergedTranscriptionResult with transcriptions, speakers, metadata
        """
        logger.info(
            f"[TranscriptionV1] Starting V1 pipeline - transcription_id={transcription_id}, "
            f"mode={mode}"
        )

        speaker_timeframes = speaker_timeframes or []
        participants = participants or []

        # Step 1: Create audio chunks
        logger.info(f"[TranscriptionV1] Step 1: Creating audio chunks")
        chunks = create_audio_chunks(
            audio_file_path=audio_file_path,
            speaker_timeframes=speaker_timeframes,
            chunk_duration_minutes=10.0,
            overlap_seconds=5.0,
            output_dir=get_chunk_output_dir(transcription_id),
            output_format='.mp3'
        )
        logger.info(f"[TranscriptionV1] Created {len(chunks)} audio chunks")

        # Step 2: Transcribe chunks in parallel
        logger.info(f"[TranscriptionV1] Step 2: Transcribing {len(chunks)} chunks")
        chunk_results = await self.chunk_transcriber.transcribe_chunks(
            chunks=chunks,
            participants=participants,
            transcription_id=transcription_id,
            tenant_id=tenant_id,
            enable_incremental_saving=True
        )
        logger.info(f"[TranscriptionV1] Transcribed {len(chunk_results)} chunks")

        # Step 3: Merge chunk results
        logger.info(f"[TranscriptionV1] Step 3: Merging transcription results")
        merged_result = await self.chunk_merger.merge_transcriptions(
            chunk_results=chunk_results,
            speaker_timeframes=speaker_timeframes,
            participants=participants
        )
        logger.info(
            f"[TranscriptionV1] Merged {len(merged_result.transcriptions)} segments "
            f"with {len(merged_result.speakers)} speakers"
        )

        # Validate transcription has segments before publishing
        if not merged_result.transcriptions or len(merged_result.transcriptions) == 0:
            error_msg = "Transcription completed but produced no segments"
            logger.error(f"[TranscriptionV1] {error_msg} - transcription_id={transcription_id}")
            raise ValueError(error_msg)

        # Step 4: Publish V1 to RabbitMQ
        logger.info(f"[TranscriptionV1] Step 4: Publishing V1 to RabbitMQ")
        await publish_v1(
            meeting_id=transcription_id,
            tenant_id=tenant_id,
            platform=platform,
            mode=mode,
            transcriptions=[seg.model_dump() for seg in merged_result.transcriptions],
            speakers=[sp.model_dump() for sp in merged_result.speakers],
            metadata=merged_result.metadata.model_dump()
        )
        logger.info(f"[TranscriptionV1] Published V1 transcription to RabbitMQ")

        logger.info(f"[TranscriptionV1] V1 pipeline completed - transcription_id={transcription_id}")
        return merged_result


# Singleton instance
transcription_v1_service = TranscriptionV1Service()
