import logging
import os
import tempfile
from typing import Dict, Any, Optional
import time
from datetime import datetime
import shutil
from pathlib import Path
import asyncio

from app.core.config import settings
from app.services.transcription_v1_service import transcription_v1_service
from app.services.transcription_v2_service import transcription_v2_service
from app.utils.adapters.transcription_v1_to_v2_adapter import transcription_v1_to_v2_adapter 
from app.utils.audio_downloader import download_audio
from app.utils.audio_chunker import get_base_dir

from app.messaging.producers.transcription_producer import publish_status_failure

logger = logging.getLogger(__name__)

class AudioTranscriptionError(Exception):
    """Custom exception for audio transcription operations."""
    pass


class TranscriptionAlreadyProcessedException(Exception):
    """Exception raised when transcription is already processed or being processed."""
    pass



class TranscriptionOrchestrator:
    """Main service for orchestrating audio transcription pipeline."""

    def __init__(self):
        self._ensure_temp_directory()

    def _ensure_temp_directory(self) -> None:
        """Ensure temp directory exists and has sufficient space."""
        temp_dir = Path(settings.TEMP_AUDIO_DIR)

        # Create directory if it doesn't exist
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Check available disk space
        stat = shutil.disk_usage(temp_dir)
        free_gb = stat.free / (1024 ** 3)

        if free_gb < settings.MIN_FREE_DISK_SPACE_GB:
            logger.warning(
                f"Low disk space in temp directory: {free_gb:.2f}GB free, "
                f"minimum required: {settings.MIN_FREE_DISK_SPACE_GB}GB"
            )

        logger.info(f"Temp directory initialized: {temp_dir} ({free_gb:.2f}GB free)")
    
    async def transcribe_audio(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process audio transcription using Gemini 2.5 Flash for unified analysis.

        Args:
            payload: Dictionary containing the audio payload

        Returns:
            Dictionary containing processing results
        """
        if not payload:
            logger.error("No payload found in event_data")
            return
                        
        # Extract meeting details from payload
        transcription_id = str(payload.get('id'))
        tenant_id = payload.get('tenantId')
        platform = payload.get('platform')
        mode = payload.get("mode")
        speaker_timeframes = payload.get("speakerTimeframes", [])

        if not all([transcription_id, tenant_id]):
            logger.error(f"Missing required fields in payload: transcription_id={transcription_id}, tenant_id={tenant_id}")
            return

        logger.info(f"Starting audio transcription - transcription_id={transcription_id}, tenant_id={tenant_id}")

        overall_start_time = time.time()

        # Ensure RabbitMQ producer is connected
        from app.messaging.producer import producer
        await producer.connect()

        try:
            ######################################################
            # Step 1: Transcription V1 (Download + Process)     #
            ######################################################
            step_start_time = time.time()
            logger.info(f"Step 1: Starting transcription V1 - transcription_id={transcription_id}")

            # Step 1.1: Download audio
            audio_url = payload.get("audioUrl")
            if not audio_url:
                raise AudioTranscriptionError("No audioUrl provided in payload")

            download_result = await download_audio(audio_url, output_dir=get_base_dir(transcription_id))
            local_audio_path = download_result["local_path"]
            logger.info(
                f"Step 1.1: Audio downloaded | path={local_audio_path} "
                f"size={download_result['file_size_bytes']} bytes"
            )

            # Step 1.2: Process through V1 pipeline (chunk → transcribe → merge → publish)
            participants = payload.get("participants", [])

            merged_result = await transcription_v1_service.process(
                audio_file_path=local_audio_path,
                transcription_id=transcription_id,
                tenant_id=tenant_id,
                platform=platform,
                mode=mode,
                speaker_timeframes=speaker_timeframes,
                participants=participants
            )

            step_duration_ms = round((time.time() - step_start_time) * 1000, 2)
            logger.info(f"Step 1 completed in {step_duration_ms}ms - transcription_id={transcription_id}")

            ########################################################
            # Step 2: Transcription V2 (Transform + Process + Publish) #
            ########################################################
            step_start_time = time.time()
            logger.info(f"Step 2: Starting transcription V2 - transcription_id={transcription_id}")

            # Build transcription dict for V2 processing
            transcription = {
                "transcriptions": [seg.model_dump() for seg in merged_result.transcriptions],
                "speakers": [sp.model_dump() for sp in merged_result.speakers],
                "metadata": merged_result.metadata.model_dump()
            }

            # Transform V1 to V2 format
            v2_transcription_input = transcription_v1_to_v2_adapter.transform(transcription)

            # Process V2 pipeline (normalize → classify → publish)
            await transcription_v2_service.process(
                v1_transcription=v2_transcription_input,
                id=transcription_id,
                tenant_id=tenant_id,
                platform=platform,
                mode=mode
            )

            step_duration_ms = round((time.time() - step_start_time) * 1000, 2)
            logger.info(f"Step 2 completed in {step_duration_ms}ms - transcription_id={transcription_id}")

            total_duration_ms = round((time.time() - overall_start_time) * 1000, 2)
            logger.info(f"Audio transcription completed successfully in {total_duration_ms}ms - transcription_id={transcription_id}, tenant_id={tenant_id}")
            return True

        except Exception as e:
            logger.error(f"Audio transcription failed - transcription_id={transcription_id}, tenant_id={tenant_id}: {str(e)}", exc_info=True)

            # Publish failure status to RabbitMQ
            try:
                # Use mode if already set, otherwise determine from payload
                failure_mode = mode if mode else ("online" if speaker_timeframes else "offline")

                await publish_status_failure(
                    meeting_id=transcription_id,
                    tenant_id=tenant_id,
                    platform=platform,
                    mode=failure_mode
                )
                logger.info(f"Published failure status to RabbitMQ - transcription_id={transcription_id}")
            except Exception as publish_error:
                logger.error(f"Failed to publish failure status: {publish_error}")

            raise AudioTranscriptionError(f"Audio transcription failed: {str(e)}") from e
