import logging
import os
import tempfile
from typing import Dict, Any, Optional
import time
from datetime import datetime
import shutil
from pathlib import Path
import asyncio

from app.services.transcription_service import TranscriptionService
from app.utils.audio_merger import find_files_from_s3, list_files_in_s3_folder
from app.repository import MeetingMetadataRepository
from app.repository.transcription_v1_repository import TranscriptionV1Repository
from app.repository.transcription_v2_repository import TranscriptionV2Repository
from app.utils.s3_client import download_s3_file
from app.core.config import settings
from app.services.transcription_v2_service import transcription_v2_service
from app.services.adapters.transcription_v1_to_v2_adapter import transcription_v1_to_v2_adapter

logger = logging.getLogger(__name__)

class MeetingAnalysisOrchestratorError(Exception):
    """Custom exception for meeting analysis orchestrator operations."""
    pass


class MeetingAlreadyProcessedException(Exception):
    """Exception raised when meeting is already processed or being processed."""
    pass



class MeetingAnalysisOrchestrator:
    """Main service for orchestrating meeting processing pipeline."""

    def __init__(self):
        self.meeting_metadata_repo = None  # Will be initialized when needed
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
    
    async def analyze_meeting(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process meeting event using Gemini 2.5 Flash for unified analysis.
        
        Args:
            payload: Dictionary containing the meeting event with payload
            
        Returns:
            Dictionary containing processing results
        """
        if not payload:
            logger.error("No payload found in event_data")
            return
                        
        # Extract meeting details from payload
        meeting_id = str(payload.get('_id'))
        tenant_id = payload.get('tenantId')
        platform = payload.get('platform')
        recurring_meeting_id = payload.get('recurring_meeting_id')

        if not all([meeting_id, tenant_id, platform]):
            logger.error(f"Missing required fields in payload: meeting_id={meeting_id}, tenant_id={tenant_id}, platform={platform}")
            return

        logger.info(f"Starting meeting analysis - meeting_id={meeting_id}, tenant_id={tenant_id}")

        # Save original platform for status updates (before it might be switched for transcription)
        original_platform = platform

        next_meeting = None
        file_url = None
        temp_directory = None
        overall_start_time = time.time()

        try:
            ###########################################
            # Step 0: Prepare meeting info and context
            ###########################################
            step_start_time = time.time()
            logger.info(f"Step 0: Preparing meeting context - meeting_id={meeting_id}")

            if platform == "google":
                if not self.meeting_metadata_repo:
                    self.meeting_metadata_repo = await MeetingMetadataRepository.from_default()
                
                meeting_metadata = await self.meeting_metadata_repo.get_meeting_metadata(meeting_id)

                logger.debug(f"Meeting metadata retrieved - meeting_id={meeting_id}")

                if meeting_metadata and meeting_metadata.get("recurring_meeting_id"):
                    recurring_meeting_id = meeting_metadata.get("recurring_meeting_id")

                    # fetch immediate next meeting from recurring_meeting_id
                    next_meeting = await self.meeting_metadata_repo.find_immediate_next_meeting(
                        current_meeting_metadata=meeting_metadata,
                        recurring_meeting_id=recurring_meeting_id,
                        platform=platform
                    )

                    if next_meeting:
                        logger.info(f"Next meeting identified for prep pack - next_meeting_id={next_meeting}")
            else:
                meeting_metadata = {}
                logger.debug(f"Using empty meeting metadata for non-Google platform - meeting_id={meeting_id}")

            step_duration_ms = round((time.time() - step_start_time) * 1000, 2)
            logger.info(f"Step 0 completed in {step_duration_ms}ms - meeting_id={meeting_id}")

            ######################################################
            # Step 1: Transcription with TranscriptionService v1 #
            ######################################################
            step_start_time = time.time()
            logger.info(f"Step 1: Starting transcription process - meeting_id={meeting_id}")

            # Check if transcription v1 already exists
            transcription_v1_repo = await TranscriptionV1Repository.from_default()
            transcription = await transcription_v1_repo.get_transcription(meeting_id, tenant_id)

            if not transcription:
                logger.info(f"Step 1.1: No existing transcription found, preparing audio file - meeting_id={meeting_id}")
                
                # if payload.get("fileUrl"):
                    # file_url = payload.get("fileUrl")
                    # temp_directory = "tmp_data"
                # else:
                prepare_audio_file = await self._prepare_audio_file(payload)
                file_url = prepare_audio_file.get("local_merged_file_path")
                temp_directory = prepare_audio_file.get("temp_directory")
                
                if not file_url:
                    raise MeetingAnalysisOrchestratorError("Failed to prepare audio file")

                # Check if we need to switch from google to offline mode
                if platform == "google":
                    speaker_timeframes = meeting_metadata.get("speaker_timeframes", [])
                    if not speaker_timeframes:
                        logger.warning(
                            f"No speaker timeframes found for Google meeting, "
                            f"switching to offline mode - meeting_id={meeting_id}"
                        )
                        platform = "offline"
                        logger.info(f"Platform switched to offline for transcription - meeting_id={meeting_id}")

                # Step 1.2: Transcription with new TranscriptionService (with orchestrator-level retry)
                logger.info(f"Step 1.2: Starting audio transcription with Gemini (platform={platform}) - meeting_id={meeting_id}")
                transcription_service = TranscriptionService(max_concurrent=5)

                # Orchestrator-level retry (2 attempts total with 5s wait)
                max_orchestrator_retries = 2
                orchestrator_retry_delay = 5
                transcription_result = None
                last_transcription_error = None

                for orchestrator_attempt in range(max_orchestrator_retries):
                    try:
                        if orchestrator_attempt > 0:
                            logger.info(f"Retrying transcription (orchestrator-level attempt {orchestrator_attempt + 1}/{max_orchestrator_retries}) after {orchestrator_retry_delay}s delay - meeting_id={meeting_id}")
                            await asyncio.sleep(orchestrator_retry_delay)

                        transcription_result = await transcription_service.transcribe_meeting(
                            audio_file_path=file_url,
                            meeting_metadata=meeting_metadata,
                            meeting_id=meeting_id,
                            tenant_id=tenant_id,
                            platform=platform,  # 'google' or 'offline'
                            chunk_duration_minutes=5.0 if platform == 'offline' else 10.0,
                            overlap_seconds=5.0,
                            output_dir=temp_directory,
                            enable_incremental_saving=True
                        )

                        # Success, break out of retry loop
                        logger.info(f"Transcription successful{' (after orchestrator retry)' if orchestrator_attempt > 0 else ''} - meeting_id={meeting_id}")
                        break

                    except Exception as e:
                        last_transcription_error = e
                        logger.error(f"Transcription failed (orchestrator-level attempt {orchestrator_attempt + 1}/{max_orchestrator_retries}) - meeting_id={meeting_id}: {e}")

                        # If this was the last attempt, re-raise the error
                        if orchestrator_attempt == max_orchestrator_retries - 1:
                            logger.error(f"Transcription failed after {max_orchestrator_retries} orchestrator-level attempts - meeting_id={meeting_id}")
                            raise

                # Step 1.3: Save transcription v1 to database
                logger.info(f"Step 1.3: Saving transcription v1 to database - meeting_id={meeting_id}")
                save_result = await transcription_v1_repo.save_transcription(
                    meeting_id=meeting_id,
                    tenant_id=tenant_id,
                    transcription_result=transcription_result
                )
                logger.info(
                    f"Saved transcription v1: {save_result['total_segments']} segments, "
                    f"{save_result['total_speakers']} speakers - meeting_id={meeting_id}"
                )

                # Get the saved transcription for downstream processing
                transcription = await transcription_v1_repo.get_transcription(meeting_id, tenant_id)

            else:
                logger.info(f"Step 1: Using existing transcription v1 - meeting_id={meeting_id}")

            step_duration_ms = round((time.time() - step_start_time) * 1000, 2)
            logger.info(f"Step 1 completed in {step_duration_ms}ms - meeting_id={meeting_id}")

            # Validate transcription has segments before proceeding to analysis
            transcript_segments = transcription.get('transcriptions', [])
            if not transcript_segments or len(transcript_segments) == 0:
                error_msg = "Transcription completed but produced no segments. Cannot proceed with analysis."
                logger.error(f"{error_msg} - meeting_id={meeting_id}")
                raise MeetingAnalysisOrchestratorError(error_msg)

            logger.info(f"Transcription validation passed: {len(transcript_segments)} segments found - meeting_id={meeting_id}")

            logger.debug("Waiting for 1 seconds before analysis...")
            await asyncio.sleep(1);

            ########################################################
            # Step 1.2: Transcription with TranscriptionService v2 #
            ########################################################
            logger.info(f"[Orchestrator] Step 1.2: Processing transcription V2 for meeting {meeting_id}")

            # Initialize V2 repository
            transcription_v2_repo = await TranscriptionV2Repository.from_default()

            # Check if V2 already exists in database
            existing_v2_doc = await transcription_v2_repo.get_by_meeting_id(
                meeting_id=meeting_id,
                tenant_id=tenant_id
            )

            if existing_v2_doc:
                logger.info(f"[Orchestrator] Transcription V2 already exists in database, skipping processing")
                # Use existing V2 from database
                transcription_v2 = {
                    "segments": [segment.model_dump() for segment in existing_v2_doc.segments],
                    "metadata": existing_v2_doc.metadata.model_dump()
                }
                v2 = {"transcription_v2": transcription_v2}
            else:
                logger.info(f"[Orchestrator] Transcription V2 not found, processing from V1")
                # Transform V1 to V2 format
                v2_transcription_input = transcription_v1_to_v2_adapter.transform(transcription)

                # Process V2 transcription (normalization + classification)
                v2 = await transcription_v2_service.process(
                    v1_transcription=v2_transcription_input,
                    id=meeting_id,
                    tenant_id=tenant_id,
                )

                # Save transcription V2 to database
                transcription_v2 = v2.get("transcription_v2")
                if transcription_v2:
                    logger.info(f"[Orchestrator] Saving transcription V2 to database")
                    await transcription_v2_repo.save_transcription(
                        meeting_id=meeting_id,
                        tenant_id=tenant_id,
                        transcription_v2=transcription_v2
                    )
                    logger.info(f"[Orchestrator] Transcription V2 saved successfully")
                
            total_duration_ms = round((time.time() - overall_start_time) * 1000, 2)
            logger.info(f"Meeting analysis completed successfully in {total_duration_ms}ms - meeting_id={meeting_id}, tenant_id={tenant_id}")
            return True

        except Exception as e:
            logger.error(f"Meeting processing failed - meeting_id={meeting_id}, tenant_id={tenant_id}: {str(e)}", exc_info=True)            
            raise MeetingAnalysisOrchestratorError(f"Meeting processing failed: {str(e)}") from e
        finally:
            cleanup_start_time = time.time()
            cleaned_files = []
            cleaned_directories = []
            
            try:
                if file_url and os.path.exists(file_url):
                    os.remove(file_url)
                    cleaned_files.append(file_url)
                    logger.debug(f"Cleaned up temporary file: {file_url}")

                if temp_directory and os.path.exists(temp_directory):
                    shutil.rmtree(temp_directory)
                    cleaned_directories.append(temp_directory)
                    logger.debug(f"Cleaned up temporary directory: {temp_directory}")
                
                if cleaned_files or cleaned_directories:
                    logger.info(f"Temporary resources cleaned up - meeting_id={meeting_id}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary resources - meeting_id={meeting_id}: {str(e)}", exc_info=True)

    async def _convert_to_m4a(self, input_file_path: str, temp_dir: str) -> str:
        """
        Convert any audio file to M4A format using FFmpeg.

        Args:
            input_file_path: Path to the audio file (any format)
            temp_dir: Temporary directory for the converted file

        Returns:
            Path to the converted M4A file

        Raises:
            MeetingAnalysisOrchestratorError: If conversion fails
        """
        input_ext = Path(input_file_path).suffix.lower()
        output_file_path = str(Path(temp_dir) / "merged_output.m4a")

        # Log conversion start
        logger.info(f"Converting {input_ext} to M4A: {input_file_path}")

        # FFmpeg command - matches audio_merger.py pattern (line 132)
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', input_file_path,
            '-c:a', 'aac',          # AAC audio codec
            '-b:a', '128k',         # 128k bitrate (matches existing pattern)
            '-vn',                  # No video (audio only)
            '-y',                   # Overwrite output file
            output_file_path
        ]

        # Execute conversion - async pattern from audio_merger.py lines 112-125
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        # Check for errors
        if process.returncode != 0:
            error_output = stderr.decode(errors='replace')
            raise MeetingAnalysisOrchestratorError(
                f"Audio to M4A conversion failed: {error_output}"
            )

        # Validate converted file exists and has content
        if not os.path.exists(output_file_path):
            raise MeetingAnalysisOrchestratorError(
                f"Converted file not found: {output_file_path}"
            )

        converted_size = os.path.getsize(output_file_path)
        if converted_size == 0:
            raise MeetingAnalysisOrchestratorError(
                "Converted file is empty"
            )

        # Log success with file sizes
        original_size_mb = os.path.getsize(input_file_path) / (1024 ** 2)
        converted_size_mb = converted_size / (1024 ** 2)
        logger.info(
            f"Audio conversion completed ({input_ext} → M4A) - "
            f"original: {original_size_mb:.2f}MB, "
            f"converted: {converted_size_mb:.2f}MB"
        )

        # Clean up original file
        await asyncio.to_thread(os.remove, input_file_path)
        logger.debug(f"Removed original file: {input_file_path}")

        return output_file_path

    async def _prepare_audio_file(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare audio file for processing."""
        bucket = payload.get('bucket', settings.MEETING_BUCKET_NAME)

        # Use configured temp directory instead of system temp
        temp_base_dir = Path(settings.TEMP_AUDIO_DIR)
        temp_base_dir.mkdir(parents=True, exist_ok=True)

        # Check disk space before creating temp directory
        stat = shutil.disk_usage(temp_base_dir)
        free_gb = stat.free / (1024 ** 3)

        if free_gb < settings.MIN_FREE_DISK_SPACE_GB:
            raise MeetingAnalysisOrchestratorError(
                f"Insufficient disk space: {free_gb:.2f}GB free, "
                f"minimum required: {settings.MIN_FREE_DISK_SPACE_GB}GB"
            )

        # Create meeting-specific temp directory
        meeting_id = payload.get('_id')
        temp_dir_name = f"audio_merge_{meeting_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_dir = temp_base_dir / temp_dir_name
        temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created temp directory: {temp_dir} ({free_gb:.2f}GB free)")
        file_url = None

        if not payload.get('fileUrl'):
            s3_folder_path = f"{payload.get('tenantId')}/{payload.get('platform')}/{payload.get('_id')}/"
            output_s3_key = f"{payload.get('tenantId')}/{payload.get('platform')}/{payload.get('_id')}/meeting.m4a"

            # Get list of audio files - check for both .wav and .m4a (pre-merged file)
            all_audio_files = await list_files_in_s3_folder(
                s3_folder_path=s3_folder_path,
                bucket_name=bucket,
                file_extension=[".webm", ".m4a", ".mp3", ".mp4"]  # Check multiple formats
            )

            if all_audio_files and len(all_audio_files) == 1:
                s3_file_path = all_audio_files[0]
                file_url = str(temp_dir / s3_file_path)
                await download_s3_file(s3_file_path, file_url, bucket)
            else:
                merge_result = await find_files_from_s3(
                    s3_folder_path=s3_folder_path,
                    output_s3_key=output_s3_key,
                    bucket_name=bucket,
                    temp_dir=str(temp_dir)
                )
                file_url = merge_result['local_merged_file_path']
        else:
            # Preserve original file extension from S3 key
            s3_file_key = payload.get('fileUrl')
            original_extension = Path(s3_file_key).suffix or '.m4a'
            file_url = str(temp_dir / f"merged_output{original_extension}")
            await download_s3_file(s3_file_key, file_url, bucket)

        # Convert to M4A if not already M4A
        file_ext = Path(file_url).suffix.lower()
        if file_ext != '.m4a':
            logger.info(f"Non-M4A format detected ({file_ext}), converting to M4A - meeting_id={meeting_id}")
            conversion_start_time = time.time()

            file_url = await self._convert_to_m4a(file_url, str(temp_dir))

            conversion_duration_ms = round((time.time() - conversion_start_time) * 1000, 2)
            logger.info(f"Audio conversion completed in {conversion_duration_ms}ms - meeting_id={meeting_id}")

        # Validate file size
        if os.path.exists(file_url):
            file_size_mb = os.path.getsize(file_url) / (1024 ** 2)
            logger.info(f"Audio file size: {file_size_mb:.2f}MB")

            if file_size_mb > settings.MAX_AUDIO_FILE_SIZE_MB:
                logger.warning(
                    f"Audio file exceeds maximum size: {file_size_mb:.2f}MB > "
                    f"{settings.MAX_AUDIO_FILE_SIZE_MB}MB"
                )

        return {
            'local_merged_file_path': file_url,
            'temp_directory': str(temp_dir)
        }

    def _is_valid_iso_date(self, date_str: Optional[str]) -> bool:
        """
        Check if a date string is in valid ISO 8601 format.

        Args:
            date_str: Date string to validate

        Returns:
            True if valid ISO format, False otherwise
        """
        if not date_str or not isinstance(date_str, str):
            return False

        try:
            from datetime import datetime
            # Try parsing ISO format (YYYY-MM-DD or full datetime with T)
            if 'T' in date_str:
                datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except (ValueError, AttributeError):
            return False
