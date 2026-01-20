import logging
import os
import tempfile
from typing import Dict, Any, Optional
import time
from datetime import datetime
import shutil
from pathlib import Path
import asyncio

from app.schemas.meeting_analysis import MeetingAnalysis
from app.services.transcription_service import TranscriptionService
from app.utils.audio_merger import find_files_from_s3, list_files_in_s3_folder
from app.services.meeting_analysis_service import MeetingAnalysisService
from app.repository import MeetingMetadataRepository
from app.repository.transcription_v1_repository import TranscriptionV1Repository
from app.repository.transcription_v2_repository import TranscriptionV2Repository
from app.utils.s3_client import download_s3_file
from app.core.config import settings
from app.services.agents import TranscriptionAgent, CallAnalysisAgent
from app.services.meeting_prep_curator_service import MeetingPrepCuratorService
from app.messaging.producers.meeting_status_producer import send_meeting_status
from app.messaging.producers.email_notification_producer import send_email_notification
from app.messaging.producers.meeting_embedding_ready_producer import send_meeting_embedding_ready
from app.messaging.producers.task_commands_producer import send_task_creation_command
from app.services.transcription_v2_service import transcription_v2_service
from app.services.adapters.transcription_v1_to_v2_adapter import transcription_v1_to_v2_adapter
from app.services.agents.call_analysis.coordinator import CallAnalysisCoordinator
from app.utils.auth_service_client import AuthServiceClient

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

            await self._update_meeting_status(meeting_id, original_platform, 'analysing', tenant_id)

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
                        await self._update_meeting_status(next_meeting, original_platform, 'analysing', tenant_id)
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
                v2 = await transcription_v2_service.process_and_publish(
                    v1_transcription=v2_transcription_input,
                    meeting_id=meeting_id,
                    tenant_id=tenant_id,
                    platform=original_platform
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

            #################################################
            # Step 2: Meeting Analysis with CallAnalysisAgent
            #################################################
            step_start_time = time.time()
            logger.info(f"Step 2: Starting meeting analysis - meeting_id={meeting_id}")

            analysis_service = await MeetingAnalysisService.from_default()
            analysis_doc = await analysis_service.get_analysis(tenant_id=tenant_id, session_id=meeting_id)

            if not analysis_doc:
                # call_analysis_agent = CallAnalysisAgent()

                # analysis = await call_analysis_agent.analyze(
                #         transcript_payload=transcription,
                #         context={
                #             "tenant_id": tenant_id,
                #             "session_id": meeting_id,
                #             "platform": platform,
                #             "meeting_title": meeting_metadata.get("summary"),
                #             "start_time": meeting_metadata.get("start_time"),
                #             "end_time": meeting_metadata.get("end_time")
                #     }
                # )

                transcription_v2 = v2.get("transcription_v2");
                analysis_coordinator = CallAnalysisCoordinator()
                
                logger.info(f"Step 2.1: Calling analysis coordinator (internal retry enabled) - meeting_id={meeting_id}")
                
                analysis_data = await analysis_coordinator.analyze_meeting(
                    meeting_id=meeting_id,
                    tenant_id=tenant_id,
                    v2_transcript=transcription_v2,  # The V2 payload
                    metadata={
                    "title": meeting_metadata.get("summary"),
                    "platform": platform,
                    "participants": meeting_metadata.get("attendees")
                    }
                )
                analysis = MeetingAnalysis(**analysis_data)

                logger.info(f"Step 2.1: Saving meeting analysis - meeting_id={meeting_id}")
                await analysis_service.save_analysis(analysis)
            else:
                analysis = MeetingAnalysis(**analysis_doc)
                logger.info(f"Step 2: Using existing analysis - meeting_id={meeting_id}")
            
            step_duration_ms = round((time.time() - step_start_time) * 1000, 2)
            logger.info(f"Step 2 completed in {step_duration_ms}ms - meeting_id={meeting_id}")

            # Generate tasks from action items
            # Determine platform type: offline if platform is "offline", otherwise online
            platform_type = "offline" if original_platform == "offline" else "online"
            await self._generate_tasks_from_action_items(
                meeting_id=meeting_id,
                tenant_id=tenant_id,
                analysis=analysis,
                meeting_metadata=meeting_metadata,
                platform=platform_type
            )

            logger.debug("Waiting for 1 seconds before meeting preparation...")
            await asyncio.sleep(1)

            ############################################################
            # Step 3: Meeting Preparation with MeetingPrepCuratorService
            ############################################################
            step_start_time = time.time()
            logger.info(f"Step 3: Starting meeting preparation - meeting_id={meeting_id}")
            if recurring_meeting_id and next_meeting:
                prep_curator_service = await MeetingPrepCuratorService.from_default()
                await prep_curator_service.generate_and_save_prep_pack(
                    next_meeting_id=next_meeting,
                    meeting_analysis=analysis,
                    meeting_metadata=meeting_metadata,
                    platform=original_platform,
                    recurring_meeting_id=recurring_meeting_id,
                    previous_meeting_counts=2,
                    context={
                        "current_meeting_id": meeting_id,
                        "tenant_id": tenant_id
                    }
                )

                await self._update_meeting_status(next_meeting, original_platform, 'scheduled', tenant_id)
                logger.info(f"Meeting prep pack generated successfully - next_meeting_id={next_meeting}")
            else:
                logger.info(f"Skipping meeting preparation - missing requirements - meeting_id={meeting_id}")
            
            step_duration_ms = round((time.time() - step_start_time) * 1000, 2)
            logger.info(f"Step 3 completed in {step_duration_ms}ms - meeting_id={meeting_id}")

            # Update meeting status to completed on success
            await self._update_meeting_status(meeting_id, original_platform, 'completed', tenant_id)
            
            # Step 4: Send email notification
            await self._send_meeting_completion_email(
                meeting_id=meeting_id,
                tenant_id=tenant_id,
                analysis=analysis,
                meeting_metadata=meeting_metadata,
                transcription=transcription
            )

            # Step 5: Send meeting embedding ready event
            send_meeting_embedding_ready(
                meeting_id=meeting_id,
                tenant_id=tenant_id,
                platform=original_platform
            )

            total_duration_ms = round((time.time() - overall_start_time) * 1000, 2)
            logger.info(f"Meeting analysis completed successfully in {total_duration_ms}ms - meeting_id={meeting_id}, tenant_id={tenant_id}")
            return True

        except Exception as e:
            logger.error(f"Meeting processing failed - meeting_id={meeting_id}, tenant_id={tenant_id}: {str(e)}", exc_info=True)

            if meeting_id and original_platform:
                await self._update_meeting_status(str(meeting_id), original_platform, 'failed', tenant_id)

                if next_meeting:
                    await self._update_meeting_status(next_meeting, original_platform, 'scheduled', tenant_id)
            
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

    async def _update_meeting_status(self, meeting_id: str, platform: str, status: str, tenant_id: str = None) -> None:
        """Update meeting status in metadata repository for supported platforms."""
        try:
            if platform == "google":
                if not self.meeting_metadata_repo:
                    self.meeting_metadata_repo = await MeetingMetadataRepository.from_default()

                await self.meeting_metadata_repo.update_meeting_status(meeting_id, platform, status)
                logger.info(f"Updated meeting {meeting_id} status to {status} for platform {platform}")
            elif platform == "offline":
                # Map internal status to RabbitMQ status
                send_meeting_status(
                    meeting_id=meeting_id,
                    status=status,
                    platform=platform,
                    tenant_id=tenant_id,
                    session_id=meeting_id
                )
                logger.info(f"Sent offline meeting {meeting_id} status {meeting_id}")
            else:
                logger.info(f"Status update not supported for platform {platform}, meeting {meeting_id} would be {status}")
        except Exception as e:
            logger.warning(f"Failed to update meeting status to {status} for meeting {meeting_id}: {e}")

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

    async def _generate_tasks_from_action_items(
        self,
        meeting_id: str,
        tenant_id: str,
        analysis: MeetingAnalysis,
        meeting_metadata: Dict[str, Any],
        platform: str
    ) -> None:
        """
        Generate tasks from meeting action items and publish to task management system.

        Args:
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier
            analysis: Meeting analysis containing action items
            meeting_metadata: Meeting metadata from repository
            platform: Platform type ("online" or "offline")
        """
        try:
            # Extract action items
            action_items = analysis.action_items
            if not action_items or len(action_items) == 0:
                logger.info(f"No action items found, skipping task generation - meeting_id={meeting_id}")
                return

            logger.info(f"Starting task generation for {len(action_items)} action items - meeting_id={meeting_id}")

            # Get organizer information (optional - used as fallback for tasks without owners)
            auth_client = AuthServiceClient()
            organizer_user = None
            organizer_email = meeting_metadata.get('organizer')

            if organizer_email:
                try:
                    organizer_users = await auth_client.search_users(organizer_email, tenant_id=tenant_id, limit=1)
                    if organizer_users and len(organizer_users) > 0:
                        organizer_user = organizer_users[0]
                        logger.info(f"Found organizer user: {organizer_user.get('name')} - meeting_id={meeting_id}")
                    else:
                        logger.warning(f"Organizer not found in auth service: {organizer_email} - meeting_id={meeting_id}")
                except Exception as e:
                    logger.warning(f"Failed to fetch organizer from auth service - meeting_id={meeting_id}: {e}")
            else:
                logger.warning(f"No organizer found in meeting metadata - meeting_id={meeting_id}. Tasks without valid owners will use meeting_id as placeholder assignee.")

            # Process each action item
            tasks = []
            for idx, action_item in enumerate(action_items):
                try:
                    assigned_user = None

                    # Try to find user by owner name if provided
                    if action_item.owner and action_item.owner.strip():
                        try:
                            owner_users = await auth_client.search_users(action_item.owner, tenant_id=tenant_id, limit=1)
                            if owner_users and len(owner_users) > 0:
                                assigned_user = owner_users[0]
                                logger.info(f"Matched owner '{action_item.owner}' to user {assigned_user.get('name')} - meeting_id={meeting_id}")
                            else:
                                logger.info(f"No user found for owner '{action_item.owner}', using organizer - meeting_id={meeting_id}")
                        except Exception as e:
                            logger.warning(f"Failed to search user '{action_item.owner}' - meeting_id={meeting_id}: {e}")

                    # Fallback to organizer if no user found
                    if not assigned_user:
                        assigned_user = organizer_user

                    # If still no assignee, create placeholder assignee with meeting_id
                    if not assigned_user:
                        meeting_title = meeting_metadata.get('summary', 'Meeting')
                        assignee = {
                            "userId": meeting_id,  # Use meeting_id as placeholder
                            "employeeId": None,
                            "name": f"Unassigned ({meeting_title})",
                            "email": None,
                            "department": None,
                            "designation": None
                        }
                        logger.warning(
                            f"No valid assignee found for action item '{action_item.task[:50]}...' "
                            f"(owner: '{action_item.owner}', organizer: {organizer_email or 'None'}), "
                            f"using meeting_id as placeholder assignee - meeting_id={meeting_id}"
                        )
                    else:
                        # Build assignee object with all required fields from found user
                        assignee = {
                            "userId": assigned_user.get("id"),
                            "employeeId": assigned_user.get("employeeId"),
                            "name": assigned_user.get("name"),
                            "email": assigned_user.get("email"),
                            "department": assigned_user.get("department"),
                            "designation": assigned_user.get("designation")
                        }

                    # Validate and set priority (default to "medium" for invalid values)
                    valid_priorities = ["low", "medium", "high", "urgent"]
                    priority = "medium"  # default
                    if action_item.priority:
                        priority_str = str(action_item.priority).lower().strip()
                        if priority_str in valid_priorities:
                            priority = priority_str
                        else:
                            logger.warning(f"Invalid priority '{action_item.priority}' for action item, using default 'medium' - meeting_id={meeting_id}")

                    # Build task object
                    task = {
                        "title": action_item.task[:200] if len(action_item.task) > 200 else action_item.task,
                        "description": action_item.task,
                        "assignees": [assignee],
                        "priority": priority,
                        "tags": []
                    }

                    # Add due date only if valid ISO format
                    if self._is_valid_iso_date(action_item.due_date):
                        task["dueDate"] = action_item.due_date

                    tasks.append(task)
                    logger.debug(f"Created task {idx + 1}/{len(action_items)} - meeting_id={meeting_id}")

                except Exception as e:
                    logger.error(f"Failed to process action item {idx + 1} - meeting_id={meeting_id}: {e}", exc_info=True)
                    continue

            # Publish tasks if any were successfully created
            if tasks:
                meeting_title = meeting_metadata.get('summary', 'Meeting')
                meeting_date = meeting_metadata.get('start_time').isoformat() if meeting_metadata.get('start_time') else datetime.now().isoformat()

                # Count placeholder assignees
                placeholder_count = sum(1 for task in tasks if task.get("assignees") and task["assignees"][0].get("userId") == meeting_id)

                send_task_creation_command(
                    meeting_id=meeting_id,
                    tenant_id=tenant_id,
                    meeting_title=meeting_title,
                    meeting_date=meeting_date,
                    tasks=tasks,
                    platform=platform
                )

                if placeholder_count > 0:
                    logger.info(
                        f"Successfully published {len(tasks)} tasks to task management system "
                        f"({placeholder_count} with placeholder assignees awaiting assignment) - meeting_id={meeting_id}"
                    )
                else:
                    logger.info(f"Successfully published {len(tasks)} tasks to task management system - meeting_id={meeting_id}")
            else:
                logger.warning(f"No tasks were successfully created from {len(action_items)} action items - meeting_id={meeting_id}")

        except Exception as e:
            logger.error(f"Task generation failed - meeting_id={meeting_id}: {str(e)}", exc_info=True)
            # Don't raise - task generation failure should not fail the entire meeting analysis

    async def _send_meeting_completion_email(
        self,
        meeting_id: str,
        tenant_id: str,
        analysis: MeetingAnalysis,
        meeting_metadata: Dict[str, Any],
        transcription: Dict[str, Any]
    ) -> None:
        """
        Send email notification to meeting participants after analysis completion.
        
        Args:
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier
            analysis: Meeting analysis results
            meeting_metadata: Meeting metadata from repository
            transcription: Meeting transcription data
        """
        try:
            # Collect all email addresses from meeting metadata and transcript participants
            all_emails = set()
            
            # Add emails from meeting metadata
            if meeting_metadata.get('attendees'):
                all_emails.update(meeting_metadata.get('attendees', []))
            
            # Add organizer email
            if meeting_metadata.get('organizer'):
                all_emails.add(meeting_metadata.get('organizer'))
                        
            # Parse exclude list from config and remove excluded emails
            exclude_emails = set()
            if settings.EMAIL_EXCLUDE_LIST:
                exclude_emails = {email.strip() for email in settings.EMAIL_EXCLUDE_LIST.split(',') if email.strip()}
                all_emails = all_emails - exclude_emails
                logger.info(f"Excluded {len(exclude_emails)} emails from notification: {exclude_emails}")
            
            # Fetch user details from auth service with fallback
            user_mapping = []
            try:
                auth_client = AuthServiceClient()
                user_details = await auth_client.fetch_users_by_emails(list(all_emails), tenant_id=tenant_id)
                user_mapping = auth_client.create_user_email_mapping(user_details)
                logger.info(f"Successfully fetched user details for {len(user_mapping)} users")
            except Exception as e:
                logger.warning(f"Auth service call failed, using fallback names - meeting_id={meeting_id}: {str(e)}")
                user_mapping = []
            
            # Transform to attendees list with proper fallback
            attendees_list = []
            organizer_email = meeting_metadata.get('organizer', list(all_emails)[0] if all_emails else 'info@cuvera.ai')
            organizer = None
            
            for email in all_emails:
                # Find user details from auth service response
                user_data = next((user for user in user_mapping if user['email'] == email), None)
                
                attendee = {
                    "name": user_data['name'] if user_data and user_data.get('name') else "Participant",
                    "email": email
                }
                attendees_list.append(attendee)
                
                # Set organizer if this is the organizer email
                if email == organizer_email:
                    organizer = attendee.copy()
            
            # Fallback organizer if not found
            if not organizer and attendees_list:
                organizer = attendees_list[0]
            
            # Calculate duration string
            # Calculate duration string from HH:MM:SS format
            duration_str = "0 Minutes"
            if analysis.duration:
                try:
                    parts = analysis.duration.split(":")
                    if len(parts) == 3:
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        duration_str = f"{hours} Hour {minutes} Minutes" if hours > 0 else f"{minutes} Minutes"
                    elif len(parts) == 2:
                        minutes = int(parts[0])
                        duration_str = f"{minutes} Minutes"
                except (ValueError, IndexError):
                    pass
            
            # Send email notification
            send_email_notification(
                attendees=attendees_list,
                organizer=organizer,
                title=meeting_metadata.get('summary', 'Meeting Analysis Complete'),
                startTime=meeting_metadata.get('start_time').isoformat() if meeting_metadata.get('start_time') else '',
                endTime=meeting_metadata.get('end_time').isoformat() if meeting_metadata.get('end_time') else '',
                duration=duration_str,
                summary=analysis.summary,
                redirectUrl=f"{settings.EMAIL_REDIRECT_BASE_URL}/meeting/online/{meeting_id}",
                noOfKeyPoints=len(analysis.key_points),
                noOfActionItems=len(analysis.action_items),
                tenant_id=tenant_id
            )
            
            logger.info(f"Email notification sent to {len(attendees_list)} attendees - meeting_id={meeting_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification - meeting_id={meeting_id}: {str(e)}")
            # Don't fail the entire process if email fails
