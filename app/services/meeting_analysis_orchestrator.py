import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any
from unittest import result
import time


from app.schemas.meeting_analysis import MeetingAnalysis
from app.services.transcription_service import TranscriptionService
from app.utils.audio_merger import merge_wav_files_from_s3, AudioMergerError
from app.services.meeting_analysis_service import MeetingAnalysisService
from app.repository import MeetingMetadataRepository
from app.utils.s3_client import download_s3_file
from app.core.config import settings
from app.services.agents import TranscriptionAgent, CallAnalysisAgent
from app.services.meeting_prep_curator_service import MeetingPrepCuratorService

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
        next_meeting = None

        if not all([meeting_id, tenant_id, platform]):
            logger.error("Missing required fields in payload: _id, tenantId, platform")
            return

        try:
            ###########################################
            # Step 0: Prepare meeting info and context
            ###########################################
            logger.info("Step 0: Preparing meeting context")

            if platform == "google":
                if not self.meeting_metadata_repo:
                    self.meeting_metadata_repo = await MeetingMetadataRepository.from_default()
                
                await self._update_meeting_status(meeting_id, platform, 'analysing')
                meeting_metadata = await self.meeting_metadata_repo.get_meeting_metadata(meeting_id)

                print(f"Meeting metadata: {meeting_metadata}")

                if meeting_metadata and meeting_metadata.get("recurring_meeting_id"):
                    recurring_meeting_id = meeting_metadata.get("recurring_meeting_id")

                    # fetch immediate next meeting from recurring_meeting_id
                    next_meeting = await self.meeting_metadata_repo.find_immediate_next_meeting(
                        current_meeting_metadata=meeting_metadata,
                        recurring_meeting_id=recurring_meeting_id,
                        platform=platform
                    )

                    if next_meeting:
                        await self._update_meeting_status(next_meeting, platform, 'analysing')
            else:
                meeting_metadata = {}

            #################################################
            # Step 1: Prepare audio file
            #################################################
            logger.info("Step 1: Transcriptions")
            transcription_service = TranscriptionService()
            transcription = await transcription_service.get_transcription(meeting_id, tenant_id)

            if not transcription:
                logger.info("Step 1.1: Preparing audio file")
                prepare_audio_file = await self._prepare_audio_file(payload)

                file_url = prepare_audio_file.get("local_merged_file_path")
                if not file_url:
                    raise MeetingAnalysisOrchestratorError("Failed to prepare audio file")

                # Step 1.2: Transcription with TranscriptionAgent
                logger.info("Step 1.2: Transcribing audio with TranscriptionAgent")
                transcription = await transcription_service.save_transcription(
                    audio_file_path=file_url,
                    meeting_id=meeting_id,
                    tenant_id=tenant_id,
                    meeting_metadata=meeting_metadata,
                )

            else:
                logger.info("Step 1: Transcription exists, Skipping audio file preparation and transcription");

            logger.info("Waiting for 2 seconds before analysis...")
            time.sleep(2)

            #################################################
            # Step 2: Meeting Analysis with CallAnalysisAgent
            #################################################
            logger.info("Step 2: Meeting Analysis with CallAnalysisAgent")

            analysis_service = await MeetingAnalysisService.from_default()
            analysis = await analysis_service.get_analysis(tenant_id=tenant_id, session_id=meeting_id)

            if not analysis:
                call_analysis_agent = CallAnalysisAgent()

                # duration in seconds, last turn end time - first turn start time
                if transcription["conversation"] and len(transcription["conversation"]) > 0:
                    duration = transcription["conversation"][-1]["end_time"] - transcription["conversation"][0]["start_time"]
                else:
                    duration = 0

                analysis = await call_analysis_agent.analyze(
                        transcript_payload={
                            **transcription,
                            "duration_sec": duration
                        },
                        context={
                            "tenant_id": tenant_id,
                            "session_id": meeting_id,
                            "platform": platform
                    }
                )

                logger.info("Step 2.1: Saving meeting analysis")
                await analysis_service.save_analysis(analysis)
            
            else:
                logger.info("Step 2: Meeting analysis exists, Skipping analysis")

            logger.info("Waiting for 2 seconds before meeting preparation...")
            time.sleep(2)

            ############################################################
            # Step 3: Meeting Preparation with MeetingPrepCuratorService
            ############################################################
            logger.info("Step 3: Meeting prep suggestion")
            if recurring_meeting_id and next_meeting:
                prep_curator_service = await MeetingPrepCuratorService.from_default()
                await prep_curator_service.generate_and_save_prep_pack(
                    next_meeting_id=next_meeting,
                    meeting_analysis=analysis,
                    meeting_metadata=meeting_metadata,
                    platform=platform,
                    recurring_meeting_id=recurring_meeting_id,
                    previous_meeting_counts=2,
                    context={
                        "current_meeting_id": meeting_id,
                        "tenant_id": tenant_id
                    }
                )

                # Update next meeting status to scheduled
                if platform == "google":
                    await self._update_meeting_status(next_meeting, platform, 'scheduled')             
            else:
                logger.info("Skipping meeting preparation - no recurring_meeting_id provided")
            
            result = {}
            # Update meeting status to completed on success
            if platform == "google":
                await self._update_meeting_status(meeting_id, platform, 'completed')
            return result

        except Exception as e:
            logger.error(f"Meeting processing failed for meeting {payload.get('_id', 'unknown')}: {str(e)}")
            
            if meeting_id and platform:
                await self._update_meeting_status(str(meeting_id), platform, 'failed')

                if next_meeting:
                    await self._update_meeting_status(next_meeting, platform, 'scheduled')
            
            raise MeetingAnalysisOrchestratorError(f"Meeting processing failed: {str(e)}") from e

    async def _update_meeting_status(self, meeting_id: str, platform: str, status: str) -> None:
        """Update meeting status in metadata repository for supported platforms."""
        try:
            if platform == "google":
                if not self.meeting_metadata_repo:
                    self.meeting_metadata_repo = await MeetingMetadataRepository.from_default()
                
                await self.meeting_metadata_repo.update_meeting_status(meeting_id, platform, status)
                logger.info(f"Updated meeting {meeting_id} status to {status} for platform {platform}")
            else:
                logger.info(f"Status update not supported for platform {platform}, meeting {meeting_id} would be {status}")
        except Exception as e:
            logger.warning(f"Failed to update meeting status to {status} for meeting {meeting_id}: {e}")

    async def _prepare_audio_file(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare audio file for processing."""
        bucket = payload.get('bucket', settings.MEETING_BUCKET_NAME)

        if not payload.get('fileUrl'):
            s3_folder_path = f"{payload.get('tenantId')}/{payload.get('platform')}/{payload.get('_id')}/"
            output_s3_key = f"{payload.get('tenantId')}/{payload.get('platform')}/{payload.get('_id')}/meeting.wav"

            merge_result = await merge_wav_files_from_s3(
                s3_folder_path=s3_folder_path,
                output_s3_key=output_s3_key,
                bucket_name=bucket
            )
            file_url = merge_result['local_merged_file_path']
        else:
            temp_dir = tempfile.mkdtemp(prefix="audio_merge_")
            file_url = os.path.join(temp_dir, "merged_output.wav")
            await download_s3_file(payload.get('fileUrl'), file_url, bucket)
            merge_result = {
                'local_merged_file_path': file_url,
                'temp_directory': temp_dir
            }

        return merge_result
