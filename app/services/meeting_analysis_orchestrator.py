import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any
from unittest import result

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
        try:
            logger.info("Starting meeting event processing with Gemini")
            if not payload:
                raise MeetingAnalysisOrchestratorError("No payload found in event_data")
            
            logger.info(f"Processing meeting: {payload}")
            
            # Extract meeting details from payload
            meeting_id = str(payload.get('_id'))
            tenant_id = payload.get('tenantId')
            platform = payload.get('platform')
            recurring_meeting_id = payload.get('recurring_meeting_id')
            
            if not all([meeting_id, tenant_id]):
                raise MeetingAnalysisOrchestratorError("Missing required fields in payload: _id, tenantId")
            
            # Update meeting status to analyzing at start of processing
            if platform == "google":
                await self._update_meeting_status(meeting_id, platform, 'analysing')
            
            # Step 1: Prepare audio file
            logger.info("Step 1: Preparing audio file")
            prepare_audio_file = await self._prepare_audio_file(payload)
            
            file_url = prepare_audio_file.get("local_merged_file_path")
            if not file_url:
                raise MeetingAnalysisOrchestratorError("Failed to prepare audio file")

            # Step 2: Prepare meeting info and context
            logger.info("Step 2: Preparing meeting context")
            meeting_info = {
                "meeting_id": meeting_id,
                "platform": platform,
                "speakerTimeframes": []
            }

            next_meeting = None
            speaker_timeframes = []
            
            # Get speaker timeframes for Google meetings
            if platform == "google":
                if not self.meeting_metadata_repo:
                    self.meeting_metadata_repo = await MeetingMetadataRepository.from_default()
                
                meeting_metadata = await self.meeting_metadata_repo.get_meeting_metadata(meeting_id)
                # print the data from meeting_metadata

                if meeting_metadata:
                    speaker_timeframes = meeting_metadata.get("speaker_timeframes", [])
                    meeting_info["speakerTimeframes"] = speaker_timeframes
                    recurring_meeting_id = meeting_metadata.get('recurring_meeting_id', None)
                    meeting_info['recurring_meeting_id'] = recurring_meeting_id
                    logger.info(f"Found {len(speaker_timeframes)} speaker timeframes for Google meeting")
                    meeting_info['attendees'] = meeting_metadata.get('attendees', [])
                    logger.info(f"Found {len(speaker_timeframes)} speaker timeframes for Google meeting")

                    if recurring_meeting_id:
                        # fetch immediate next meeting from recurring_meeting_id
                        next_meeting = await self.meeting_metadata_repo.find_immediate_next_meeting(
                            current_meeting_metadata=meeting_metadata,
                            recurring_meeting_id=recurring_meeting_id,
                            platform=platform
                        )

                        if next_meeting:
                            print("next meeting found: ", next_meeting)
                            # update meeting status
                            await self._update_meeting_status(next_meeting, platform, 'analysing')
            
            # Step 3: Transcription with TranscriptionAgent
            logger.info("Step 3: Transcribing audio with TranscriptionAgent")
            # transcription_agent = TranscriptionAgent()
            transcription_service = TranscriptionService()
            transcription_result = await transcription_service.save_transcription(
                audio_file_path=file_url,
                meeting_id=meeting_id,
                tenant_id=tenant_id,
                meeting_metadata=meeting_info,
            )

            # Transcription saved result
            logger.info(f"Transcription saved: {transcription_result}")

            # Step 4: Meeting Analysis with CallAnalysisAgent  
            logger.info("Step 4: Analyzing meeting with CallAnalysisAgent")
            
            # Transform transcription for CallAnalysisAgent
            transcript_payload = {
                "tenant_id": tenant_id,
                "session_id": meeting_id,
                "conversation": transcription_result["conversation"],
                "participants": [],  # Will be derived from conversation
                "language": "en-US",
                "duration_sec": sum((turn.get("end_time", 0) - turn.get("start_time", 0)) for turn in transcription_result["conversation"])
            }
            
            # Wait for 5 sec
            import time
            print("Waiting for 2 seconds before analysis...")
            time.sleep(2)

            call_analysis_agent = CallAnalysisAgent()
            meeting_analysis = await call_analysis_agent.analyze(
                transcript_payload=transcript_payload,
                context={
                    "tenant_id": tenant_id,
                    "session_id": meeting_id,
                    "platform": platform
                }
            )
            
            # Step 5: Save Meeting Analysis
            logger.info("Step 5: Saving meeting analysis")
            analysis_service = await MeetingAnalysisService.from_default()
            analysis_save_result = await analysis_service.save_analysis(meeting_analysis)            

            # Step 7: Meeting Preparation with MeetingPrepCuratorService
            logger.info("Step 7: Processing meeting preparation")
            prep_pack = None
            prep_save_result = None
            
            print("recurring_meeting_id: ", recurring_meeting_id)
            print("next_meeting: ", next_meeting)

            if recurring_meeting_id and next_meeting:
                print("Waiting for 2 seconds before meeting preparation...")
                time.sleep(2)

                try:
                    prep_curator_service = await MeetingPrepCuratorService.from_default()
                    prep_result = await prep_curator_service.generate_and_save_prep_pack(
                        meeting_id=next_meeting,
                        meeting_analysis=meeting_analysis,
                        platform=platform,
                        recurring_meeting_id=recurring_meeting_id,
                        previous_meeting_counts=2,
                        context={
                            "current_meeting_id": meeting_id,
                            "tenant_id": tenant_id
                        }
                    )
                    
                    prep_pack = prep_result.get("prep_pack")
                    prep_save_result = prep_result.get("save_result")
                    
                    # Update next meeting status to scheduled
                    if platform == "google":
                        await self._update_meeting_status(next_meeting, platform, 'scheduled')
                        
                except Exception as e:
                    logger.error(f"Failed to generate prep pack: {e}")
                    # Don't fail the entire process if prep pack generation fails
                    
            elif not recurring_meeting_id:
                logger.info("Skipping meeting preparation - no recurring_meeting_id provided")
            
            result = prepare_audio_file

            # Prepare return result
            result["transcription"] = {
                "conversation": transcription_result["conversation"],
                "total_speakers": transcription_result["total_speakers"],
                "sentiments": transcription_result["sentiments"]
            }
            result["analysis"] = meeting_analysis.model_dump() if meeting_analysis else None
            result["prep_pack"] = prep_pack
            result["success"] = True
            result["meeting_id"] = meeting_id
            result["save_results"] = {
                "analysis": analysis_save_result,
                "prep_pack": prep_save_result
            }
            
            # Update meeting status to completed on success
            if platform == "google":
                await self._update_meeting_status(meeting_id, platform, 'completed')
            
            return result

        except Exception as e:
            logger.error(f"Meeting processing failed for meeting {payload.get('_id', 'unknown')}: {str(e)}")
            
            # Update meeting status to failed on error
            meeting_id = payload.get('_id')
            platform = payload.get('platform')
            if meeting_id and platform:
                await self._update_meeting_status(str(meeting_id), platform, 'failed')
            
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
