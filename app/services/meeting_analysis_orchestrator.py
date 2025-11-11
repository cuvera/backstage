from ast import dump
import asyncio
import logging
import os
import tempfile
import shutil
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path
import json
from app.db.mongodb import get_database
from app.models.meeting import (
    AudioProcessingResult,
    MeetingRecord,
    MeetingParticipant,
    ProcessingStatus,
    TranscriptionResult,
)
from app.schemas.meeting_analysis import MeetingAnalysis
from app.services.agents.call_analysis_agent import CallAnalysisAgent
from app.services.transcription_service import TranscriptionService
from app.utils.audio_merger import merge_wav_files_from_s3, AudioMergerError
from app.services.vox_scribe.audio_preprocessing_service import AudioPreprocessor
from app.services.vox_scribe.quadrant_service import Quadrant_service
from app.services.vox_scribe.transcription_diarization_service import TranscriptionDiarizationService
from app.services.vox_scribe.speaker_assignment_service import SpeakerAssignmentService
from app.services.vox_scribe.main_pipeline import diarization_pipeline
from app.services.vox_scribe.transcription_pipeline import transcription_with_timeframes, validate_speaker_timeframes
from app.services.meeting_analysis_service import MeetingAnalysisService
from app.repository import MeetingMetadataRepository
from app.utils.s3_client import download_s3_file
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
        self.analysis_agent = CallAnalysisAgent()
        self.preprocessor = AudioPreprocessor()
        self.qdrant_service = Quadrant_service()
        self.td_service = TranscriptionDiarizationService()
        self.assignment_service = SpeakerAssignmentService(self.qdrant_service)
        self.meeting_metadata_repo = None  # Will be initialized when needed
    
    async def analyze_meeting(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process meeting event by extracting payload and running audio merge and transcription pipeline.
        
        Args:
            event_data: Dictionary containing the meeting event with payload
            
        Returns:
            Dictionary containing processing results
        """
        try:
            logger.info("Starting meeting event processing")
            if not payload:
                raise MeetingAnalysisOrchestratorError("No payload found in event_data")
            
            logger.info(f"Processing meeting: {payload.get('summary', 'Unknown')}")
            
            # Extract meeting details from payload
            meeting_id = str(payload.get('_id'))
            tenant_id = payload.get('tenantId')
            platform = payload.get('platform')
            bucket = payload.get('bucket', 'recordings')
            recurring_meeting_id = payload.get('recurring_meeting_id')
            
            if not all([meeting_id, tenant_id]):
                raise MeetingAnalysisOrchestratorError("Missing required fields in payload: _id, tenantId, or fileUrl")
            
            # Update meeting status to pending at start of processing
            await self._update_meeting_status(meeting_id, platform, 'analysing')
            
            # Step 1: Merge audio files from S3
            logger.info("Step 1: Merging audio files from S3")
            if not payload.get('fileUrl'):
                s3_folder_path = f"{tenant_id}/{platform}/{meeting_id}/"
                output_s3_key = f"{tenant_id}/{platform}/{meeting_id}/meeting.wav"

                # output_s3_key
                merge_result = await merge_wav_files_from_s3(
                    s3_folder_path=s3_folder_path,
                    output_s3_key=output_s3_key,
                    bucket_name=bucket
                )
                file_url = merge_result['local_merged_file_path']
            else:
                # download file from s3
                temp_dir = tempfile.mkdtemp(prefix="audio_merge_")
                file_url = os.path.join(temp_dir, "merged_output.wav")
                await download_s3_file(payload.get('fileUrl'), file_url ,bucket)
                # Initialize merge_result for the downloaded file case
                merge_result = {
                    'local_merged_file_path': file_url,
                    'temp_directory': temp_dir
                }
            
            # Step 2: Process with vox_scribe pipeline (platform-specific)
            logger.info("Step 2: Processing with vox_scribe pipeline")
            logger.info(f"Step 2: {file_url}")
            
            if platform == "google":
                # For Google meetings, use transcription with pre-identified speaker timeframes
                logger.info("Using Google meeting transcription with timeframes")
                
                # Initialize repository if needed and fetch speaker timeframes
                if not self.meeting_metadata_repo:
                    self.meeting_metadata_repo = await MeetingMetadataRepository.from_default()
                
                speaker_timeframes = await self.meeting_metadata_repo.get_speaker_timeframes(meeting_id)
                
                if speaker_timeframes and validate_speaker_timeframes(speaker_timeframes):
                    logger.info(f"Found {len(speaker_timeframes)} speaker timeframes for Google meeting")
                    vox_scribe_result = transcription_with_timeframes(
                        meeting_audio_path=file_url,
                        speaker_timeframes=speaker_timeframes
                    )
                    
                    # Check if transcription with timeframes produced any results
                    if not vox_scribe_result or len(vox_scribe_result) == 0:
                        logger.warning("Google timeframe transcription produced no results, falling back to diarization")
                        vox_scribe_result = diarization_pipeline(
                            meeting_audio_path=file_url,
                            known_number_of_speakers=0
                        )
                else:
                    logger.warning("No valid speaker timeframes found, falling back to diarization")
                    vox_scribe_result = diarization_pipeline(
                        meeting_audio_path=file_url,
                        known_number_of_speakers=0
                    )
            else:
                # For non-Google platforms, use existing diarization pipeline
                logger.info(f"Using diarization pipeline for platform: {platform}")
                vox_scribe_result = diarization_pipeline(
                    meeting_audio_path=file_url,
                    known_number_of_speakers=0
                )

            if (len(vox_scribe_result) > 0):
                await save_transcription(meeting_id, tenant_id, vox_scribe_result)
            
            logger.info(f"Step 3: Meeting analysis")
            # Handle case where transcript_payload is a list (from vox_scribe pipeline)
            if isinstance(vox_scribe_result, list):
                # Convert list to expected dictionary format
                conversation_list = vox_scribe_result
                transcript_payload = {
                    "conversation": conversation_list,
                    "tenant_id": tenant_id,
                    "session_id": meeting_id
                }

            # printtranscript_payload
            analysis_result = self.analysis_agent.analyze(
                transcript_payload=transcript_payload or {}
            )

            meeting_analysis_service = await MeetingAnalysisService.from_default()
            await meeting_analysis_service.save_analysis(analysis_result)

            logger.info(f"Step 4: Generate meeting prepration suggestion")
            prep_pack = None
            if recurring_meeting_id:
                meeting_prep_service = await MeetingPrepCuratorService.from_default()
                prep_pack = await meeting_prep_service.generate_and_save_prep_pack(
                    meeting_id=meeting_id,
                    meeting_analysis=analysis_result,
                    platform=platform
                )
            else:
                logger.info("Skipping meeting preparation - no recurring_meeting_id provided")

            merge_result["analysis"] = analysis_result
            merge_result["prep_pack"] = prep_pack
            merge_result["success"] = True
            merge_result["meeting_id"] = meeting_id
            
            # Update meeting status to completed on success
            await self._update_meeting_status(meeting_id, platform, 'completed')
            
            return merge_result

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

async def save_transcription(meeting_id, tenant_id, vox_scribe_result):
    try:
        transcription_service = await TranscriptionService.from_default()
        
        transcription_result = await transcription_service.save_transcription(
            meeting_id=meeting_id,
            tenant_id=tenant_id,
            conversation=vox_scribe_result,
            processing_metadata={
                "vox_scribe_version": "1.0",
                "known_speakers": 0,
                "audio_duration_seconds": None
            }
        )
        logger.info(f"Saved transcription to MongoDB: {transcription_result}")
        
    except Exception as e:
        logger.error(f"Failed to save transcription: {e}")
