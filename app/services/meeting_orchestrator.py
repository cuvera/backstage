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
from app.services.meeting_analysis_service import MeetingAnalysisService
from app.utils.s3_client import download_s3_file
from app.services.meeting_prep_curator_service import MeetingPrepCuratorService

logger = logging.getLogger(__name__)

class MeetingOrchestratorError(Exception):
    """Custom exception for meeting orchestrator operations."""
    pass


class MeetingAlreadyProcessedException(Exception):
    """Exception raised when meeting is already processed or being processed."""
    pass
class MeetingOrchestrator:
    """Main service for orchestrating meeting processing pipeline."""
    
    def __init__(self):
        self.analysis_agent = CallAnalysisAgent()
        self.preprocessor = AudioPreprocessor()
        self.qdrant_service = Quadrant_service()
        self.td_service = TranscriptionDiarizationService()
        self.assignment_service = SpeakerAssignmentService(self.qdrant_service)
    
    async def process_meeting_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process meeting event by extracting payload and running audio merge and transcription pipeline.
        
        Args:
            event_data: Dictionary containing the meeting event with payload
            
        Returns:
            Dictionary containing processing results
        """
        try:
            logger.info("Starting meeting event processing")
            
            # Extract payload from event_data
            payload = event_data.get('payload')
            if not payload:
                raise MeetingOrchestratorError("No payload found in event_data")
            
            logger.info(f"Processing meeting: {payload.get('summary', 'Unknown')}")
            
            # Extract meeting details from payload
            meeting_id = str(payload.get('_id'))
            tenant_id = payload.get('tenantId')
            platform = payload.get('platform')
            bucket = payload.get('bucket', 'recordings')
            
            if not all([meeting_id, tenant_id]):
                raise MeetingOrchestratorError("Missing required fields in payload: _id, tenantId, or fileUrl")
            
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
            
            # Step 2: Process with vox_scribe pipeline
            logger.info("Step 2: Processing with vox_scribe pipeline")
            logger.info(f"Step 2: {file_url}")
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

            meeting_analysis_service = MeetingAnalysisService()
            await meeting_analysis_service.save_analysis(analysis_result)

            logger.info(f"Step 4: Generate meeting prepration suggestion")
            meeting_prep_service = await MeetingPrepCuratorService.from_default()
            prep_pack = await meeting_prep_service.generate_and_save_prep_pack(
                meeting_id=meeting_id,
                meeting_analysis=analysis_result,
                platform=platform
            )

            merge_result["analysis"] = analysis_result
            merge_result["prep_pack"] = prep_pack
            merge_result["success"] = True
            merge_result["meeting_id"] = meeting_id
            return merge_result

        except Exception as e:
            logger.error(f"Meeting processing failed for meeting {payload.get('_id', 'unknown')}: {str(e)}")
            raise MeetingOrchestratorError(f"Meeting processing failed: {str(e)}") from e

async def save_transcription(meeting_id, tenant_id, vox_scribe_result):
    try:
        transcription_service = TranscriptionService()
        await transcription_service.ensure_indexes()
        
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
