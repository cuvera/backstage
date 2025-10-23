import asyncio
import logging
import os
import tempfile
import shutil
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path

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
from app.utils.audio_merger import merge_wav_files_from_s3, AudioMergerError
from app.services.vox_scribe.audio_preprocessing_service import AudioPreprocessor
from app.services.vox_scribe.quadrant_service import Quadrant_service
from app.services.vox_scribe.transcription_diarization_service import TranscriptionDiarizationService
from app.services.vox_scribe.speaker_assignment_service import SpeakerAssignmentService
from app.services.vox_scribe.main_pipeline import diarization_pipeline
from app.services.call_analysis_service import CallAnalysisService

logger = logging.getLogger(__name__)


class MeetingProcessingServiceError(Exception):
    """Custom exception for meeting processing service operations."""
    pass


class MeetingAlreadyProcessedException(Exception):
    """Exception raised when meeting is already processed or being processed."""
    pass


class MeetingProcessingService:
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
                raise MeetingProcessingServiceError("No payload found in event_data")
            
            logger.info(f"Processing meeting: {payload.get('summary', 'Unknown')}")
            
            # Extract meeting details from payload
            meeting_id = str(payload.get('_id'))
            tenant_id = payload.get('tenantId')
            platform = payload.get('platform')
            bucket = payload.get('bucket', 'recordings')
            
            if not all([meeting_id, tenant_id]):
                raise MeetingProcessingServiceError("Missing required fields in payload: _id, tenantId, or fileUrl")
            
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
                file_url = payload.get('fileUrl')
            
            # Step 2: Process with vox_scribe pipeline
            logger.info("Step 2: Processing with vox_scribe pipeline")
            logger.info(f"Step 2: {file_url}")
            vox_scribe_result = diarization_pipeline(
                meeting_audio_path=file_url,
                known_number_of_speakers=0
            )
            logger.info(f"VoxScribe result: {vox_scribe_result}")

            if (len(vox_scribe_result) > 0):
                merge_result["diarization"] = vox_scribe_result
                merge_result["success"] = True
                merge_result["meeting_id"] = meeting_id
            else:
                merge_result["success"] = False
                merge_result["meeting_id"] = meeting_id
                return merge_result
            
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

            call_analysis_service = CallAnalysisService()
            await call_analysis_service.save_analysis(analysis_result)

            merge_result["analysis"] = analysis_result
            merge_result["success"] = True
            merge_result["meeting_id"] = meeting_id
            return merge_result

        except Exception as e:
            logger.error(f"Meeting processing failed for meeting {payload.get('_id', 'unknown')}: {str(e)}")
            raise MeetingProcessingServiceError(f"Meeting processing failed: {str(e)}") from e
        