import logging
import os
import tempfile
from typing import Dict, Any
import time
from datetime import datetime
import shutil

from app.schemas.meeting_analysis import MeetingAnalysis
from app.services.transcription_service import TranscriptionService
from app.utils.audio_merger import merge_wav_files_from_s3, AudioMergerError
from app.services.meeting_analysis_service import MeetingAnalysisService
from app.repository import MeetingMetadataRepository
from app.utils.s3_client import download_s3_file
from app.core.config import settings
from app.services.agents import TranscriptionAgent, CallAnalysisAgent
from app.services.meeting_prep_curator_service import MeetingPrepCuratorService
from app.messaging.producers.meeting_status_producer import send_meeting_status

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

        if not all([meeting_id, tenant_id, platform]):
            logger.error(
                "Missing required fields in payload",
                extra={
                    "missing_fields": [f for f, v in [("_id", meeting_id), ("tenantId", tenant_id), ("platform", platform)] if not v],
                    "payload_keys": list(payload.keys()) if payload else []
                }
            )
            return

        logger.info(
            "Starting meeting analysis",
            extra={
                "meeting_id": meeting_id,
                "tenant_id": tenant_id,
                "platform": platform,
                "recurring_meeting_id": recurring_meeting_id
            }
        )

        next_meeting = None
        file_url = None
        temp_directory = None
        overall_start_time = time.time()

        try:
            ###########################################
            # Step 0: Prepare meeting info and context
            ###########################################
            step_start_time = time.time()
            logger.info(
                "Step 0: Preparing meeting context",
                extra={"meeting_id": meeting_id, "platform": platform}
            )

            await self._update_meeting_status(meeting_id, platform, 'analysing', tenant_id)

            if platform == "google":
                if not self.meeting_metadata_repo:
                    self.meeting_metadata_repo = await MeetingMetadataRepository.from_default()
                
                meeting_metadata = await self.meeting_metadata_repo.get_meeting_metadata(meeting_id)

                logger.debug(
                    "Meeting metadata retrieved",
                    extra={
                        "meeting_id": meeting_id,
                        "has_metadata": bool(meeting_metadata),
                        "has_recurring_id": bool(meeting_metadata and meeting_metadata.get("recurring_meeting_id"))
                    }
                )

                if meeting_metadata and meeting_metadata.get("recurring_meeting_id"):
                    recurring_meeting_id = meeting_metadata.get("recurring_meeting_id")

                    # fetch immediate next meeting from recurring_meeting_id
                    next_meeting = await self.meeting_metadata_repo.find_immediate_next_meeting(
                        current_meeting_metadata=meeting_metadata,
                        recurring_meeting_id=recurring_meeting_id,
                        platform=platform
                    )

                    if next_meeting:
                        await self._update_meeting_status(next_meeting, platform, 'analysing', tenant_id)
                        logger.info(
                            "Next meeting identified for prep pack",
                            extra={"next_meeting_id": next_meeting, "recurring_meeting_id": recurring_meeting_id}
                        )
            else:
                meeting_metadata = {}
                logger.debug("Using empty meeting metadata for non-Google platform")

            logger.info(
                "Step 0 completed",
                extra={
                    "meeting_id": meeting_id,
                    "duration_ms": round((time.time() - step_start_time) * 1000, 2),
                    "step": "meeting_context_preparation"
                }
            )

            #################################################
            # Step 1: Prepare audio file
            #################################################
            step_start_time = time.time()
            logger.info(
                "Step 1: Starting transcription process",
                extra={"meeting_id": meeting_id, "tenant_id": tenant_id}
            )
            transcription_service = TranscriptionService()
            transcription = await transcription_service.get_transcription(meeting_id, tenant_id)

            if not transcription:
                logger.info(
                    "Step 1.1: No existing transcription found, preparing audio file",
                    extra={"meeting_id": meeting_id}
                )
                prepare_audio_file = await self._prepare_audio_file(payload)

                file_url = prepare_audio_file.get("local_merged_file_path")
                temp_directory = prepare_audio_file.get("temp_directory")
                if not file_url:
                    raise MeetingAnalysisOrchestratorError("Failed to prepare audio file")

                # Step 1.2: Transcription with TranscriptionAgent
                logger.info(
                    "Step 1.2: Starting audio transcription",
                    extra={"meeting_id": meeting_id, "audio_file_path": file_url}
                )
                transcription = await transcription_service.save_transcription(
                    audio_file_path=file_url,
                    meeting_id=meeting_id,
                    tenant_id=tenant_id,
                    meeting_metadata=meeting_metadata,
                )

            else:
                logger.info(
                    "Step 1: Using existing transcription",
                    extra={
                        "meeting_id": meeting_id,
                        "conversation_turns": len(transcription.get("conversation", [])) if transcription else 0
                    }
                )

            logger.info(
                "Step 1 completed",
                extra={
                    "meeting_id": meeting_id,
                    "duration_ms": round((time.time() - step_start_time) * 1000, 2),
                    "step": "transcription_preparation",
                    "transcription_existed": transcription is not None
                }
            )
            
            logger.debug("Waiting for 2 seconds before analysis...")
            time.sleep(2)

            #################################################
            # Step 2: Meeting Analysis with CallAnalysisAgent
            #################################################
            step_start_time = time.time()
            logger.info(
                "Step 2: Starting meeting analysis",
                extra={"meeting_id": meeting_id, "tenant_id": tenant_id}
            )

            analysis_service = await MeetingAnalysisService.from_default()
            analysis_doc = await analysis_service.get_analysis(tenant_id=tenant_id, session_id=meeting_id)

            if not analysis_doc:
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

                logger.info(
                    "Step 2.1: Saving meeting analysis",
                    extra={
                        "meeting_id": meeting_id,
                        "analysis_summary_length": len(analysis.summary) if analysis else 0,
                        "action_items_count": len(analysis.action_items) if analysis else 0,
                        "decisions_count": len(analysis.decisions) if analysis else 0
                    }
                )
                await analysis_service.save_analysis(analysis)
            else:
                analysis = MeetingAnalysis(**analysis_doc)
                logger.info(
                    "Step 2: Using existing analysis",
                    extra={"meeting_id": meeting_id, "session_id": analysis.session_id}
                )
            
            logger.info(
                "Step 2 completed",
                extra={
                    "meeting_id": meeting_id,
                    "duration_ms": round((time.time() - step_start_time) * 1000, 2),
                    "step": "meeting_analysis",
                    "analysis_existed": analysis_doc is not None
                }
            )
            
            logger.debug("Waiting for 2 seconds before meeting preparation...")
            time.sleep(2)

            ############################################################
            # Step 3: Meeting Preparation with MeetingPrepCuratorService
            ############################################################
            step_start_time = time.time()
            logger.info(
                "Step 3: Starting meeting preparation",
                extra={
                    "meeting_id": meeting_id,
                    "recurring_meeting_id": recurring_meeting_id,
                    "next_meeting": next_meeting
                }
            )
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

                await self._update_meeting_status(next_meeting, platform, 'scheduled', tenant_id)
                logger.info(
                    "Meeting prep pack generated successfully",
                    extra={
                        "next_meeting_id": next_meeting,
                        "recurring_meeting_id": recurring_meeting_id
                    }
                )
            else:
                logger.info(
                    "Skipping meeting preparation - missing requirements",
                    extra={
                        "has_recurring_meeting_id": bool(recurring_meeting_id),
                        "has_next_meeting": bool(next_meeting)
                    }
                )
            
            logger.info(
                "Step 3 completed",
                extra={
                    "meeting_id": meeting_id,
                    "duration_ms": round((time.time() - step_start_time) * 1000, 2),
                    "step": "meeting_preparation",
                    "prep_pack_generated": bool(recurring_meeting_id and next_meeting)
                }
            )

            # Update meeting status to completed on success
            await self._update_meeting_status(meeting_id, platform, 'completed', tenant_id)
            
            logger.info(
                "Meeting analysis completed successfully",
                extra={
                    "meeting_id": meeting_id,
                    "tenant_id": tenant_id,
                    "platform": platform,
                    "total_duration_ms": round((time.time() - overall_start_time) * 1000, 2),
                    "transcription_existed": transcription is not None,
                    "analysis_existed": analysis_doc is not None,
                    "prep_pack_generated": bool(recurring_meeting_id and next_meeting)
                }
            )
            return True

        except Exception as e:
            logger.error(
                "Meeting processing failed",
                extra={
                    "meeting_id": meeting_id,
                    "tenant_id": tenant_id,
                    "platform": platform,
                    "error_type": type(e).__name__,
                    "step": "analyze_meeting",
                    "total_duration_ms": round((time.time() - overall_start_time) * 1000, 2) if 'overall_start_time' in locals() else None
                },
                exc_info=True
            )
            
            if meeting_id and platform:
                await self._update_meeting_status(str(meeting_id), platform, 'failed', tenant_id)

                if next_meeting:
                    await self._update_meeting_status(next_meeting, platform, 'scheduled', tenant_id)
            
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
                    logger.info(
                        "Temporary resources cleaned up",
                        extra={
                            "meeting_id": meeting_id,
                            "cleaned_files_count": len(cleaned_files),
                            "cleaned_directories_count": len(cleaned_directories),
                            "cleanup_duration_ms": round((time.time() - cleanup_start_time) * 1000, 2)
                        }
                    )
            except Exception as e:
                logger.error(
                    "Failed to clean up temporary resources",
                    extra={
                        "meeting_id": meeting_id,
                        "file_url": file_url,
                        "temp_directory": temp_directory,
                        "cleanup_step": "finally_block",
                        "error_type": type(e).__name__
                    },
                    exc_info=True
                )

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

    async def _prepare_audio_file(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare audio file for processing."""
        bucket = payload.get('bucket', settings.MEETING_BUCKET_NAME)
        temp_dir = tempfile.mkdtemp(prefix="audio_merge_", suffix=datetime.now().strftime("_%Y%m%d_%H%M%S"))
        
        if not payload.get('fileUrl'):
            s3_folder_path = f"{payload.get('tenantId')}/{payload.get('platform')}/{payload.get('_id')}/"
            output_s3_key = f"{payload.get('tenantId')}/{payload.get('platform')}/{payload.get('_id')}/meeting.wav"

            merge_result = await merge_wav_files_from_s3(
                s3_folder_path=s3_folder_path,
                output_s3_key=output_s3_key,
                bucket_name=bucket,
                temp_dir=temp_dir
            )
            file_url = merge_result['local_merged_file_path']
            temp_directory = merge_result['temp_directory']

            merge_result = {
                'local_merged_file_path': file_url,
                'temp_directory': temp_directory
            }
        else:
            file_url = os.path.join(temp_dir, "merged_output.wav")
            await download_s3_file(payload.get('fileUrl'), file_url, bucket)
            merge_result = {
                'local_merged_file_path': file_url,
                'temp_directory': temp_dir
            }

        return merge_result
