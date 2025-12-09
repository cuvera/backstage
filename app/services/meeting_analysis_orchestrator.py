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
from app.messaging.producers.email_notification_producer import send_email_notification
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

            await self._update_meeting_status(meeting_id, platform, 'analysing', tenant_id)

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
                        await self._update_meeting_status(next_meeting, platform, 'analysing', tenant_id)
                        logger.info(f"Next meeting identified for prep pack - next_meeting_id={next_meeting}")
            else:
                meeting_metadata = {}
                logger.debug(f"Using empty meeting metadata for non-Google platform - meeting_id={meeting_id}")

            step_duration_ms = round((time.time() - step_start_time) * 1000, 2)
            logger.info(f"Step 0 completed in {step_duration_ms}ms - meeting_id={meeting_id}")

            #################################################
            # Step 1: Prepare audio file
            #################################################
            step_start_time = time.time()
            logger.info(f"Step 1: Starting transcription process - meeting_id={meeting_id}")
            transcription_service = TranscriptionService()
            transcription = await transcription_service.get_transcription(meeting_id, tenant_id)

            if not transcription:
                logger.info(f"Step 1.1: No existing transcription found, preparing audio file - meeting_id={meeting_id}")
                prepare_audio_file = await self._prepare_audio_file(payload)

                file_url = prepare_audio_file.get("local_merged_file_path")
                temp_directory = prepare_audio_file.get("temp_directory")
                if not file_url:
                    raise MeetingAnalysisOrchestratorError("Failed to prepare audio file")

                # Step 1.2: Transcription with TranscriptionAgent
                logger.info(f"Step 1.2: Starting audio transcription - meeting_id={meeting_id}")
                transcription = await transcription_service.save_transcription(
                    audio_file_path=file_url,
                    meeting_id=meeting_id,
                    tenant_id=tenant_id,
                    meeting_metadata=meeting_metadata,
                )

            else:
                logger.info(f"Step 1: Using existing transcription - meeting_id={meeting_id}")

            step_duration_ms = round((time.time() - step_start_time) * 1000, 2)
            logger.info(f"Step 1 completed in {step_duration_ms}ms - meeting_id={meeting_id}")
            
            logger.debug("Waiting for 2 seconds before analysis...")
            time.sleep(2)

            #################################################
            # Step 2: Meeting Analysis with CallAnalysisAgent
            #################################################
            step_start_time = time.time()
            logger.info(f"Step 2: Starting meeting analysis - meeting_id={meeting_id}")

            analysis_service = await MeetingAnalysisService.from_default()
            analysis_doc = await analysis_service.get_analysis(tenant_id=tenant_id, session_id=meeting_id)

            if not analysis_doc:
                call_analysis_agent = CallAnalysisAgent()

                analysis = await call_analysis_agent.analyze(
                        transcript_payload=transcription,
                        context={
                            "tenant_id": tenant_id,
                            "session_id": meeting_id,
                            "platform": platform
                    }
                )

                logger.info(f"Step 2.1: Saving meeting analysis - meeting_id={meeting_id}")
                await analysis_service.save_analysis(analysis)
            else:
                analysis = MeetingAnalysis(**analysis_doc)
                logger.info(f"Step 2: Using existing analysis - meeting_id={meeting_id}")
            
            step_duration_ms = round((time.time() - step_start_time) * 1000, 2)
            logger.info(f"Step 2 completed in {step_duration_ms}ms - meeting_id={meeting_id}")
            
            logger.debug("Waiting for 2 seconds before meeting preparation...")
            time.sleep(2)

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
                    platform=platform,
                    recurring_meeting_id=recurring_meeting_id,
                    previous_meeting_counts=2,
                    context={
                        "current_meeting_id": meeting_id,
                        "tenant_id": tenant_id
                    }
                )

                await self._update_meeting_status(next_meeting, platform, 'scheduled', tenant_id)
                logger.info(f"Meeting prep pack generated successfully - next_meeting_id={next_meeting}")
            else:
                logger.info(f"Skipping meeting preparation - missing requirements - meeting_id={meeting_id}")
            
            step_duration_ms = round((time.time() - step_start_time) * 1000, 2)
            logger.info(f"Step 3 completed in {step_duration_ms}ms - meeting_id={meeting_id}")

            # Update meeting status to completed on success
            await self._update_meeting_status(meeting_id, platform, 'completed', tenant_id)
            
            # Step 4: Send email notification
            await self._send_meeting_completion_email(
                meeting_id=meeting_id,
                tenant_id=tenant_id,
                analysis=analysis,
                meeting_metadata=meeting_metadata,
                transcription=transcription
            )
            
            total_duration_ms = round((time.time() - overall_start_time) * 1000, 2)
            logger.info(f"Meeting analysis completed successfully in {total_duration_ms}ms - meeting_id={meeting_id}, tenant_id={tenant_id}")
            return True

        except Exception as e:
            logger.error(f"Meeting processing failed - meeting_id={meeting_id}, tenant_id={tenant_id}: {str(e)}", exc_info=True)
            
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

    async def _prepare_audio_file(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare audio file for processing."""
        bucket = payload.get('bucket', settings.MEETING_BUCKET_NAME)
        temp_dir = tempfile.mkdtemp(prefix="audio_merge_", suffix=datetime.now().strftime("_%Y%m%d_%H%M%S"))
        file_url = None
        
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
        else:
            file_url = os.path.join(temp_dir, "merged_output.wav")
            await download_s3_file(payload.get('fileUrl'), file_url, bucket)

        return {
            'local_merged_file_path': file_url,
            'temp_directory': temp_dir
        }

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
                user_details = await auth_client.fetch_users_by_emails(list(all_emails))
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
            duration_sec = int(analysis.duration_sec)/1000 or 0
            hours = int(duration_sec // 3600)
            minutes = int((duration_sec % 3600) // 60)
            duration_str = f"{hours} Hour {minutes} Minutes" if hours > 0 else f"{minutes} Minutes"
            
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
