import logging
from typing import Any, Dict, List, Optional

from app.repository import TranscriptionRepository
from app.services.agents import TranscriptionAgent
from app.utils.auth_service_client import AuthServiceClient

logger = logging.getLogger(__name__)


class TranscriptionServiceError(Exception):
    """Raised when the transcription service cannot complete its task."""


class TranscriptionService:
    """Service for managing meeting transcriptions. Delegates data operations to TranscriptionRepository."""

    def __init__(self, repository: Optional[TranscriptionRepository] = None):
        self._repository = repository

    @classmethod
    async def from_default(cls) -> "TranscriptionService":
        repository = await TranscriptionRepository.from_default()
        return cls(repository=repository)

    async def save_transcription(
        self, 
        audio_file_path: str,
        meeting_id: str,
        tenant_id: str,
        meeting_metadata: Dict[str, Any],
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Transcribe audio file and save transcription to database.
        
        Args:
            audio_file_path: Path to audio file to transcribe
            meeting_id: Meeting identifier
            tenant_id: Tenant identifier  
            meeting_metadata: Meeting metadata including speaker timeframes
            processing_metadata: Optional processing metadata
            
        Returns:
            Saved transcription document
            
        Raises:
            TranscriptionServiceError: If transcription or saving fails
        """
        self._repository = await TranscriptionRepository.from_default()
        try:
            logger.info(f"Starting transcription service for meeting={meeting_id}")
            
            # From meeting_metadata extract participants emails
            attendees_email = meeting_metadata.get("attendees", [])

            print(attendees_email)

            authClient = AuthServiceClient()
            try:
                participants = await authClient.fetch_users_by_emails(attendees_email)
            except Exception as e:
                logger.warning(f"Failed to fetch user details from auth service: {e}, proceeding without participant data")
                participants = []

            # Use TranscriptionAgent to transcribe audio
            agent = TranscriptionAgent()
            transcription_result = await agent.transcribe(audio_file_path, meeting_metadata, participants)
            
            # Extract conversation from transcription result
            # The conversation entries should already contain userId from the agent
            conversation = transcription_result["conversation"]
                        
            # Add transcription metadata to processing metadata
            enriched_metadata = processing_metadata or {}
            enriched_metadata.update({
                "total_speakers": transcription_result.get("total_speakers", 0),
                "transcription_agent_version": "1.0"
            })
            
            # Extract and enhance sentiments with user IDs
            sentiments = transcription_result.get("sentiments", {})
            enhanced_sentiments = self._map_sentiments_to_users(sentiments, participants)

            # Save to database using private method
            result = await self._repository.save_transcription(
                meeting_id=meeting_id,
                tenant_id=tenant_id,
                conversation=conversation,
                sentiments=enhanced_sentiments,
            )
            
            logger.info(f"Transcription completed and saved for meeting={meeting_id}")
            return {
                "conversation": conversation,
                "total_speakers": transcription_result.get("total_speakers", 0),
                "sentiments": enhanced_sentiments,
                "save_result": result
            }
            
        except Exception as exc:
            logger.exception(f"Failed to transcribe and save for meeting={meeting_id}: {exc}")
            raise TranscriptionServiceError(f"Failed to transcribe and save: {exc}") from exc

    def _map_sentiments_to_users(self, sentiments: Dict[str, Any], participants: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Map sentiment data to include user IDs from participants.
        
        Args:
            sentiments: Raw sentiment data from TranscriptionAgent
            participants: List of participant data from auth service
            
        Returns:
            Enhanced sentiments with user IDs mapped
        """
        # Create mapping of participant names to user IDs
        name_to_user_id = {}
        email_to_user_id = {}
        
        for participant in participants:
            if participant.get("name"):
                name_to_user_id[participant["name"]] = participant.get("id")
            if participant.get("email"):
                email_to_user_id[participant["email"]] = participant.get("id")
        
        # Enhance participant sentiments with user IDs
        enhanced_participants = []
        for participant_sentiment in sentiments.get("participant", []):
            participant_name = participant_sentiment.get("name", "")
            user_id = name_to_user_id.get(participant_name)
            
            # If no direct name match, try email-based matching
            if not user_id and "@" in participant_name:
                user_id = email_to_user_id.get(participant_name)
            
            enhanced_participants.append({
                "name": participant_name,
                "userId": user_id,
                "sentiment": participant_sentiment.get("sentiment", "neutral")
            })
        
        return speakers_array

    def _merge_transcription_results(self, transcription_results: List[Dict[str, Any]], meeting_metadata: Dict = None) -> Dict[str, Any]:
        """
        Merge all chunk transcription results into a single transcription with absolute timeline and speaker mapping
        
        Args:
            transcription_results: List of transcription results from chunks
            meeting_metadata: Meeting metadata containing speaker_timeframes
            
        Returns:
            Merged transcription result with absolute timeline, speaker mapping, and speaker summary
        """
        logger.info(f"[TranscriptionService] Merging {len(transcription_results)} transcription results")
        
        all_segments = []
        successful_chunks = 0
        failed_chunks = 0
        
        for result in transcription_results:
            chunk_info = result.get("chunk_info", {})

            if "error" in chunk_info:
                failed_chunks += 1
                logger.warning(f"[TranscriptionService] Skipping failed chunk {chunk_info.get('chunk_id')}")
                continue

            chunk_start_seconds = self._time_to_seconds(chunk_info.get("start_time", "00:00"))

            transcriptions = result.get("transcriptions", [])
            for transcription in transcriptions:
                # Get segment times relative to chunk
                segment_start_seconds = self._time_to_seconds(transcription.get("start", "00:00"))
                segment_end_seconds = self._time_to_seconds(transcription.get("end", "00:00"))

                # Calculate absolute start/end from chunk_start_time
                absolute_start = chunk_start_seconds + segment_start_seconds
                absolute_end = chunk_start_seconds + segment_end_seconds

                # Update start/end with absolute values
                transcription["start"] = self._seconds_to_time(absolute_start)
                transcription["end"] = self._seconds_to_time(absolute_end)

                # Keep chunk context
                transcription["source_chunk"] = chunk_info.get("chunk_id")
                transcription["chunk_start_time"] = chunk_info.get("start_time", "00:00")
                transcription["chunk_end_time"] = chunk_info.get("end_time", "00:00")

                all_segments.append(transcription)

            successful_chunks += 1

        # Sort by source_chunk (ascending), then by start time within chunk
        all_segments.sort(key=lambda x: (x.get("source_chunk", 0), self._time_to_seconds(x["start"])))
        
        # Add speaker mapping if available
        if meeting_metadata and meeting_metadata.get("speaker_timeframes"):
            all_segments = self._map_speakers_to_segments(all_segments, meeting_metadata)
        
        # Generate speaker summary
        speakers_summary = self._extract_speaker_summary(all_segments)
        
        merged_result = {
            "transcriptions": all_segments,
            "speakers": speakers_summary,
            "metadata": {
                "total_segments": len(all_segments),
                "successful_chunks": successful_chunks,
                "failed_chunks": failed_chunks,
                "total_chunks": len(transcription_results),
                "has_speaker_mapping": bool(meeting_metadata and meeting_metadata.get("speaker_timeframes"))
            }
        }

    async def get_transcription(self, meeting_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get transcription using repository."""
        self._repository = await TranscriptionRepository.from_default()        
        
        try:
            return await self._repository.get_transcription(meeting_id, tenant_id)
        except Exception as exc:
            logger.exception("Failed to get transcription for meeting=%s: %s", meeting_id, exc)
            raise TranscriptionServiceError(f"Failed to get transcription: {exc}") from exc

    async def delete_transcription(self, meeting_id: str, tenant_id: str) -> Dict[str, Any]:
        """Delete transcription using repository."""
        if not self._repository:
            raise RuntimeError("Repository not initialized")
        
        try:
            return await self._repository.delete_transcription(meeting_id, tenant_id)
        except Exception as exc:
            logger.exception("Failed to delete transcription for meeting=%s: %s", meeting_id, exc)
            raise TranscriptionServiceError(f"Failed to delete transcription: {exc}") from exc