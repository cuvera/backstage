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
            participants = await authClient.fetch_users_by_emails(attendees_email)

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
            
            # write to json file
            import json
            with open(f"transcription_{meeting_id}.json", "w") as f:
                json.dump({
                "meeting_id": meeting_id,
                "tenant_id": tenant_id,
                "conversation": conversation,
                "total_speakers": transcription_result.get("total_speakers", 0),
                "sentiments": transcription_result.get("sentiments", {})
                }, f, indent=2)
            
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
        
        return {
            "overall": sentiments.get("overall", "neutral"),
            "participant": enhanced_participants
        }

    async def get_transcription(self, meeting_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get transcription using repository."""
        if not self._repository:
            raise RuntimeError("Repository not initialized")
        
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