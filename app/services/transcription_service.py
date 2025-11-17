import logging
from typing import Any, Dict, List, Optional

from app.repository import TranscriptionRepository
from app.services.agents import TranscriptionAgent

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

    async def _save_transcription(
        self, 
        meeting_id: str,
        tenant_id: str,
        conversation: List[Dict[str, Any]],
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Save transcription using repository."""
        if not self._repository:
            raise RuntimeError("Repository not initialized")
        
        try:
            return await self._repository.save_transcription(
                meeting_id=meeting_id,
                tenant_id=tenant_id,
                conversation=conversation,
                processing_metadata=processing_metadata
            )
        except Exception as exc:
            logger.exception("Failed to save transcription for meeting=%s: %s", meeting_id, exc)
            raise TranscriptionServiceError(f"Failed to save transcription: {exc}") from exc

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
        try:
            logger.info(f"Starting transcription service for meeting={meeting_id}")
            
            # Use TranscriptionAgent to transcribe audio
            agent = TranscriptionAgent()
            transcription_result = await agent.transcribe(audio_file_path, meeting_metadata)
            
            # Extract conversation from transcription result
            conversation = transcription_result["conversation"]
            
            # Add transcription metadata to processing metadata
            enriched_metadata = processing_metadata or {}
            enriched_metadata.update({
                "total_speakers": transcription_result.get("total_speakers", 0),
                "sentiments": transcription_result.get("sentiments", {}),
                "transcription_agent_version": "1.0"
            })
            
            # Save to database using private method
            result = await self._save_transcription(
                meeting_id=meeting_id,
                tenant_id=tenant_id,
                conversation=conversation,
                processing_metadata=enriched_metadata
            )
            
            logger.info(f"Transcription completed and saved for meeting={meeting_id}")
            return result
            
        except Exception as exc:
            logger.exception(f"Failed to transcribe and save for meeting={meeting_id}: {exc}")
            raise TranscriptionServiceError(f"Failed to transcribe and save: {exc}") from exc

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