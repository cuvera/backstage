import logging
from typing import Any, Dict, List, Optional

from app.repository import TranscriptionRepository

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