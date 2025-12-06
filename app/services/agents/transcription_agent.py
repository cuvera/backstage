from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.services.providers import (
    BaseTranscriptionProvider,
    TranscriptionProviderError,
    create_transcription_provider
)

logger = logging.getLogger(__name__)


class TranscriptionAgentError(Exception):
    """Raised when the transcription agent cannot complete its task."""


class TranscriptionAgent:
    """
    Transcription agent that processes audio files and returns structured transcription
    data with sentiment analysis using configurable transcription providers.
    
    Supports both OpenAI-compatible and Gemini native providers.
    """

    def __init__(self, provider: Optional[BaseTranscriptionProvider] = None, provider_name: Optional[str] = "gemini"):
        """
        Initialize TranscriptionAgent.
        
        Args:
            provider: Pre-configured transcription provider instance
            provider_name: Name of provider to create ("openai" or "gemini")
                         If neither provider nor provider_name is specified, uses default config
        """
        if provider:
            self.provider = provider
        else:
            self.provider = create_transcription_provider(provider_name)

    async def transcribe(
        self, 
        audio_file_path: str, 
        meeting_metadata: Dict[str, Any],
        participants: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Transcribe audio file and perform sentiment analysis.
        
        Args:
            audio_file_path: Path to the audio file to transcribe
            meeting_metadata: Meeting metadata including speaker timeframes
            participants: List of participant information
            
        Returns:
            Dictionary containing:
            - conversation: List of transcription entries with speaker diarization
            - total_speakers: Number of unique speakers detected
            - sentiments: Overall and per-participant sentiment analysis
            
        Raises:
            TranscriptionAgentError: If transcription fails
        """
        try:
            logger.info(f"[TranscriptionAgent] Starting transcription for: {audio_file_path}")
            
            # Delegate to the configured provider
            result = await self.provider.transcribe(
                audio_file_path=audio_file_path,
                meeting_metadata=meeting_metadata,
                participants=participants
            )
            
            logger.info(f"[TranscriptionAgent] Transcription completed successfully. Found {result.get('total_speakers', 0)} speakers")
            return result
            
        except TranscriptionProviderError as exc:
            logger.exception(f"[TranscriptionAgent] Provider error for {audio_file_path}: {exc}")
            raise TranscriptionAgentError(f"Transcription provider failed: {exc}") from exc
        except Exception as exc:
            logger.exception(f"[TranscriptionAgent] Unexpected error for {audio_file_path}: {exc}")
            raise TranscriptionAgentError(f"Transcription processing failed: {exc}") from exc