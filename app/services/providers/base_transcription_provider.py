from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTranscriptionProvider(ABC):
    """
    Abstract base class for transcription providers.
    
    Defines the interface that all transcription providers must implement.
    """

    @abstractmethod
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
            TranscriptionProviderError: If transcription fails
        """
        pass


class TranscriptionProviderError(Exception):
    """Raised when a transcription provider cannot complete its task."""
    pass