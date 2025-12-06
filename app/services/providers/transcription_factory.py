import logging
from typing import Optional

from app.core.config import settings
from .base_transcription_provider import BaseTranscriptionProvider, TranscriptionProviderError
from .openai_transcription_provider import OpenAITranscriptionProvider
from .gemini_transcription_provider import GeminiTranscriptionProvider

logger = logging.getLogger(__name__)


def create_transcription_provider(provider_name: Optional[str] = None) -> BaseTranscriptionProvider:
    """
    Factory function to create transcription providers based on configuration.
    
    Args:
        provider_name: Override provider name. If None, uses settings.TRANSCRIPTION_PROVIDER
        
    Returns:
        Configured transcription provider instance
        
    Raises:
        TranscriptionProviderError: If provider is unknown or configuration is invalid
    """
    provider = provider_name or settings.TRANSCRIPTION_PROVIDER
    
    logger.info(f"Creating transcription provider: {provider}")
    
    if provider == "openai":
        return OpenAITranscriptionProvider()
    elif provider == "gemini":
        return GeminiTranscriptionProvider()
    else:
        available_providers = ["openai", "gemini"]
        raise TranscriptionProviderError(
            f"Unknown transcription provider: {provider}. "
            f"Available providers: {available_providers}"
        )


# Convenience function for getting the default configured provider
def get_default_transcription_provider() -> BaseTranscriptionProvider:
    """
    Get the default transcription provider based on configuration.
    
    Returns:
        Default configured transcription provider
    """
    return create_transcription_provider()