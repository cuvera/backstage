import asyncio
import json
import logging
import time
import os
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from app.core.config import settings

logger = logging.getLogger(__name__)


class TranscriptionAgentError(Exception):
    """Custom exception for transcription agent operations."""
    pass


class GeminiTranscriptionAgent:
    """
    Gemini transcription agent with model fallback capability.
    
    Takes a prompt and audio file, uploads to Gemini, and generates
    transcriptions with segment-level output and sentiment analysis.
    """

    def __init__(self, models: Optional[List[str]] = None):
        """
        Initialize Gemini transcription agent.
        
        Args:
            models: List of Gemini models to try in order (fallback chain)
        """
        if not settings.GEMINI_API_KEY:
            raise TranscriptionAgentError("GEMINI_API_KEY is required for Gemini agent")
        
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.models = models or ["gemini-2.5-flash", "gemini-2.5-pro"]
        logger.info(f"[Gemini Agent] Initialized with fallback chain: {' -> '.join(self.models)}")

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error should trigger fallback to next model.

        Args:
            error: The exception that occurred

        Returns:
            True if error is retryable, False otherwise
        """
        error_message = str(error).lower()

        # Rate limiting errors
        if any(keyword in error_message for keyword in [
            "rate limit", "quota exceeded", "too many requests", "429"
        ]):
            return True

        # Model availability errors
        if any(keyword in error_message for keyword in [
            "model not found", "model unavailable", "service unavailable",
            "internal error", "503", "502", "500"
        ]):
            return True

        # Timeout errors
        if any(keyword in error_message for keyword in [
            "timeout", "deadline exceeded", "connection reset"
        ]):
            return True

        # Model capacity errors
        if any(keyword in error_message for keyword in [
            "overloaded", "capacity", "resource exhausted"
        ]):
            return True

        return False

    async def _upload_audio_file(self, audio_file_path: str):
        """
        Upload audio file to Gemini Files API.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Uploaded file object
            
        Raises:
            TranscriptionAgentError: If upload fails
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_file_path):
                raise TranscriptionAgentError(f"Audio file not found: {audio_file_path}")
            
            logger.info(f"[Gemini Agent] Uploading audio file: {audio_file_path}")
            upload_start_time = time.time()
            
            uploaded_file = await asyncio.to_thread(self.client.files.upload, file=audio_file_path)
            
            # Wait for file processing
            while uploaded_file.state == "PROCESSING":
                logger.info("[Gemini Agent] File processing, waiting...")
                await asyncio.sleep(1)
                uploaded_file = await asyncio.to_thread(self.client.files.get, uploaded_file.name)
            
            if uploaded_file.state == "FAILED":
                raise TranscriptionAgentError("File upload to Gemini failed")
            
            upload_duration_ms = round((time.time() - upload_start_time) * 1000, 2)
            logger.info(f"[Gemini Agent] File uploaded and processed in {upload_duration_ms}ms")
            
            return uploaded_file
            
        except Exception as e:
            logger.error(f"[Gemini Agent] File upload failed: {e}")
            if isinstance(e, TranscriptionAgentError):
                raise
            raise TranscriptionAgentError(f"File upload failed: {e}") from e

    async def _transcribe_with_model(
        self,
        model: str,
        prompt: str,
        uploaded_file,
    ) -> Dict[str, Any]:
        """
        Transcribe audio using specific Gemini model.

        Args:
            model: Gemini model name
            prompt: Custom transcription prompt
            uploaded_file: Uploaded file object

        Returns:
            Transcription result dictionary
        """
        logger.info(f"[Gemini Agent] Starting transcription with model: {model}")
        llm_start_time = time.time()

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=model,
            contents=[
                prompt,
                uploaded_file
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )

        llm_duration_ms = round((time.time() - llm_start_time) * 1000, 2)
        logger.info(f"[Gemini Agent] Transcription completed with {model} in {llm_duration_ms}ms")

        response_content = response.text

        if not response_content:
            raise TranscriptionAgentError(f"Model {model} returned empty response")

        # Parse JSON response
        try:
            # Check if response was truncated
            if response.candidates[0].finish_reason == "MAX_TOKENS":
                logger.warning(f"[Gemini Agent] Response from {model} was truncated due to token limit")

            result = json.loads(response_content)
            logger.info(f"[Gemini Agent] Successfully parsed response from {model}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"[Gemini Agent] Failed to parse response from {model} as JSON: {e}")
            logger.error(f"Response content length: {len(response_content)}")
            logger.error(f"Last 500 chars: {response_content[-500:]}")
            raise TranscriptionAgentError(f"Malformed JSON response from {model}: {e}")

    def _format_to_segments(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the raw transcription result to match the required segment format.
        
        Args:
            raw_result: Raw result from Gemini
            
        Returns:
            Formatted result with transcriptions array
        """
        try:
            # If the response already has the correct format, return as-is
            if "transcriptions" in raw_result:
                return raw_result
            
            # Convert from conversation format to segments format
            transcriptions = []
            
            if "conversation" in raw_result:
                for idx, entry in enumerate(raw_result["conversation"]):
                    transcription_entry = {
                        "segment_id": idx + 1,
                        "start": entry.get("start_time", "00:00:00"),
                        "end": entry.get("end_time", "00:00:00"), 
                        "transcription": entry.get("text", ""),
                        "sentiment": entry.get("sentiment", "neutral")
                    }
                    transcriptions.append(transcription_entry)
            
            return {"transcriptions": transcriptions}
            
        except Exception as e:
            logger.error(f"[Gemini Agent] Failed to format response: {e}")
            # Return minimal valid response
            return {"transcriptions": []}

    async def transcribe_with_prompt(
        self,
        prompt: str,
        audio_file_path: str
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using custom prompt with model fallback.

        Args:
            prompt: Custom transcription prompt
            audio_file_path: Path to the audio file

        Returns:
            Dictionary containing transcriptions array with segment data

        Raises:
            TranscriptionAgentError: If all models fail
        """
        uploaded_file = None
        last_exception = None

        try:
            # Upload file once, reuse for all model attempts
            uploaded_file = await self._upload_audio_file(audio_file_path)

            # Try each model in the fallback chain
            for model_idx, model in enumerate(self.models):
                try:
                    logger.info(f"[Gemini Agent] Attempting transcription with {model} "
                               f"({model_idx + 1}/{len(self.models)})")

                    raw_result = await self._transcribe_with_model(
                        model, prompt, uploaded_file,
                    )
                    formatted_result = self._format_to_segments(raw_result)

                    logger.info(f"[Gemini Agent] Successfully transcribed with {model}")
                    return formatted_result

                except Exception as e:
                    last_exception = e

                    if self._is_retryable_error(e):
                        logger.warning(f"[Gemini Agent] Model {model} failed with retryable error: {e}")
                        if model_idx < len(self.models) - 1:
                            logger.info(f"[Gemini Agent] Falling back to next model: {self.models[model_idx + 1]}")
                            continue
                    else:
                        logger.error(f"[Gemini Agent] Model {model} failed with non-retryable error: {e}")
                        raise TranscriptionAgentError(f"Non-retryable error with {model}: {e}") from e

            # All models failed
            logger.error(f"[Gemini Agent] All models failed. Last error: {last_exception}")
            raise TranscriptionAgentError(f"All models in fallback chain failed. Last error: {last_exception}")

        except Exception as e:
            logger.error(f"[Gemini Agent] Transcription failed: {e}")
            if isinstance(e, TranscriptionAgentError):
                raise
            raise TranscriptionAgentError(f"Transcription failed: {e}") from e

        finally:
            # Clean up uploaded file
            if uploaded_file:
                try:
                    logger.info(f"[Gemini Agent] Cleaning up uploaded file: {uploaded_file.name}")
                    await asyncio.to_thread(self.client.files.delete, name = uploaded_file.name)
                except Exception as cleanup_exc:
                    logger.warning(f"[Gemini Agent] Failed to clean up uploaded file: {cleanup_exc}")


# Convenience function for direct usage
async def transcribe(
    prompt: str,
    audio_file_path: str,
    models: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to transcribe audio with custom prompt.
    
    Args:
        prompt: Custom transcription prompt
        audio_file_path: Path to the audio file
        models: Optional list of models for fallback chain
        
    Returns:
        Dictionary containing transcriptions array
    """
    agent = GeminiTranscriptionAgent(models)
    return await agent.transcribe_with_prompt(prompt, audio_file_path)