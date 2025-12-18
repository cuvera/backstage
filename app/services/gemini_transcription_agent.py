import asyncio
import json
import logging
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from app.core.config import settings

logger = logging.getLogger(__name__)


class TranscriptionAgentError(Exception):
    """Custom exception for transcription agent operations."""
    pass


class TruncationError(TranscriptionAgentError):
    """Exception raised when response is truncated due to token limits."""
    pass


class EmptyResponseError(TranscriptionAgentError):
    """Exception raised when model returns empty response."""
    pass


class GeminiTranscriptionAgent:
    """
    Gemini transcription agent with model fallback capability.
    
    Takes a prompt and audio file, uploads to Gemini, and generates
    transcriptions with segment-level output and sentiment analysis.
    """

    def __init__(self, models: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize Gemini transcription agent.

        Args:
            models: List of model configs with 'model', 'timeout', 'max_tokens'
                    Example: [
                        {"model": "gemini-2.0-flash-exp", "timeout": 180, "max_tokens": 20000},
                        {"model": "gemini-2.5-pro", "timeout": 300, "max_tokens": 65535}
                    ]
        """
        if not settings.GEMINI_API_KEY:
            raise TranscriptionAgentError("GEMINI_API_KEY is required for Gemini agent")

        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

        # Default configs if none provided
        self.models = models or [
            {"model": "gemini-2.5-flash", "timeout": 90, "max_tokens": 20000},
            {"model": "gemini-2.5-flash", "timeout": 180, "max_tokens": 24000},
            {"model": "gemini-2.5-pro", "timeout": 300, "max_tokens": 65535}
        ]

        model_names = [m["model"] for m in self.models]
        logger.info(f"[Gemini Agent] Initialized with fallback chain: {' -> '.join(model_names)}")

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error should trigger fallback to next model.

        Args:
            error: The exception that occurred

        Returns:
            True if error is retryable, False otherwise
        """
        # Empty response errors - retry with fallback model
        if isinstance(error, EmptyResponseError):
            return True

        # Token limit truncation errors - retry with more capable model
        if isinstance(error, TruncationError):
            return True

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

    def _get_mime_type(self, file_path: str) -> str:
        """
        Get MIME type for audio file based on extension.

        Args:
            file_path: Path to the audio file

        Returns:
            MIME type string
        """
        extension = Path(file_path).suffix.lower()
        mime_types = {
            '.m4a': 'audio/mp4',
            '.mp4': 'audio/mp4',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.flac': 'audio/flac',
            '.ogg': 'audio/ogg',
            '.aac': 'audio/aac',
            '.webm': 'audio/webm',
        }
        return mime_types.get(extension, 'audio/mpeg')  # Default to audio/mpeg

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

            # Determine MIME type based on file extension
            mime_type = self._get_mime_type(audio_file_path)
            logger.info(f"[Gemini Agent] Using MIME type: {mime_type}")

            uploaded_file = await asyncio.to_thread(
                self.client.files.upload,
                file=audio_file_path,
                config=types.UploadFileConfig(mime_type=mime_type)
            )
            
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
        model_config: Dict[str, Any],
        prompt: str,
        uploaded_file
    ) -> Dict[str, Any]:
        """
        Transcribe audio using specific Gemini model.

        Args:
            model_config: Dict with 'model', 'timeout', 'max_tokens'
            prompt: Custom transcription prompt
            uploaded_file: Uploaded file object

        Returns:
            Transcription result dictionary
        """
        model = model_config["model"]
        timeout_seconds = model_config.get("timeout", 180)
        max_tokens = model_config.get("max_tokens", 20000)

        logger.info(f"[Gemini Agent] Starting transcription with model: {model} "
                    f"(timeout: {timeout_seconds}s, max_tokens: {max_tokens})")
        llm_start_time = time.time()

        try:
            # Wrap generate_content with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model,
                    contents=[
                        prompt,
                        uploaded_file
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                        max_output_tokens=max_tokens
                    )
                ),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning(f"[Gemini Agent] Model {model} timed out after {timeout_seconds}s")
            raise EmptyResponseError(f"Model {model} timed out after {timeout_seconds}s")

        llm_duration_ms = round((time.time() - llm_start_time) * 1000, 2)
        logger.info(f"[Gemini Agent] Transcription completed with {model} in {llm_duration_ms}ms")

        response_content = response.text

        if not response_content:
            logger.warning(f"[Gemini Agent] Model {model} returned empty response")
            raise EmptyResponseError(f"Model {model} returned empty response")

        # Check if response was truncated before parsing
        is_truncated = response.candidates[0].finish_reason == "MAX_TOKENS"
        if is_truncated:
            logger.warning(f"[Gemini Agent] Response from {model} was truncated due to token limit")

        # Parse JSON response
        try:
            result = json.loads(response_content)

            # If truncated but JSON is valid, log warning but continue
            if is_truncated:
                logger.warning(f"[Gemini Agent] Truncated response from {model} was still valid JSON")

            # Log the structure of the response for debugging
            transcriptions_count = 0
            if "transcriptions" in result:
                transcriptions_count = len(result.get("transcriptions", []))
            elif "conversation" in result:
                transcriptions_count = len(result.get("conversation", []))

            logger.info(f"[Gemini Agent] Successfully parsed response from {model} - {transcriptions_count} segments found")

            # Warn if response is empty
            if transcriptions_count == 0:
                logger.warning(f"[Gemini Agent] Gemini returned valid JSON but with 0 transcriptions/conversation entries")
                logger.warning(f"[Gemini Agent] Response keys: {list(result.keys())}")
                logger.warning(f"[Gemini Agent] Full response: {json.dumps(result, indent=2)}")

            return result

        except json.JSONDecodeError as e:
            # JSON parse errors are retryable - try fallback models
            logger.error(f"[Gemini Agent] Failed to parse response from {model} as JSON: {e}")
            logger.error(f"Response content length: {len(response_content)}")
            logger.error(f"Last 500 chars: {response_content[-500:]}")

            if is_truncated:
                logger.error(f"[Gemini Agent] Response was truncated (MAX_TOKENS)")
                raise TruncationError(f"Response truncated and malformed from {model}: {e}")
            else:
                # Malformed JSON without truncation - retryable for fallback
                raise EmptyResponseError(f"Malformed JSON response from {model}: {e}")

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
            for model_idx, model_config in enumerate(self.models):
                try:
                    model_name = model_config["model"]
                    logger.info(f"[Gemini Agent] Attempting transcription with {model_name} "
                               f"({model_idx + 1}/{len(self.models)})")

                    raw_result = await self._transcribe_with_model(
                        model_config, prompt, uploaded_file
                    )
                    formatted_result = self._format_to_segments(raw_result)

                    logger.info(f"[Gemini Agent] Successfully transcribed with {model_name}")
                    return formatted_result

                except Exception as e:
                    last_exception = e
                    model_name = model_config["model"]

                    if self._is_retryable_error(e):
                        logger.warning(f"[Gemini Agent] Model {model_name} failed with retryable error: {e}")
                        if model_idx < len(self.models) - 1:
                            next_model = self.models[model_idx + 1]["model"]
                            logger.info(f"[Gemini Agent] Falling back to next model: {next_model}")
                            continue
                    else:
                        logger.error(f"[Gemini Agent] Model {model_name} failed with non-retryable error: {e}")
                        raise TranscriptionAgentError(f"Non-retryable error with {model_name}: {e}") from e

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
    models: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Convenience function to transcribe audio with custom prompt.

    Args:
        prompt: Custom transcription prompt
        audio_file_path: Path to the audio file
        models: Optional list of model configs
                Example: [
                    {"model": "gemini-2.0-flash-exp", "timeout": 180, "max_tokens": 20000},
                    {"model": "gemini-1.5-pro", "timeout": 300, "max_tokens": 65535}
                ]

    Returns:
        Dictionary containing transcriptions array
    """
    agent = GeminiTranscriptionAgent(models)
    return await agent.transcribe_with_prompt(prompt, audio_file_path)