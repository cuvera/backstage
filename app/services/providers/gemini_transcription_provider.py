import json
import logging
import time
import os
from typing import Any, Dict, List

from google import genai
from google.genai import types
from app.core.config import settings
from app.core.prompts import TRANSCRIPTION_AND_SENTIMENT_ANALYSIS_PROMPT
from .base_transcription_provider import BaseTranscriptionProvider, TranscriptionProviderError

logger = logging.getLogger(__name__)


class GeminiTranscriptionProvider(BaseTranscriptionProvider):
    """
    Gemini native transcription provider using file upload API.
    
    This provider uploads audio files to Gemini's Files API and processes them
    without base64 encoding limitations.
    """

    def __init__(self, model="gemini-2.5-flash"):
        """
        Initialize Gemini transcription provider.
        
        Args:
            model: Gemini model to use for transcription
        """
        if not settings.GEMINI_API_KEY:
            raise TranscriptionProviderError("GEMINI_API_KEY is required for Gemini provider")
        
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = model

    def _attempt_json_recovery(self, truncated_json: str) -> str:
        """
        Attempt to recover a valid JSON from a truncated response.
        
        Args:
            truncated_json: The truncated JSON string
            
        Returns:
            A potentially valid JSON string
        """
        try:
            # Find the last complete conversation entry
            last_complete_entry = truncated_json.rfind('    }')
            if last_complete_entry == -1:
                # No complete entry found, return minimal valid JSON
                return '{"conversation": [], "total_speakers": 0, "sentiments": {}}'
            
            # Truncate to the last complete entry and close the JSON properly
            recovered = truncated_json[:last_complete_entry + 5]  # Include the closing brace
            
            # Close the conversation array and main object
            if not recovered.rstrip().endswith(']'):
                recovered += '\n  ],\n  "total_speakers": 0,\n  "sentiments": {}\n}'
            else:
                recovered += ',\n  "total_speakers": 0,\n  "sentiments": {}\n}'
                
            return recovered
            
        except Exception:
            # If recovery fails, return minimal valid JSON
            return '{"conversation": [], "total_speakers": 0, "sentiments": {}}'

    async def transcribe(
        self, 
        audio_file_path: str, 
        meeting_metadata: Dict[str, Any],
        participants: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using Gemini file upload API.
        """
        uploaded_file = None
        try:
            logger.info(f"[Gemini Provider] Starting transcription for: {audio_file_path}")
            
            # Check if file exists
            if not os.path.exists(audio_file_path):
                raise TranscriptionProviderError(f"Audio file not found: {audio_file_path}")
            
            # Upload file to Gemini
            logger.info("[Gemini Provider] Uploading audio file")
            upload_start_time = time.time()
            
            uploaded_file = self.client.files.upload(
                file=audio_file_path,
                # display_name=f"audio_{int(time.time())}"
            )
            
            # Wait for file processing
            while uploaded_file.state == "PROCESSING":
                logger.info("[Gemini Provider] File processing, waiting...")
                time.sleep(1)
                uploaded_file = self.client.files.get(uploaded_file.name)
            
            if uploaded_file.state == "FAILED":
                raise TranscriptionProviderError("File upload to Gemini failed")
            
            upload_duration_ms = round((time.time() - upload_start_time) * 1000, 2)
            logger.info(f"[Gemini Provider] File uploaded and processed in {upload_duration_ms}ms")
            
            # Prepare context message
            context_parts = [
                f"meeting_metadata: {json.dumps(meeting_metadata, default=str)}"
            ]
            
            # Include speaker timeframes if available
            if meeting_metadata.get('speaker_timeframes'):
                context_parts.append(f"speaker_timeframes: {json.dumps(meeting_metadata['speaker_timeframes'], default=str)}")
            
            # Include participant information if available
            if participants:
                context_parts.append(f"participants: {json.dumps(participants)}")

            context_message = "\n".join(context_parts)
            
            # Make API call with timing
            logger.info("[Gemini Provider] Starting LLM call")
            llm_start_time = time.time()

            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    TRANSCRIPTION_AND_SENTIMENT_ANALYSIS_PROMPT,
                    context_message,
                    uploaded_file
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    # thinking_config=types.ThinkingConfig(
                    #     thinking_budget=128
                    # )
                )
            )

            llm_duration_ms = round((time.time() - llm_start_time) * 1000, 2)
            logger.info(f"[Gemini Provider] LLM call completed in {llm_duration_ms}ms")

            response_content = response.text
            
            if not response_content:
                raise TranscriptionProviderError("Gemini returned empty response content")
            
            # Parse and validate JSON response
            result = {}
            try:
                # Check if response was truncated
                if response.candidates[0].finish_reason == "MAX_TOKENS":
                    logger.warning("[Gemini Provider] Response was truncated due to token limit")
                    # Try to salvage partial JSON by finding the last complete entry
                    response_content = self._attempt_json_recovery(response_content)
                
                print("#"*20)
                print(response_content)
                result = json.loads(response_content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response as JSON: {e}")
                # Log the problematic content for debugging
                logger.error(f"Response content length: {len(response_content)}")
                logger.error(f"Last 500 chars: {response_content[-500:]}")
                raise TranscriptionProviderError(f"Malformed JSON response: {e}")

            # Validate required fields
            required_fields = ["conversation", "total_speakers", "sentiments"]
            for field in required_fields:
                if field not in result:
                    raise TranscriptionProviderError(f"Missing required field in response: {field}")
            
            logger.info(f"[Gemini Provider] Transcription completed. Found {result.get('total_speakers', 0)} speakers")
            return result
            
        except Exception as exc:
            logger.exception(f"[Gemini Provider] Transcription failed for {audio_file_path}: {exc}")
            if isinstance(exc, TranscriptionProviderError):
                raise
            raise TranscriptionProviderError(f"Gemini transcription processing failed: {exc}") from exc
        
        finally:
            # Clean up uploaded file
            if uploaded_file:
                try:
                    logger.info(f"[Gemini Provider] Cleaning up uploaded file: {uploaded_file.name}")
                    self.client.files.delete(uploaded_file.name)
                except Exception as cleanup_exc:
                    logger.warning(f"[Gemini Provider] Failed to clean up uploaded file: {cleanup_exc}")