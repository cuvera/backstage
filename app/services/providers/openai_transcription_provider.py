import base64
import json
import logging
import time
from typing import Any, Dict, List

from app.core.openai_client import llm_client
from app.core.prompts import TRANSCRIPTION_AND_SENTIMENT_ANALYSIS_PROMPT
from .base_transcription_provider import BaseTranscriptionProvider, TranscriptionProviderError

logger = logging.getLogger(__name__)


class OpenAITranscriptionProvider(BaseTranscriptionProvider):
    """
    OpenAI-compatible transcription provider using base64 audio encoding.
    
    This provider uses the existing OpenAI client with base64 encoded audio data.
    """

    def __init__(self, client=None, model="gemini-2.5-pro"):
        """
        Initialize OpenAI transcription provider.
        
        Args:
            client: OpenAI client instance (defaults to llm_client)
            model: Model to use for transcription
        """
        self.client = client or llm_client
        self.model = model

    async def transcribe(
        self, 
        audio_file_path: str, 
        meeting_metadata: Dict[str, Any],
        participants: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using OpenAI-compatible client with base64 encoding.
        """
        try:
            logger.info(f"[OpenAI Provider] Starting transcription for: {audio_file_path}")
            
            # Read and encode audio file
            with open(audio_file_path, "rb") as audio_file:
                audio_data = base64.b64encode(audio_file.read()).decode()
            
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
            logger.info("[OpenAI Provider] Starting LLM call")
            llm_start_time = time.time()

            response = await self.client.chat.completions.create(
                model=self.model,
                reasoning_effort="low",
                messages=[
                    {
                        "role": "system",
                        "content": TRANSCRIPTION_AND_SENTIMENT_ANALYSIS_PROMPT
                    },
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": context_message
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_data,
                                    "format": "wav"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            llm_duration_ms = round((time.time() - llm_start_time) * 1000, 2)
            logger.info(f"[OpenAI Provider] LLM call completed in {llm_duration_ms}ms")

            response_content = response.choices[0].message.content
            
            if not response_content:
                raise TranscriptionProviderError("OpenAI returned empty response content")
            
            # Parse and validate JSON response
            result = {}
            try:
                result = json.loads(response_content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI response as JSON: {e}")
                raise TranscriptionProviderError(f"Malformed JSON response: {e}")

            # Validate required fields
            required_fields = ["conversation", "total_speakers", "sentiments"]
            for field in required_fields:
                if field not in result:
                    raise TranscriptionProviderError(f"Missing required field in response: {field}")
            
            logger.info(f"[OpenAI Provider] Transcription completed. Found {result.get('total_speakers', 0)} speakers")
            return result
            
        except Exception as exc:
            logger.exception(f"[OpenAI Provider] Transcription failed for {audio_file_path}: {exc}")
            if isinstance(exc, TranscriptionProviderError):
                raise
            raise TranscriptionProviderError(f"OpenAI transcription processing failed: {exc}") from exc