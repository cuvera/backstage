from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, Optional

from app.core.openai_client import llm_client
from app.core.prompts import TRANSCRIPTION_AND_SENTIMENT_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


class TranscriptionAgentError(Exception):
    """Raised when the transcription agent cannot complete its task."""


class TranscriptionAgent:
    """
    Transcription agent that processes audio files and returns structured transcription
    data with sentiment analysis using Gemini multimodal capabilities.
    
    Uses the same audio processing pattern as MeetingAnalysisOrchestrator.
    """

    def __init__(self, client=None, model="gemini-exp-1206"):
        """
        Initialize TranscriptionAgent.
        
        Args:
            client: OpenAI client instance (defaults to llm_client)
            model: Gemini model to use for transcription
        """
        self.client = client or llm_client
        self.model = model

    async def transcribe(
        self, 
        audio_file_path: str, 
        meeting_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transcribe audio file and perform sentiment analysis.
        
        Args:
            audio_file_path: Path to the audio file to transcribe
            meeting_metadata: Meeting metadata including speaker timeframes
            
        Returns:
            Dictionary containing:
            - conversation: List of transcription entries with speaker diarization
            - total_speakers: Number of unique speakers detected
            - sentiments: Overall and per-participant sentiment analysis
            
        Raises:
            TranscriptionAgentError: If transcription fails
        """
        try:
            logger.info(f"Starting transcription for audio file: {audio_file_path}")
            
            # Read and encode audio file (copied from orchestrator lines 194-195)
            with open(audio_file_path, "rb") as audio_file:
                audio_data = base64.b64encode(audio_file.read()).decode()
            
            # Prepare context message (similar to orchestrator lines 198-208)
            context_parts = [
                f"meeting_metadata: {json.dumps(meeting_metadata)}"
            ]
            
            # Include speaker timeframes if available
            if meeting_metadata.get('speakerTimeframes'):
                context_parts.append(f"speakerTimeframes: {json.dumps(meeting_metadata['speakerTimeframes'])}")
            
            context_message = "\n".join(context_parts)
            
            # Make API call to Gemini (copied from orchestrator lines 211-237)
            response = await self.client.chat.completions.parse(
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
                temperature=0.1
            )
            
            # Parse JSON response (copied from orchestrator lines 244-249)
            logger.info(f"Gemini response structure: {response}")
            logger.info(f"Response choices: {response.choices}")
            logger.info(f"First choice: {response.choices[0] if response.choices else 'No choices'}")
            
            response_content = response.choices[0].message.content
            logger.info(f"Response content: {response_content}")
            
            if not response_content:
                raise TranscriptionAgentError("Gemini returned empty response content")
            
            # Parse and validate JSON response
            try:
                result = json.loads(response_content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response as JSON: {e}")
                raise TranscriptionAgentError(f"Invalid JSON response from Gemini: {e}") from e
            
            # Validate required fields
            required_fields = ["conversation", "total_speakers", "sentiments"]
            for field in required_fields:
                if field not in result:
                    raise TranscriptionAgentError(f"Missing required field in response: {field}")
            
            logger.info(f"Transcription completed successfully. Found {result.get('total_speakers', 0)} speakers")
            return result
            
        except Exception as exc:
            logger.exception(f"Transcription failed for audio file {audio_file_path}: {exc}")
            if isinstance(exc, TranscriptionAgentError):
                raise
            raise TranscriptionAgentError(f"Transcription processing failed: {exc}") from exc