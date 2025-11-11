from ast import dump
import asyncio
import logging
import os
import tempfile
import shutil
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path
import json
import base64
from openai import OpenAI
from app.db.mongodb import get_database
from bson import ObjectId
from app.models.meeting import (
    AudioProcessingResult,
    MeetingRecord,
    MeetingParticipant,
    ProcessingStatus,
    TranscriptionResult,
)
from app.schemas.meeting_analysis import MeetingAnalysis
from app.services.transcription_service import TranscriptionService
from app.utils.audio_merger import merge_wav_files_from_s3, AudioMergerError
from app.services.meeting_analysis_service import MeetingAnalysisService
from app.repository import MeetingMetadataRepository
from app.utils.s3_client import download_s3_file

logger = logging.getLogger(__name__)

class MeetingAnalysisOrchestratorError(Exception):
    """Custom exception for meeting analysis orchestrator operations."""
    pass


class MeetingAlreadyProcessedException(Exception):
    """Exception raised when meeting is already processed or being processed."""
    pass

class GeminiService:
    """Service for interacting with Gemini 2.5 Flash via OpenAI compatible API."""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=1200.0  # 20 minutes timeout
        )
        self.model = "gemini-2.5-flash"
    
    def _get_unified_prompt(self) -> str:
        return """You are an intelligent meeting analyzer. Using the inputs, you must:

1. **Diarize** the transcript,
2. Produce a **post-meeting analysis**, and
3. Generate the **Executive Prep Pack** for the next meeting.

Return **one** valid JSON object only (no prose, no comments, no markdown formatting, no code blocks) with **exact** top-level keys:

{
  "diarization": [],
  "meeting_analysis": {},
  "meeting_preparation": {}
}

### Inputs

* `audio` (optional) or `transcript`: list of turns `{index, start_ms, end_ms, text}`
* `meeting_info`: includes `speakerTimeframes: [{speakerName, start, end}]` (ms)
* `attendees` (optional)
* `previous_meetings_context` (optional, ≥3 prior summaries)

---

## A) Diarization → `"diarization"` (array)

For each transcript turn, assign `speakerName` using **max time overlap** with `speakerTimeframes`. If no overlap, use `""`. Do **not** invent names.

Item schema:

{"index": number, "start_ms": number, "end_ms": number, "speakerName": "string", "text": "string"}

---

## B) Post-Meeting Analysis → `"meeting_analysis"` (object)

Schema:

{
  "summary":"string",
  "key_points":["string"],
  "decisions":[{"title":"string","owner":"string|null","due_date":"string|null","references":[int]}],
  "action_items":[{"task":"string","owner":"string|null","due_date":"string|null","priority":"string|null","references":[int]}],
  "risks_issues":[{"description":"string","mitigation":"string|null","severity":"string|null","references":[int]}],
  "open_questions":["string"],
  "topics":["string"],
  "confidence":"low"|"medium"|"high"
}

Rules: Ground in transcript; **summary 5–8 sentences**; use **0-based turn indices** in `references` (or `[]` if none); use **null** for unknown owner/due/priority; required arrays must exist (use `[]`); valid JSON only.

---

## C) Executive Prep Pack → `"meeting_preparation"` (object)

Schema:

{
  "title":"string",
  "timezone":"string",
  "locale":"en-US",
  "purpose":"string",
  "confidence":"low|medium|high",
  "expected_outcomes":[{"description":"string","owner":"email","type":"decision|approval|alignment"}],
  "blocking_items":[{"title":"string","owner":"email","eta":"YYYY-MM-DD","impact":"string","severity":"low|medium|high","status":"open|mitigating|cleared"}],
  "decision_queue":[{"id":"string","title":"string","needs":["string"],"owner":"email"}],
  "key_points":["string"],
  "open_questions":["string"],
  "risks_issues":["string"],
  "leadership_asks":["string"]
}

Rules: Synthesize from `meeting_info` and **≥1 prior meeting** if available (cite prior IDs in bullets when possible). Keep fields **empty strings** if data like IDs/emails are missing. Include blocking items, decision queue (with explicit needs), risks (delivery/budget/people/compliance), and actionable open questions (who to ask, why, what it unblocks). Set `"timezone"` appropriately (e.g., `"Asia/Kolkata"`).

---

## Execution Order
1. If only `audio`, transcribe to get `transcript`.
2. **Diarize** using max-overlap with `speakerTimeframes`.
3. Build `"meeting_analysis"` from the diarized transcript (with `references`).
4. Build `"meeting_preparation"` for the next meeting (use prior context if present).
5. Return the **single** JSON object exactly as specified."""

    async def analyze_meeting_with_audio(
        self, 
        audio_file_path: str, 
        meeting_info: Dict[str, Any],
        attendees: Optional[list] = None,
        previous_meetings_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze meeting using Gemini 2.5 Flash with audio input.
        
        Args:
            audio_file_path: Path to the audio file
            meeting_info: Meeting metadata including speaker timeframes
            attendees: List of attendees (optional)
            previous_meetings_context: Context from previous meetings (optional)
            
        Returns:
            Dictionary containing diarization, meeting_analysis, and meeting_preparation
        """
        try:
            # Read and encode audio file
            with open(audio_file_path, "rb") as audio_file:
                audio_data = base64.b64encode(audio_file.read()).decode()
            
            # Prepare context message
            context_parts = [
                f"meeting_info: {json.dumps(meeting_info)}"
            ]
            
            if attendees:
                context_parts.append(f"attendees: {json.dumps(attendees)}")
            
            if previous_meetings_context:
                context_parts.append(f"previous_meetings_context: {previous_meetings_context}")
            
            context_message = "\n".join(context_parts)
            
            # Make API call to Gemini
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_unified_prompt()
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
            
            # Parse JSON response
            logger.info(f"Gemini response structure: {response}")
            logger.info(f"Response choices: {response.choices}")
            logger.info(f"First choice: {response.choices[0] if response.choices else 'No choices'}")
            
            response_content = response.choices[0].message.content
            logger.info(f"Response content: {response_content}")
            
            if not response_content:
                raise ValueError("Gemini returned empty response content")
            
            # Strip markdown formatting if present
            if response_content.strip().startswith("```json"):
                start_marker = "```json"
                end_marker = "```"
                start_idx = response_content.find(start_marker) + len(start_marker)
                end_idx = response_content.rfind(end_marker)
                response_content = response_content[start_idx:end_idx].strip()
            elif response_content.strip().startswith("```"):
                # Handle generic code block
                lines = response_content.strip().split('\n')
                response_content = '\n'.join(lines[1:-1])
            
            result = json.loads(response_content)
            
            # Validate required keys
            required_keys = ["diarization", "meeting_analysis", "meeting_preparation"]
            if not all(key in result for key in required_keys):
                raise ValueError(f"Missing required keys in response: {required_keys}")
            
            return result
            
        except json.JSONDecodeError as e:
            raise MeetingAnalysisOrchestratorError(f"Failed to parse Gemini response as JSON: {str(e)}")
        except Exception as e:
            raise MeetingAnalysisOrchestratorError(f"Gemini API call failed: {str(e)}")

async def find_upcoming_meeting_by_recurring_id(recurring_meeting_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
    """
    Find the next upcoming meeting for a given recurring meeting ID.
    
    Args:
        recurring_meeting_id: The recurring meeting ID to search for
        tenant_id: The tenant ID for filtering
        
    Returns:
        Dictionary containing the upcoming meeting data or None if not found
    """
    try:
        db = await get_database()
        meetings_collection = db.get_collection("meetings")
        
        # Find upcoming meetings for the recurring meeting ID
        current_time = datetime.now(timezone.utc)
        
        upcoming_meeting = await meetings_collection.find_one({
            "recurring_meeting_id": recurring_meeting_id,
            "tenantId": tenant_id,
            "start_time": {"$gt": current_time.isoformat()},
            "status": {"$in": ["scheduled", "pending"]}
        }, sort=[("start_time", 1)])  # Get the earliest upcoming meeting
        
        return upcoming_meeting
        
    except Exception as e:
        logger.error(f"Failed to find upcoming meeting for recurring_id {recurring_meeting_id}: {e}")
        return None

async def save_meeting_preparation_directly(
    meeting_id: str,
    preparation_data: Dict[str, Any],
    source_meeting_id: str,
    tenant_id: str
) -> Optional[str]:
    """
    Save meeting preparation data directly to the meeting_preparations collection.
    
    Args:
        meeting_id: The target meeting ID to save preparation for
        preparation_data: The preparation data from Gemini
        source_meeting_id: The source meeting ID that generated this preparation
        tenant_id: The tenant ID
        
    Returns:
        The inserted document ID or None if failed
    """
    try:
        db = await get_database()
        preparations_collection = db.get_collection("meeting_preparations")
        
        prep_document = {
            "meeting_id": meeting_id,
            "source_meeting_id": source_meeting_id,
            "tenant_id": tenant_id,
            "preparation_data": preparation_data,
            "created_at": datetime.now(timezone.utc),
            "status": "active"
        }
        
        result = await preparations_collection.insert_one(prep_document)
        logger.info(f"Saved meeting preparation for meeting {meeting_id}: {result.inserted_id}")
        
        return str(result.inserted_id)
        
    except Exception as e:
        logger.error(f"Failed to save meeting preparation for meeting {meeting_id}: {e}")
        return None

class MeetingAnalysisOrchestrator:
    """Main service for orchestrating meeting processing pipeline."""
    
    def __init__(self):
        self.gemini_service = GeminiService()
        self.meeting_metadata_repo = None  # Will be initialized when needed
    
    async def analyze_meeting(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process meeting event using Gemini 2.5 Flash for unified analysis.
        
        Args:
            payload: Dictionary containing the meeting event with payload
            
        Returns:
            Dictionary containing processing results
        """
        try:
            logger.info("Starting meeting event processing with Gemini")
            if not payload:
                raise MeetingAnalysisOrchestratorError("No payload found in event_data")
            
            logger.info(f"Processing meeting: {payload.get('summary', 'Unknown')}")
            
            # Extract meeting details from payload
            meeting_id = str(payload.get('_id'))
            tenant_id = payload.get('tenantId')
            platform = payload.get('platform')
            bucket = payload.get('bucket', 'recordings')
            recurring_meeting_id = payload.get('recurring_meeting_id')
            
            if not all([meeting_id, tenant_id]):
                raise MeetingAnalysisOrchestratorError("Missing required fields in payload: _id, tenantId")
            
            # Update meeting status to analyzing at start of processing
            await self._update_meeting_status(meeting_id, platform, 'analysing')
            
            # Step 1: Prepare audio file
            logger.info("Step 1: Preparing audio file")
            if not payload.get('fileUrl'):
                s3_folder_path = f"{tenant_id}/{platform}/{meeting_id}/"
                output_s3_key = f"{tenant_id}/{platform}/{meeting_id}/meeting.wav"

                merge_result = await merge_wav_files_from_s3(
                    s3_folder_path=s3_folder_path,
                    output_s3_key=output_s3_key,
                    bucket_name=bucket
                )
                file_url = merge_result['local_merged_file_path']
            else:
                temp_dir = tempfile.mkdtemp(prefix="audio_merge_")
                file_url = os.path.join(temp_dir, "merged_output.wav")
                await download_s3_file(payload.get('fileUrl'), file_url, bucket)
                merge_result = {
                    'local_merged_file_path': file_url,
                    'temp_directory': temp_dir
                }
            
            # Step 2: Prepare meeting info and context
            logger.info("Step 2: Preparing meeting context")
            meeting_info = {
                "meeting_id": meeting_id,
                "platform": platform,
                "speakerTimeframes": []
            }
            
            # Get speaker timeframes for Google meetings
            if platform == "google":
                if not self.meeting_metadata_repo:
                    self.meeting_metadata_repo = await MeetingMetadataRepository.from_default()
                
                speaker_timeframes = await self.meeting_metadata_repo.get_speaker_timeframes(meeting_id)
                if speaker_timeframes:
                    meeting_info["speakerTimeframes"] = speaker_timeframes
                    logger.info(f"Found {len(speaker_timeframes)} speaker timeframes for Google meeting")
            
            # Step 3: Analyze with Gemini
            logger.info("Step 3: Analyzing meeting with Gemini 2.5 Flash")
            gemini_result = await self.gemini_service.analyze_meeting_with_audio(
                audio_file_path=file_url,
                meeting_info=meeting_info,
                attendees=payload.get('attendees'),
                previous_meetings_context=None  # TODO: Add previous meetings context
            )
            
            # Step 4: Save results to existing collections
            logger.info("Step 4: Saving results to collections")
            
            # Save transcription/diarization
            diarization = gemini_result.get('diarization', [])
            if diarization:
                await save_transcription(meeting_id, tenant_id, diarization)
            
            # Save meeting analysis
            analysis_result = gemini_result.get('meeting_analysis', {})
            if analysis_result:
                # Add required fields for compatibility
                analysis_result.update({
                    "meeting_id": meeting_id,
                    "session_id": meeting_id,
                    "tenant_id": tenant_id,
                    "platform": platform,
                    "processed_at": datetime.now(timezone.utc).isoformat()
                })
                
                # Convert to Pydantic model
                meeting_analysis_model = MeetingAnalysis(**analysis_result)
                
                meeting_analysis_service = await MeetingAnalysisService.from_default()
                await meeting_analysis_service.save_analysis(meeting_analysis_model)

            # Step 5: Save meeting preparation (conditional)
            logger.info("Step 5: Processing meeting preparation")
            prep_pack = None
            prep_result = gemini_result.get('meeting_preparation', {})
            
            if recurring_meeting_id and prep_result:
                # Add required fields for compatibility
                prep_result.update({
                    "meeting_id": meeting_id,
                    "recurring_meeting_id": recurring_meeting_id,
                    "tenant_id": tenant_id,
                    "platform": platform,
                    "created_at": datetime.now(timezone.utc).isoformat()
                })
                
                meeting_prep_service = await MeetingPrepCuratorService.from_default()
                prep_pack = await meeting_prep_service.generate_and_save_prep_pack(
                    meeting_id=meeting_id,
                    meeting_analysis=prep_result,
                    platform=platform
                )
            elif not recurring_meeting_id:
                logger.info("Skipping meeting preparation - no recurring_meeting_id provided")
            
            # Prepare return result
            merge_result["analysis"] = analysis_result
            merge_result["prep_pack"] = prep_pack
            merge_result["diarization"] = diarization
            merge_result["success"] = True
            merge_result["meeting_id"] = meeting_id
            
            # Update meeting status to completed on success
            await self._update_meeting_status(meeting_id, platform, 'completed')
            
            return merge_result

        except Exception as e:
            logger.error(f"Meeting processing failed for meeting {payload.get('_id', 'unknown')}: {str(e)}")
            
            # Update meeting status to failed on error
            meeting_id = payload.get('_id')
            platform = payload.get('platform')
            if meeting_id and platform:
                await self._update_meeting_status(str(meeting_id), platform, 'failed')
            
            raise MeetingAnalysisOrchestratorError(f"Meeting processing failed: {str(e)}") from e

    async def _update_meeting_status(self, meeting_id: str, platform: str, status: str) -> None:
        """Update meeting status in metadata repository for supported platforms."""
        try:
            if platform == "google":
                if not self.meeting_metadata_repo:
                    self.meeting_metadata_repo = await MeetingMetadataRepository.from_default()
                
                await self.meeting_metadata_repo.update_meeting_status(meeting_id, platform, status)
                logger.info(f"Updated meeting {meeting_id} status to {status} for platform {platform}")
            else:
                logger.info(f"Status update not supported for platform {platform}, meeting {meeting_id} would be {status}")
        except Exception as e:
            logger.warning(f"Failed to update meeting status to {status} for meeting {meeting_id}: {e}")

async def save_transcription(meeting_id, tenant_id, vox_scribe_result):
    try:
        transcription_service = await TranscriptionService.from_default()
        
        transcription_result = await transcription_service.save_transcription(
            meeting_id=meeting_id,
            tenant_id=tenant_id,
            conversation=vox_scribe_result,
            processing_metadata={
                "vox_scribe_version": "1.0",
                "known_speakers": 0,
                "audio_duration_seconds": None
            }
        )
        logger.info(f"Saved transcription to MongoDB: {transcription_result}")
        
    except Exception as e:
        logger.error(f"Failed to save transcription: {e}")
