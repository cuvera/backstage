import logging
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core.config import settings
from app.core.openai_client import llm_client
from app.repository.meeting_vector_repository import MeetingVectorRepository
from app.repository.transcription_v1_repository import TranscriptionV1Repository
from app.repository.meeting_analysis_repository import MeetingAnalysisRepository
from qdrant_client import models

logger = logging.getLogger(__name__)

class MeetingEmbeddingService:
    """
    Production-grade service for meeting transcript embedding and storage.
    Uses the global OpenAI-compatible client pointed at Gemini.
    """

    def __init__(self):
        self.embedding_model = "models/embedding-001"
        self.meeting_vector_repo = MeetingVectorRepository()
        self._transcription_repo: Optional[TranscriptionV1Repository] = None
        self._analysis_repo: Optional[MeetingAnalysisRepository] = None

    async def _get_transcription_repo(self) -> TranscriptionV1Repository:
        if self._transcription_repo is None:
            self._transcription_repo = await TranscriptionV1Repository.from_default()
        return self._transcription_repo

    async def _get_analysis_repo(self) -> MeetingAnalysisRepository:
        if self._analysis_repo is None:
            self._analysis_repo = await MeetingAnalysisRepository.from_default()
        return self._analysis_repo

    def _chunk_transcript(self, transcript_segments: List[Dict[str, Any]], target_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Chunks transcript segments into overlapping windows.
        Extracts unique speakers for each chunk.
        """
        chunks = []
        current_chunk_text = ""
        current_chunk_segments = []
        
        # Sort segments by start time
        sorted_segments = sorted(transcript_segments, key=lambda x: x.get("start", "00:00"))
        
        for segment in sorted_segments:
            text = segment.get("transcription", "") or segment.get("text", "")
            speaker = segment.get("speaker", "Unknown")
            start_time = segment.get("start", "00:00")
            formatted_text = f"[{start_time}] {speaker}: {text}\n"
            
            if len(current_chunk_text) + len(formatted_text) > target_size and current_chunk_text:
                # Extract unique speakers for the current chunk
                speakers = list(set(s.get("speaker", "Unknown") for s in current_chunk_segments))
                
                chunks.append({
                    "text": current_chunk_text.strip(),
                    "segments": current_chunk_segments,
                    "speakers": speakers,
                    "startTime": current_chunk_segments[0].get("start", "00:00"),
                    "endTime": current_chunk_segments[-1].get("end", "00:00")
                })
                
                # Overlap: keep context
                overlap_text = ""
                overlap_segments = []
                for s in reversed(current_chunk_segments):
                    s_text = f"[{s.get('start', '00:00')}] {s.get('speaker', 'Unknown')}: {s.get('transcription', '') or s.get('text', '')}\n"
                    if len(overlap_text) + len(s_text) < overlap:
                        overlap_text = s_text + overlap_text
                        overlap_segments.insert(0, s)
                    else:
                        break
                
                current_chunk_text = overlap_text
                current_chunk_segments = overlap_segments
            
            current_chunk_text += formatted_text
            current_chunk_segments.append(segment)
            
        if current_chunk_text:
            speakers = list(set(s.get("speaker", "Unknown") for s in current_chunk_segments))
            chunks.append({
                "text": current_chunk_text.strip(),
                "segments": current_chunk_segments,
                "speakers": speakers,
                "startTime": current_chunk_segments[0].get("start", "00:00"),
                "endTime": current_chunk_segments[-1].get("end", "00:00")
            })
            
        return chunks

    async def generate_embeddings_batch(self, texts: List[str], title: Optional[str] = None) -> List[List[float]]:
        """
        Generates embeddings for a batch of texts using the OpenAI client (pointed at Gemini).
        Note: task_type and title are removed as they are not supported by the current OpenAI shim.
        """
        vectors = []
        for text in texts:
            try:
                response = await llm_client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                vectors.append(response.data[0].embedding)
            except Exception as e:
                logger.error(f"Error generating embedding via openai_client: {e}")
                raise
        return vectors

    async def process_meeting_for_rag(
        self, 
        meeting_id: str, 
        tenant_id: str, 
        transcript: Optional[Dict[str, Any]] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Processes transcript into semantic chunks and stores in Qdrant.
        Consolidates transcript and analysis (summary, decisions) in one collection.
        """
        logger.info(f"[MeetingEmbeddingService] Embedding meeting {meeting_id} (Model: {self.embedding_model})")
        
        # Fetch transcription if not provided
        if not transcript:
            trans_repo = await self._get_transcription_repo()
            transcript = await trans_repo.get_transcription(meeting_id, tenant_id)
            if not transcript:
                logger.error(f"[MeetingEmbeddingService] Transcription not found for {meeting_id}")
                return

        # Fetch analysis/metadata if not provided
        if not metadata:
            analysis_repo = await self._get_analysis_repo()
            metadata = await analysis_repo.get_analysis(tenant_id=tenant_id, session_id=meeting_id)
            if not metadata:
                logger.warning(f"[MeetingEmbeddingService] Analysis/Metadata not found for {meeting_id}")

        segments = transcript.get("transcriptions") or transcript.get("conversation") or []
        meeting_title = (metadata or {}).get("summary") or (metadata or {}).get("title") or "Meeting Transcription"

        # 1. Prepare items to embed
        # We consolidate: Chunks + Analysis (Summary, Decisions, Key Points)
        embedding_items = [] # List of {text: str, type: str, metadata: dict}
        
        # A. Transcript Chunks
        if segments:
            chunks = self._chunk_transcript(segments)
            for i, chunk in enumerate(chunks):
                embedding_items.append({
                    "text": chunk["text"],
                    "type": "transcript_chunk",
                    "chunk_idx": i,
                    "speakers": chunk["speakers"],
                    "startTime": chunk["startTime"],
                    "endTime": chunk["endTime"]
                })

        # B. Analysis Items (if available)
        if metadata:
            # Summary
            summary = metadata.get("summary")
            if summary:
                embedding_items.append({
                    "text": f"Meeting Summary: {summary}",
                    "type": "meeting_summary"
                })
            
            # Decisions
            decisions = metadata.get("decisions", [])
            for i, dec in enumerate(decisions):
                dec_text = dec if isinstance(dec, str) else dec.get("title", "")
                if dec_text:
                    owner = dec.get("owner") if isinstance(dec, dict) else None
                    text = f"Meeting Decision: {dec_text}" + (f" (Owner: {owner})" if owner else "")
                    embedding_items.append({
                        "text": text,
                        "type": "meeting_decision",
                        "idx": i
                    })
            
            # Key Points
            key_points = metadata.get("key_points", [])
            for i, kp in enumerate(key_points):
                if kp:
                    embedding_items.append({
                        "text": f"Key Point: {kp}",
                        "type": "meeting_key_point",
                        "idx": i
                    })
            
            # Action Items
            action_items = metadata.get("action_items", [])
            for i, ai in enumerate(action_items):
                ai_text = ai if isinstance(ai, str) else ai.get("task", "")
                if ai_text:
                    owner = ai.get("owner") if isinstance(ai, dict) else None
                    text = f"Action Item: {ai_text}" + (f" (Owner: {owner})" if owner else "")
                    embedding_items.append({
                        "text": text,
                        "type": "meeting_action_item",
                        "idx": i
                    })
            
            # Agenda (from call_scoring)
            call_scoring = metadata.get("call_scoring")
            if call_scoring and call_scoring.get("identified_agenda"):
                embedding_items.append({
                    "text": f"Identified Agenda: {call_scoring['identified_agenda']}",
                    "type": "meeting_agenda"
                })

        if not embedding_items:
            logger.warning(f"[MeetingEmbeddingService] No items to embed for {meeting_id}")
            return

        # 2. Embedding Generation in batches
        batch_size = 10
        all_texts = [item["text"] for item in embedding_items]
        all_vectors = []
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i : i + batch_size]
            vectors = await self.generate_embeddings_batch(batch_texts, title=meeting_title)
            all_vectors.extend(vectors)

        # 3. Preparation for Qdrant
        points = []
        from datetime import timezone
        created_at = datetime.now(timezone.utc).isoformat()
        
        for idx, (item, vector) in enumerate(zip(embedding_items, all_vectors)):
            # Create a deterministic ID
            import hashlib
            content_to_hash = f"{item['text']}|{item.get('chunk_idx', idx)}"
            checksum = hashlib.sha1(content_to_hash.encode()).hexdigest()
            deterministic_id_input = f"{meeting_id}|{item['type']}|{checksum}"
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, deterministic_id_input))

            payload = {
                "tenantId": tenant_id,
                "meetingId": meeting_id,
                "meeting_text": item["text"],
                "contentType": item["type"],
                "checksum": checksum,
                "meta": {
                    "embedding_model": self.embedding_model,
                    "embedding_dim": len(vector),
                    "schema_version": 2,
                    "created_at": created_at,
                    "source": "backstage-meeting-pipeline",
                    "meeting_title": meeting_title
                }
            }
            
            # Add type-specific metadata
            if item["type"] == "transcript_chunk":
                payload["meeting_chunk_index"] = item["chunk_idx"]
                payload["speakers"] = item["speakers"]
                payload["startTime"] = item["startTime"]
                payload["endTime"] = item["endTime"]
            elif "idx" in item:
                payload["item_index"] = item["idx"]

            points.append(models.PointStruct(
                id=point_uuid,
                vector=vector,
                payload=payload
            ))

        if points:
            # Ensure collection exists
            self.meeting_vector_repo.ensure_collection(vector_size=len(all_vectors[0]))
            
            # Clean up old points for this meeting to avoid duplicates if re-processing
            self.meeting_vector_repo.delete_by_meeting_id(meeting_id, tenant_id)
            
            self.meeting_vector_repo.upsert_points(points)
            logger.info(f"[MeetingEmbeddingService] Upserted {len(points)} points for meetingId {meeting_id} (Consolidated)")
        
        return len(points)
