"""
Chunk Transcriber
Handles parallel transcription of audio chunks with incremental saving
"""

import asyncio
import base64
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from app.core.llm_client import llm_client
from app.repository.transcription_chunk_repository import TranscriptionChunkRepository
from app.core.prompts import (
    ONLINE_TRANSCRIPTION,
    OFFLINE_TRANSCRIPTION
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkTranscriptionResult:
    """Result of transcribing a single chunk"""
    chunk_id: int
    status: str  # "success" or "failed"
    transcriptions: List[Dict]
    chunk_info: Dict
    error: Optional[str] = None


class ChunkTranscriber:
    """
    Service for parallel transcription of audio chunks with incremental saving

    Features:
    - Parallel processing with semaphore (max 5 concurrent)
    - Incremental chunk saving to database
    - Resume capability (skip already completed chunks)
    - Retry/fallback handled by llm_client (no custom retry logic needed)
    """

    def __init__(self, max_concurrent: int = 5):
        """
        Initialize ChunkTranscriber

        Args:
            max_concurrent: Maximum number of concurrent transcription calls
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        logger.info(f"ChunkTranscriber initialized | max_concurrent={max_concurrent}")

    async def transcribe_chunks(
        self,
        chunks: List[Dict],
        participants: List[Dict],
        transcription_id: str,
        tenant_id: str,
        enable_incremental_saving: bool = True
    ) -> List[ChunkTranscriptionResult]:
        """
        Transcribe all chunks in parallel with incremental saving

        Args:
            chunks: List of audio chunks from chunker
            participants: List of participant objects with structure:
                [{"id": str, "name": str, "email": str, "department": str}]
            transcription_id: Transcription identifier
            tenant_id: Tenant identifier
            enable_incremental_saving: Enable saving chunks incrementally (default: True)

        Returns:
            List of ChunkTranscriptionResult objects
        """
        logger.info(f"Transcribing | chunks={len(chunks)} participants={len(participants)}")

        # Initialize repository
        chunk_repo = None
        if enable_incremental_saving:
            chunk_repo = await TranscriptionChunkRepository.from_default()

        # Check for existing completed chunks (resume capability)
        existing_chunks = []
        chunks_to_process = chunks

        if enable_incremental_saving and chunk_repo:
            existing_chunks = await chunk_repo.get_chunks(transcription_id, tenant_id, status="success")

            if existing_chunks:
                completed_ids = {c["chunk_id"] for c in existing_chunks}
                chunks_to_process = [c for c in chunks if c.get("chunk_id") not in completed_ids]

                logger.info(f"Resuming | {len(existing_chunks)} done, {len(chunks_to_process)} remaining")

        # Convert existing to results
        results = []
        for existing_chunk in existing_chunks:
            if existing_chunk.get("result"):
                results.append(ChunkTranscriptionResult(
                    chunk_id=existing_chunk["chunk_id"],
                    status="success",
                    transcriptions=existing_chunk["result"].get("transcriptions", []),
                    chunk_info=existing_chunk["result"].get("chunk_info", {})
                ))

        # Process remaining chunks in parallel
        if chunks_to_process:
            new_results = await self._transcribe_chunks_parallel(
                chunks_to_process,
                participants,
                transcription_id,
                tenant_id,
                chunk_repo
            )
            results.extend(new_results)

        logger.info(f"All chunks done | total={len(results)}")
        return results

    async def _transcribe_chunks_parallel(
        self,
        chunks: List[Dict],
        participants: List[Dict],
        transcription_id: str,
        tenant_id: str,
        chunk_repo: Optional[TranscriptionChunkRepository]
    ) -> List[ChunkTranscriptionResult]:
        """
        Process chunks in parallel with semaphore limiting concurrency

        Args:
            chunks: List of chunks to process
            participants: List of participant objects
            transcription_id: Transcription identifier
            tenant_id: Tenant identifier
            chunk_repo: Repository for saving chunks (None to disable saving)

        Returns:
            List of ChunkTranscriptionResult objects
        """
        # Extract participant names for prompt
        participant_names = [p.get("name", "") for p in participants]

        # Create tasks for parallel processing
        tasks = []
        for chunk in chunks:
            # Determine mode based on chunk structure
            mode = "online" if chunk.get("segments") else "offline"

            # Build prompt
            prompt = self._build_prompt(mode, chunk, participant_names)

            # Create task
            task = asyncio.create_task(
                self._transcribe_chunk(
                    chunk=chunk,
                    prompt=prompt,
                    transcription_id=transcription_id,
                    tenant_id=tenant_id,
                    chunk_repo=chunk_repo
                )
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    async def _transcribe_chunk(
        self,
        chunk: Dict,
        prompt: str,
        transcription_id: str,
        tenant_id: str,
        chunk_repo: Optional[TranscriptionChunkRepository]
    ) -> ChunkTranscriptionResult:
        """
        Transcribe single chunk with incremental saving
        Retry/fallback handled by llm_client
        """
        chunk_id = chunk.get("chunk_id")

        # Mark as processing
        if chunk_repo:
            try:
                await chunk_repo.save_chunk(
                    transcription_id=transcription_id,
                    tenant_id=tenant_id,
                    chunk_id=chunk_id,
                    status="processing",
                    chunk_info=chunk
                )
            except Exception as e:
                logger.warning(f"Failed to mark chunk {chunk_id} as processing: {e}")

        try:
            # Transcribe with semaphore (retry + fallback handled by llm_client)
            async with self.semaphore:
                # 1. Read audio file and convert to base64
                base64_audio = await asyncio.to_thread(
                    self._read_and_encode_audio, chunk["file_path"]
                )

                # 2. Build multimodal message
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": base64_audio,
                                    "format": "mp3"
                                }
                            }
                        ]
                    }
                ]

                # 3. Call llm_client
                response_text = await llm_client.chat_completion(
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )

                # 4. Parse JSON response
                result = json.loads(response_text)

                # Normalize segment field names (Gemini inconsistently uses "text" vs "transcription")
                for seg in result.get("transcriptions", []):
                    if "transcription" not in seg and "text" in seg:
                        seg["transcription"] = seg.pop("text")

            # Add chunk info to result
            result["chunk_info"] = {
                "chunk_id": chunk_id,
                "start_time": chunk.get("start_time"),
                "end_time": chunk.get("end_time"),
                "file_path": chunk.get("file_path")
            }

            # Validate segment count
            segments_sent = len(chunk.get("segments", []))
            segments_returned = len(result.get("transcriptions", []))

            if segments_sent > 0 and segments_sent != segments_returned:
                logger.warning(f"Chunk {chunk_id} segment mismatch: sent={segments_sent} received={segments_returned}")

            # Save success
            if chunk_repo:
                try:
                    await chunk_repo.save_chunk(
                        transcription_id=transcription_id,
                        tenant_id=tenant_id,
                        chunk_id=chunk_id,
                        status="success",
                        chunk_info=chunk,
                        result=result
                    )
                except Exception as save_error:
                    logger.error(f"Failed to save chunk {chunk_id}: {save_error}")

            logger.info(f"Chunk {chunk_id} succeeded")

            return ChunkTranscriptionResult(
                chunk_id=chunk_id,
                status="success",
                transcriptions=result.get("transcriptions", []),
                chunk_info=result["chunk_info"]
            )

        except Exception as e:
            logger.error(f"Chunk {chunk_id} failed: {e}")

            # Save as failed
            if chunk_repo:
                try:
                    await chunk_repo.save_chunk(
                        transcription_id=transcription_id,
                        tenant_id=tenant_id,
                        chunk_id=chunk_id,
                        status="failed",
                        chunk_info=chunk,
                        error=str(e)
                    )
                except Exception as save_error:
                    logger.error(f"Failed to save failed chunk {chunk_id}: {save_error}")

            return ChunkTranscriptionResult(
                chunk_id=chunk_id,
                status="failed",
                transcriptions=[],
                chunk_info={"chunk_id": chunk_id, "error": str(e)},
                error=str(e)
            )

    @staticmethod
    def _read_and_encode_audio(file_path: str) -> str:
        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        return base64.b64encode(audio_bytes).decode("utf-8")

    def _build_prompt(self, mode: str, chunk: Dict, participant_names: List[str]) -> str:
        """
        Build prompt with placeholders replaced

        Args:
            mode: "online" or "offline"
            chunk: Chunk dictionary
            participant_names: List of participant names

        Returns:
            Formatted prompt string
        """
        # Select base prompt
        if mode == "offline":
            base_prompt = OFFLINE_TRANSCRIPTION
        else:
            base_prompt = ONLINE_TRANSCRIPTION

        # Replace placeholders
        prompt = base_prompt
        prompt = prompt.replace("{{start}}", chunk.get("start_time", "00:00"))
        prompt = prompt.replace("{{end}}", chunk.get("end_time", "00:00"))
        prompt = prompt.replace("{{segments}}", json.dumps(chunk.get("segments", [])))

        # Format participants
        participants_str = "\n".join(f"- {name}" for name in participant_names) if participant_names else "Not specified"
        prompt = prompt.replace("{{participants}}", participants_str)

        return prompt
