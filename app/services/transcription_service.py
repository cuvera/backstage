import asyncio
import json
import logging
from typing import Dict, List, Any, Optional

from app.utils.audio_chunker import chunk_audio_by_segments, chunk_audio_file
from app.services.gemini_transcription_agent import transcribe
from app.core.prompts import (
    TRANSCRIPTION_AND_SENTIMENT_ANALYSIS_PROMPT_ONLINE, 
    TRANSCRIPTION_AND_SENTIMENT_ANALYSIS_PROMPT_OFFLINE
)

logger = logging.getLogger(__name__)


class TranscriptionService:
    """
    Comprehensive transcription service that handles:
    - Platform-specific chunking strategies (online/offline)
    - Parallel transcription processing with semaphore
    - Merging chunk results into single transcription
    """
    
    def __init__(self, max_concurrent: int = 5):
        """
        Initialize TranscriptionService
        
        Args:
            max_concurrent: Maximum number of concurrent transcription calls
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(f"[TranscriptionService] Initialized with max {max_concurrent} concurrent transcriptions")
    
    def _ms_to_hhmmss(self, ms: int) -> str:
        """
        Convert milliseconds to HH:MM:SS format
        
        Args:
            ms: Time in milliseconds
            
        Returns:
            Time string in HH:MM:SS format
        """
        s = ms // 1000
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"
    
    def _generate_segments_from_speaker_timeframes(self, speaker_timeframes: List[Dict]) -> Dict:
        """
        Convert speaker timeframes to segments format for chunking
        
        Args:
            speaker_timeframes: List of dicts with speakerName, start, end (in ms)
            
        Returns:
            Dict with segments array
        """
        segments = []
        for idx, seg in enumerate(speaker_timeframes, start=1):
            segments.append({
                "segment_id": idx,
                "start": self._ms_to_hhmmss(seg["start"]),
                "end": self._ms_to_hhmmss(seg["end"]),
                "label": seg["speakerName"]
            })
        
        logger.info(f"[TranscriptionService] Generated {len(segments)} segments from speaker timeframes")
        return {"segments": segments}
    
    async def _chunk_audio(
        self, 
        audio_file_path: str, 
        meeting_metadata: Dict, 
        platform: str,
        chunk_duration_minutes: float = 10.0,
        overlap_seconds: float = 5.0,
        output_dir: str = './chunks/'
    ) -> List[Dict]:
        """
        Choose chunking strategy based on platform
        
        Args:
            audio_file_path: Path to the audio file
            meeting_metadata: Meeting metadata containing speaker_timeframes
            platform: "online" or "offline"
            chunk_duration_minutes: Duration of each chunk in minutes
            overlap_seconds: Overlap between chunks in seconds
            output_dir: Output directory for chunks
            
        Returns:
            List of chunk dictionaries
        """
        logger.info(f"[TranscriptionService] Chunking audio for {platform} platform")
        
        if platform.lower() == "offline":
            # Simple time-based chunking for offline
            chunks = chunk_audio_file(
                input_file_path=audio_file_path,
                chunk_duration_minutes=chunk_duration_minutes,
                overlap_seconds=overlap_seconds,
                output_dir=output_dir
            )
        else:
            # Speaker-based chunking for online
            speaker_timeframes = meeting_metadata.get("speaker_timeframes", [])
            
            if not speaker_timeframes:
                logger.warning("[TranscriptionService] No speaker timeframes found, falling back to time-based chunking")
                chunks = chunk_audio_file(
                    input_file_path=audio_file_path,
                    chunk_duration_minutes=chunk_duration_minutes,
                    overlap_seconds=overlap_seconds,
                    output_dir=output_dir
                )
            else:
                segments_data = self._generate_segments_from_speaker_timeframes(speaker_timeframes)
                chunks = chunk_audio_by_segments(
                    input_file_path=audio_file_path,
                    segments_data=segments_data,
                    chunk_duration_minutes=chunk_duration_minutes,
                    overlap_seconds=overlap_seconds,
                    output_dir=output_dir
                )
        
        logger.info(f"[TranscriptionService] Generated {len(chunks)} audio chunks")
        return chunks
    
    async def _transcribe_chunk_with_semaphore(
        self, 
        prompt: str, 
        audio_file_path: str, 
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe a single chunk with semaphore control
        
        Args:
            prompt: Transcription prompt
            audio_file_path: Path to audio chunk
            models: Optional list of models for fallback chain
            
        Returns:
            Transcription result
        """
        async with self.semaphore:
            return await transcribe(
                prompt=prompt,
                audio_file_path=audio_file_path,
                models=models or []
            )
    
    async def _transcribe_chunks_parallel(
        self, 
        chunks: List[Dict], 
        platform: str,
        models: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Transcribe all chunks in parallel with semaphore limiting concurrency
        
        Args:
            chunks: List of audio chunks
            platform: "online" or "offline" for prompt selection
            models: Optional list of models for fallback chain
            
        Returns:
            List of transcription results
        """
        logger.info(f"[TranscriptionService] Starting parallel transcription of {len(chunks)} chunks")
        
        # Select prompt based on platform
        if platform.lower() == "offline":
            base_prompt = TRANSCRIPTION_AND_SENTIMENT_ANALYSIS_PROMPT_OFFLINE
        else:
            base_prompt = TRANSCRIPTION_AND_SENTIMENT_ANALYSIS_PROMPT_ONLINE
        
        # Create all tasks
        tasks = []
        for chunk in chunks:
            # Replace placeholders in prompt
            prompt = base_prompt
            prompt = prompt.replace("{{start}}", chunk.get("start_time", ""))
            prompt = prompt.replace("{{end}}", chunk.get("end_time", ""))
            prompt = prompt.replace("{{segments}}", json.dumps(chunk.get("segments", [])))
            
            # Create task with semaphore control
            task = asyncio.create_task(
                self._transcribe_chunk_with_semaphore(prompt, chunk["file_path"], models)
            )
            tasks.append((task, chunk))
        
        # Wait for all tasks to complete
        results = []
        for task, chunk in tasks:
            try:
                result = await task
                result["chunk_info"] = {
                    "chunk_id": chunk.get("chunk_id"),
                    "start_time": chunk.get("start_time"),
                    "end_time": chunk.get("end_time"),
                    "file_path": chunk.get("file_path")
                }
                results.append(result)
                logger.info(f"[TranscriptionService] Completed transcription for chunk {chunk.get('chunk_id')}")
            except Exception as e:
                logger.error(f"[TranscriptionService] Failed to transcribe chunk {chunk.get('chunk_id')}: {e}")
                # Add empty result to maintain order
                results.append({
                    "transcriptions": [],
                    "chunk_info": {
                        "chunk_id": chunk.get("chunk_id"),
                        "error": str(e)
                    }
                })
        
        logger.info(f"[TranscriptionService] Completed parallel transcription of {len(results)} chunks")
        return results
    
    def _merge_transcription_results(self, transcription_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge all chunk transcription results into a single transcription
        
        Args:
            transcription_results: List of transcription results from chunks
            
        Returns:
            Merged transcription result
        """
        logger.info(f"[TranscriptionService] Merging {len(transcription_results)} transcription results")
        
        merged_transcriptions = []
        total_segments = 0
        successful_chunks = 0
        failed_chunks = 0
        
        for result in transcription_results:
            chunk_info = result.get("chunk_info", {})
            
            if "error" in chunk_info:
                failed_chunks += 1
                logger.warning(f"[TranscriptionService] Skipping failed chunk {chunk_info.get('chunk_id')}")
                continue
            
            transcriptions = result.get("transcriptions", [])
            if transcriptions:
                # Add chunk context to each transcription segment
                for transcription in transcriptions:
                    transcription["source_chunk"] = chunk_info.get("chunk_id")
                    transcription["chunk_start_time"] = chunk_info.get("start_time")
                    transcription["chunk_end_time"] = chunk_info.get("end_time")
                
                merged_transcriptions.extend(transcriptions)
                total_segments += len(transcriptions)
                successful_chunks += 1
        
        # Sort by segment timing if available
        try:
            merged_transcriptions.sort(key=lambda x: x.get("start", "00:00:00"))
        except:
            logger.warning("[TranscriptionService] Could not sort transcriptions by time")
        
        merged_result = {
            "transcriptions": merged_transcriptions,
            "metadata": {
                "total_segments": total_segments,
                "successful_chunks": successful_chunks,
                "failed_chunks": failed_chunks,
                "total_chunks": len(transcription_results)
            }
        }
        
        logger.info(f"[TranscriptionService] Merged transcription complete: {total_segments} segments from {successful_chunks}/{len(transcription_results)} chunks")
        return merged_result
    
    async def transcribe_meeting(
        self, 
        audio_file_path: str, 
        meeting_metadata: Dict, 
        platform: str = "online",
        chunk_duration_minutes: float = 10.0,
        overlap_seconds: float = 5.0,
        output_dir: str = './chunks/',
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main method to transcribe a meeting with platform-specific chunking and parallel processing
        
        Args:
            audio_file_path: Path to the audio file
            meeting_metadata: Meeting metadata containing speaker_timeframes for online platform
            platform: "online" or "offline" - determines chunking strategy
            chunk_duration_minutes: Duration of each chunk in minutes
            overlap_seconds: Overlap between chunks in seconds
            output_dir: Output directory for chunks
            models: Optional list of models for fallback chain
            
        Returns:
            Merged transcription result with metadata
        """
        logger.info(f"[TranscriptionService] Starting meeting transcription for {platform} platform")
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 1. Chunk audio based on platform strategy
            chunks = await self._chunk_audio(
                audio_file_path, 
                meeting_metadata, 
                platform,
                chunk_duration_minutes,
                overlap_seconds,
                output_dir
            )
            
            if not chunks:
                raise ValueError("No audio chunks generated")
            
            # 2. Parallel transcription with semaphore
            transcription_results = await self._transcribe_chunks_parallel(chunks, platform, models)
            
            # 3. Merge all chunk results into single transcription
            merged_transcription = self._merge_transcription_results(transcription_results)
            
            # Add processing metadata
            end_time = asyncio.get_event_loop().time()
            processing_time = round((end_time - start_time) * 1000, 2)
            
            merged_transcription["processing_metadata"] = {
                "platform": platform,
                "audio_file_path": audio_file_path,
                "processing_time_ms": processing_time,
                "chunk_duration_minutes": chunk_duration_minutes,
                "overlap_seconds": overlap_seconds,
                "max_concurrent": self.semaphore._value
            }
            
            logger.info(f"[TranscriptionService] Meeting transcription completed in {processing_time}ms")
            return merged_transcription
            
        except Exception as e:
            logger.error(f"[TranscriptionService] Meeting transcription failed: {e}")
            raise