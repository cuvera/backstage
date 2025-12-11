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
    
    def _ms_to_mmss(self, ms: int) -> str:
        """
        Convert milliseconds to MM:SS format

        Args:
            ms: Time in milliseconds

        Returns:
            Time string in MM:SS format
        """
        total_seconds = ms // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
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
                "start": self._ms_to_mmss(seg["start"]),
                "end": self._ms_to_mmss(seg["end"]),
                "label": seg["speaker_name"]
            })

        logger.info(f"[TranscriptionService] Generated {len(segments)} segments from speaker timeframes")
        if segments:
            logger.info(f"[TranscriptionService] First segment: {segments[0]}")
            logger.info(f"[TranscriptionService] Last segment: {segments[-1]}")
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
        # print(chunks)
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

        # start_time = chunks[0].get("start_time", "")
        # end_time = chunks[-1].get("end_time", "")

        for chunk in chunks:
            # Replace placeholders in prompt
            prompt = base_prompt
            prompt = prompt.replace("{{start}}", chunk.get("start_time", ""))
            prompt = prompt.replace("{{end}}", chunk.get("end_time", ""))
            prompt = prompt.replace("{{segments}}", json.dumps(chunk.get("segments", [])))
            
            # print chunk dictionary
            print(chunks)

            print(prompt)

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
    
    def _time_to_seconds(self, time_str: str) -> float:
        """Convert MM:SS to seconds (Gemini format)"""
        parts = time_str.split(':')
        if len(parts) == 2:
            # MM:SS format from Gemini
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 3:
            # HH:MM:SS format (fallback)
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        else:
            return 0.0

    def _seconds_to_time(self, seconds: float) -> str:
        """Convert seconds to MM:SS format (Gemini format)"""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def _map_speakers_to_segments(self, segments, meeting_metadata):
        """
        Map speaker names to transcription segments based on timeframes.
        Uses maximum overlap logic to handle overlapping speaker timeframes.
        When overlaps are equal, prefers the more specific (shorter) timeframe.
        """
        speaker_timeframes = meeting_metadata.get("speaker_timeframes", [])

        # Convert speaker timeframes to MM:SS format once
        speaker_timeframes_mmss = []
        for speaker_frame in speaker_timeframes:
            speaker_timeframes_mmss.append({
                "speaker_name": speaker_frame["speaker_name"],
                "start": self._ms_to_mmss(speaker_frame["start"]),
                "end": self._ms_to_mmss(speaker_frame["end"])
            })

        for segment in segments:
            segment_start = self._time_to_seconds(segment["start"])
            segment_end = self._time_to_seconds(segment["end"])

            best_match = None
            max_overlap = 0
            best_duration = float('inf')  # Track duration of best match

            for speaker_frame in speaker_timeframes_mmss:
                speaker_start = self._time_to_seconds(speaker_frame["start"])
                speaker_end = self._time_to_seconds(speaker_frame["end"])
                speaker_duration = speaker_end - speaker_start

                # Calculate overlap duration
                overlap_start = max(segment_start, speaker_start)
                overlap_end = min(segment_end, speaker_end)
                overlap_duration = max(0, overlap_end - overlap_start)

                # Prefer higher overlap, or if equal overlap, prefer shorter (more specific) timeframe
                if (overlap_duration > max_overlap or
                    (overlap_duration == max_overlap and speaker_duration < best_duration)):
                    max_overlap = overlap_duration
                    best_duration = speaker_duration
                    best_match = speaker_frame["speaker_name"]

            segment["speaker"] = best_match if best_match else "Unknown"

        return segments

    def _calculate_average_sentiment(self, sentiments):
        """Calculate average sentiment from list of sentiment strings"""
        if not sentiments:
            return "neutral"
        
        sentiment_values = {"positive": 1, "neutral": 0, "negative": -1}
        total_value = sum(sentiment_values.get(s.lower(), 0) for s in sentiments)
        average = total_value / len(sentiments)
        
        if average > 0.33:
            return "positive"
        elif average < -0.33:
            return "negative"
        else:
            return "neutral"

    def _extract_speaker_summary(self, segments):
        """Extract speaker summary as array of objects"""
        speaker_stats = {}
        
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            sentiment = segment.get("sentiment", "neutral")
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {"segments": 0, "total_duration_seconds": 0, "sentiments": []}
            
            speaker_stats[speaker]["segments"] += 1
            start_seconds = self._time_to_seconds(segment["start"])
            end_seconds = self._time_to_seconds(segment["end"])
            speaker_stats[speaker]["total_duration_seconds"] += (end_seconds - start_seconds)
            speaker_stats[speaker]["sentiments"].append(sentiment)
        
        speakers_array = []
        for speaker_name, stats in speaker_stats.items():
            speakers_array.append({
                "speaker": speaker_name,
                "segments": stats["segments"],
                "duration": self._seconds_to_time(stats["total_duration_seconds"]),
                "sentiment": self._calculate_average_sentiment(stats["sentiments"])
            })
        
        return speakers_array

    def _merge_transcription_results(self, transcription_results: List[Dict[str, Any]], meeting_metadata: Dict = None) -> Dict[str, Any]:
        """
        Merge all chunk transcription results into a single transcription with absolute timeline and speaker mapping

        Args:
            transcription_results: List of transcription results from chunks
            meeting_metadata: Meeting metadata containing speaker_timeframes

        Returns:
            Merged transcription result with absolute timeline, speaker mapping, and speaker summary

        Raises:
            ValueError: If any chunks failed to transcribe
        """
        logger.info(f"[TranscriptionService] Merging {len(transcription_results)} transcription results")

        # Count failed chunks first
        failed_chunks = 0
        failed_chunk_ids = []
        for result in transcription_results:
            chunk_info = result.get("chunk_info", {})
            if "error" in chunk_info:
                failed_chunks += 1
                chunk_id = chunk_info.get('chunk_id', 'unknown')
                failed_chunk_ids.append(chunk_id)
                logger.error(f"[TranscriptionService] Chunk {chunk_id} failed: {chunk_info.get('error')}")

        # Fail if any chunks failed
        if failed_chunks > 0:
            raise ValueError(
                f"Transcription incomplete: {failed_chunks}/{len(transcription_results)} chunks failed "
                f"(chunk IDs: {failed_chunk_ids}). Cannot merge partial results."
            )

        all_segments = []

        for result in transcription_results:
            chunk_info = result.get("chunk_info", {})

            chunk_start_seconds = self._time_to_seconds(chunk_info.get("start_time", "00:00"))

            transcriptions = result.get("transcriptions", [])
            for transcription in transcriptions:
                # Convert relative timestamps to absolute timeline
                segment_start_seconds = self._time_to_seconds(transcription.get("start", "00:00"))
                segment_end_seconds = self._time_to_seconds(transcription.get("end", "00:00"))

                # All transcriptions now come with relative timestamps, convert to absolute
                absolute_start = chunk_start_seconds + segment_start_seconds
                absolute_end = chunk_start_seconds + segment_end_seconds

                # Update start/end with absolute values
                transcription["start"] = self._seconds_to_time(absolute_start)
                transcription["end"] = self._seconds_to_time(absolute_end)

                # Keep chunk context (already in MM:SS format from audio_chunker)
                transcription["source_chunk"] = chunk_info.get("chunk_id")
                transcription["chunk_start_time"] = chunk_info.get("start_time", "00:00")
                transcription["chunk_end_time"] = chunk_info.get("end_time", "00:00")

                all_segments.append(transcription)

        # Sort by source_chunk first (ascending), then by segment_id within chunk
        all_segments.sort(key=lambda x: (x.get("source_chunk", 0), x.get("segment_id", 0)))
        
        # Add speaker mapping if available
        if meeting_metadata and meeting_metadata.get("speaker_timeframes"):
            all_segments = self._map_speakers_to_segments(all_segments, meeting_metadata)
        
        # Generate speaker summary
        speakers_summary = self._extract_speaker_summary(all_segments)
        
        merged_result = {
            "transcriptions": all_segments,
            "speakers": speakers_summary,
            "metadata": {
                "total_segments": len(all_segments),
                "total_chunks": len(transcription_results),
                "has_speaker_mapping": bool(meeting_metadata and meeting_metadata.get("speaker_timeframes"))
            }
        }
        
        logger.info(f"[TranscriptionService] Merged transcription complete: {len(all_segments)} segments from {len(transcription_results)} chunks")
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
            merged_transcription = self._merge_transcription_results(transcription_results, meeting_metadata)
            
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
