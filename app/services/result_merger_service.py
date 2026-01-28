"""
Result Merger Service
Merges chunk transcription results into single transcript with speaker mapping
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from app.schemas.transcription_v1 import TranscriptionSegment, SpeakerSummary, TranscriptionMetadata

logger = logging.getLogger(__name__)


@dataclass
class MergedTranscriptionResult:
    """Result of merging all chunk transcriptions"""
    transcriptions: List[TranscriptionSegment]
    speakers: List[SpeakerSummary]
    metadata: TranscriptionMetadata


class ResultMergerService:
    """
    Service for merging chunk transcription results

    Features:
    - Validates all chunks succeeded (fails if any chunk failed)
    - Converts relative timestamps to absolute timeline
    - Maps speakers using maximum overlap logic
    - Calculates speaker summaries (duration, sentiment)
    - Sorts segments chronologically
    """

    async def merge_transcriptions(
        self,
        chunk_results: List,  # List[ChunkTranscriptionResult]
        speaker_timeframes: Optional[List[Dict]] = None,
        participants: Optional[List[Dict]] = None
    ) -> MergedTranscriptionResult:
        """
        Merge all chunk results into single transcription

        Args:
            chunk_results: List of ChunkTranscriptionResult objects
            speaker_timeframes: Optional speaker timeframes for speaker mapping
                [{"speakerName": str, "start": int (ms), "end": int (ms)}]
            participants: Optional participant objects (not used currently)

        Returns:
            MergedTranscriptionResult with transcriptions, speakers, and metadata

        Raises:
            ValueError: If any chunks failed
        """
        logger.info(f"[ResultMerger] Merging {len(chunk_results)} chunk results")

        # 1. Validate all chunks succeeded
        failed_chunks = [r for r in chunk_results if r.status != "success"]
        if failed_chunks:
            failed_ids = [r.chunk_id for r in failed_chunks]
            error_msg = (
                f"Transcription incomplete: {len(failed_chunks)}/{len(chunk_results)} chunks failed "
                f"(chunk IDs: {failed_ids}). Cannot merge partial results."
            )
            logger.error(f"[ResultMerger] {error_msg}")
            raise ValueError(error_msg)

        # 2. Merge segments with absolute timestamps
        all_segments = []
        for result in chunk_results:
            chunk_start_seconds = self._time_to_seconds(result.chunk_info["start_time"])

            for transcription in result.transcriptions:
                # Normalize: handle cached segments where Gemini used "text" instead of "transcription"
                if "transcription" not in transcription and "text" in transcription:
                    transcription["transcription"] = transcription.pop("text")

                # Convert relative to absolute
                segment_start = self._time_to_seconds(transcription["start"])
                segment_end = self._time_to_seconds(transcription["end"])

                absolute_start = chunk_start_seconds + segment_start
                absolute_end = chunk_start_seconds + segment_end

                # Update segment with absolute times
                transcription["start"] = self._seconds_to_time(absolute_start)
                transcription["end"] = self._seconds_to_time(absolute_end)
                transcription["source_chunk"] = result.chunk_id
                transcription["chunk_start_time"] = result.chunk_info["start_time"]
                transcription["chunk_end_time"] = result.chunk_info["end_time"]

                all_segments.append(transcription)

        # 3. Sort chronologically
        all_segments.sort(key=lambda x: (x.get("source_chunk", 0), x.get("segment_id", 0)))

        # Assign segment_id if missing
        for idx, segment in enumerate(all_segments, start=1):
            if "segment_id" not in segment:
                segment["segment_id"] = idx

        # 4. Map speakers if timeframes available
        if speaker_timeframes:
            all_segments = self._map_speakers_to_segments(all_segments, speaker_timeframes)
        else:
            for segment in all_segments:
                segment["speaker"] = "Unknown"

        # 5. Calculate speaker summaries
        speakers_summary = self._calculate_speaker_summaries(all_segments)

        # 6. Create metadata
        metadata = TranscriptionMetadata(
            total_segments=len(all_segments),
            successful_chunks=len(chunk_results),
            failed_chunks=0,
            total_chunks=len(chunk_results),
            has_speaker_mapping=bool(speaker_timeframes)
        )

        logger.info(
            f"[ResultMerger] Merge complete | "
            f"segments={len(all_segments)} speakers={len(speakers_summary)}"
        )

        return MergedTranscriptionResult(
            transcriptions=[TranscriptionSegment(**seg) for seg in all_segments],
            speakers=speakers_summary,
            metadata=metadata
        )

    def _map_speakers_to_segments(
        self,
        segments: List[Dict],
        speaker_timeframes: List[Dict]
    ) -> List[Dict]:
        """
        Map speakers to segments using maximum overlap logic

        Args:
            segments: List of segment dictionaries
            speaker_timeframes: List of speaker timeframes (ms)

        Returns:
            Segments with "speaker" field added
        """
        # Convert timeframes to MM:SS
        timeframes_mmss = []
        for frame in speaker_timeframes:
            timeframes_mmss.append({
                "speaker_name": frame["speakerName"],
                "start": self._ms_to_mmss(frame["start"]),
                "end": self._ms_to_mmss(frame["end"])
            })

        # Map each segment
        for segment in segments:
            segment_start = self._time_to_seconds(segment["start"])
            segment_end = self._time_to_seconds(segment["end"])

            best_match = None
            max_overlap = 0
            best_duration = float('inf')

            for frame in timeframes_mmss:
                frame_start = self._time_to_seconds(frame["start"])
                frame_end = self._time_to_seconds(frame["end"])
                frame_duration = frame_end - frame_start

                # Calculate overlap
                overlap_start = max(segment_start, frame_start)
                overlap_end = min(segment_end, frame_end)
                overlap_duration = max(0, overlap_end - overlap_start)

                # Prefer higher overlap, or shorter timeframe if equal
                if (overlap_duration > max_overlap or
                    (overlap_duration == max_overlap and frame_duration < best_duration)):
                    max_overlap = overlap_duration
                    best_duration = frame_duration
                    best_match = frame["speaker_name"]

            segment["speaker"] = best_match if best_match else "Unknown"

        return segments

    def _calculate_speaker_summaries(self, segments: List[Dict]) -> List[SpeakerSummary]:
        """
        Calculate speaker statistics

        Args:
            segments: List of segments with speaker field

        Returns:
            List of SpeakerSummary objects
        """
        speaker_stats = {}

        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            sentiment = segment.get("sentiment", "neutral")

            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "segments": 0,
                    "total_duration_seconds": 0,
                    "sentiments": []
                }

            speaker_stats[speaker]["segments"] += 1

            start_seconds = self._time_to_seconds(segment["start"])
            end_seconds = self._time_to_seconds(segment["end"])
            speaker_stats[speaker]["total_duration_seconds"] += (end_seconds - start_seconds)
            speaker_stats[speaker]["sentiments"].append(sentiment)

        # Convert to SpeakerSummary objects
        summaries = []
        for speaker, stats in speaker_stats.items():
            summaries.append(SpeakerSummary(
                speaker=speaker,
                segments=stats["segments"],
                duration=self._seconds_to_time(stats["total_duration_seconds"]),
                sentiment=self._calculate_average_sentiment(stats["sentiments"])
            ))

        return summaries

    def _calculate_average_sentiment(self, sentiments: List[str]) -> str:
        """Calculate average sentiment from list"""
        if not sentiments:
            return "neutral"

        # Check for explicit mixed
        if any(s.lower() == "mixed" for s in sentiments):
            return "mixed"

        # Count sentiment types
        sentiment_values = {"positive": 1, "neutral": 0, "negative": -1, "mixed": 0}
        total_value = sum(sentiment_values.get(s.lower(), 0) for s in sentiments)
        average = total_value / len(sentiments)

        # Check for mixed (both positive and negative)
        has_positive = any(s.lower() == "positive" for s in sentiments)
        has_negative = any(s.lower() == "negative" for s in sentiments)
        if has_positive and has_negative and -0.25 <= average <= 0.25:
            return "mixed"

        # Standard thresholds
        if average > 0.33:
            return "positive"
        elif average < -0.33:
            return "negative"
        else:
            return "neutral"

    def _time_to_seconds(self, time_str: str) -> float:
        """Convert MM:SS to seconds"""
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(float(parts[1]))
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return 0.0

    def _seconds_to_time(self, seconds: float) -> str:
        """Convert seconds to MM:SS"""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def _ms_to_mmss(self, ms: int) -> str:
        """Convert milliseconds to MM:SS"""
        total_seconds = ms // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
