import logging
import os
import tempfile
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class AudioChunkerError(Exception):
    """Custom exception for audio chunker operations."""
    pass


def parse_timestamp(timestamp_str: str) -> float:
    """
    Parse timestamp string (HH:MM:SS or MM:SS) to seconds.
    
    Args:
        timestamp_str: Timestamp in format "HH:MM:SS" or "MM:SS"
        
    Returns:
        Float seconds from start
    """
    parts = timestamp_str.split(':')
    if len(parts) == 2:
        minutes, seconds = map(float, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")


def seconds_to_timestamp(seconds: float) -> str:
    """
    Convert seconds to MM:SS timestamp format.

    Args:
        seconds: Time in seconds

    Returns:
        Timestamp string in MM:SS format
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def group_segments_by_duration(segments: List[Dict], target_duration_minutes: float = 10.0) -> List[List[Dict]]:
    """
    Group segments to approximate the target duration.
    
    Args:
        segments: List of segment dictionaries with 'start' and 'end' timestamps
        target_duration_minutes: Target duration for each group in minutes
        
    Returns:
        List of grouped segments, each group approximating target duration
    """
    if not segments:
        return []
    
    target_duration_seconds = target_duration_minutes * 60
    grouped_segments = []
    current_group = []
    current_duration = 0.0
    
    for segment in segments:
        start_seconds = parse_timestamp(segment['start'])
        end_seconds = parse_timestamp(segment['end'])
        segment_duration = end_seconds - start_seconds
        
        # If adding this segment would exceed target duration significantly,
        # start a new group (unless current group is empty)
        if current_group and (current_duration + segment_duration) > target_duration_seconds * 1.2:
            grouped_segments.append(current_group)
            current_group = [segment]
            current_duration = segment_duration
        else:
            current_group.append(segment)
            current_duration += segment_duration
    
    # Add the last group if it has segments
    if current_group:
        grouped_segments.append(current_group)
    
    return grouped_segments


class AudioChunker:
    """Utility for chunking audio files into segments with overlap using ffmpeg."""

    def __init__(self, chunk_duration_minutes: float = 10.0, overlap_seconds: float = 5.0):
        """
        Initialize the AudioChunker.

        Args:
            chunk_duration_minutes: Duration of each chunk in minutes
            overlap_seconds: Overlap duration between chunks in seconds
        """
        self.chunk_duration_minutes = chunk_duration_minutes
        self.overlap_seconds = overlap_seconds
        self.chunk_duration_seconds = chunk_duration_minutes * 60

    def _get_audio_duration_ffprobe(self, file_path: str) -> float:
        """
        Get audio duration using ffprobe without loading file into memory.

        Args:
            file_path: Path to the audio file

        Returns:
            Duration in seconds

        Raises:
            AudioChunkerError: If ffprobe fails
        """
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                file_path
            ], capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            raise AudioChunkerError(f"Failed to get audio duration: {e}")

    def _get_audio_metadata_ffprobe(self, file_path: str) -> dict:
        """
        Get audio metadata using ffprobe without loading file into memory.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary with duration and sample_rate

        Raises:
            AudioChunkerError: If ffprobe fails
        """
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration:stream=sample_rate',
                '-of', 'json',
                file_path
            ], capture_output=True, text=True, check=True)

            data = json.loads(result.stdout)

            duration = float(data['format']['duration'])
            # Get sample rate from first audio stream
            sample_rate = 44100  # Default fallback
            if 'streams' in data and len(data['streams']) > 0:
                sample_rate = int(data['streams'][0].get('sample_rate', 44100))

            return {'duration': duration, 'sample_rate': sample_rate}
        except Exception as e:
            raise AudioChunkerError(f"Failed to get audio metadata: {e}")

    def _extract_chunk_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        start_seconds: float,
        duration_seconds: float
    ) -> None:
        """
        Extract audio chunk using ffmpeg without loading into memory.

        Args:
            input_path: Path to input audio file
            output_path: Path to output chunk file
            start_seconds: Start time in seconds
            duration_seconds: Duration in seconds

        Raises:
            AudioChunkerError: If ffmpeg fails
        """
        try:
            # Get input file extension to preserve format
            input_ext = Path(input_path).suffix.lower()
            output_ext = Path(output_path).suffix.lower()

            # Build ffmpeg command
            cmd = [
                'ffmpeg', '-i', input_path,
                '-ss', str(start_seconds),
                '-t', str(duration_seconds),
            ]

            # Use codec copy if same format, otherwise re-encode
            if input_ext == output_ext:
                cmd.extend(['-c', 'copy'])
            else:
                # Re-encode with reasonable quality
                cmd.extend(['-c:a', 'aac', '-b:a', '128k'])

            cmd.extend(['-y', output_path])  # Overwrite if exists

            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.decode(errors='replace') if e.stderr else str(e)
            raise AudioChunkerError(f"ffmpeg chunk extraction failed: {stderr_output}")
        except Exception as e:
            raise AudioChunkerError(f"Failed to extract chunk: {e}")
        
    def chunk_audio_file(
        self,
        input_file_path: str,
        output_dir: Optional[str] = None,
        output_prefix: str = "chunk"
    ) -> List[Dict[str, Any]]:
        """
        Chunk an audio file into segments with overlap using ffmpeg (no memory loading).

        Args:
            input_file_path: Path to the input audio file
            output_dir: Directory to save chunks (defaults to temp directory)
            output_prefix: Prefix for output chunk files

        Returns:
            List of chunk information dictionaries

        Raises:
            AudioChunkerError: For various audio processing errors
        """
        try:
            input_path = Path(input_file_path)
            if not input_path.exists():
                raise AudioChunkerError(f"Input file not found: {input_file_path}")

            # Setup output directory
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix="audio_chunks_")
            else:
                os.makedirs(output_dir, exist_ok=True)

            logger.info(f"Starting audio chunking for: {input_file_path}")

            # Get audio duration using ffprobe (no memory loading)
            total_duration_seconds = self._get_audio_duration_ffprobe(input_file_path)
            logger.info(f"Audio duration: {total_duration_seconds:.2f} seconds")

            # Determine output format based on input
            input_ext = input_path.suffix.lower()
            output_ext = '.m4a'  # Default output format

            chunks = []
            chunk_index = 0
            start_time_seconds = 0.0

            while start_time_seconds < total_duration_seconds:
                # Calculate end time for this chunk
                end_time_seconds = min(
                    start_time_seconds + self.chunk_duration_seconds,
                    total_duration_seconds
                )
                chunk_duration = end_time_seconds - start_time_seconds

                # Generate output filename
                chunk_filename = f"{output_prefix}_{chunk_index:03d}{output_ext}"
                chunk_path = os.path.join(output_dir, chunk_filename)

                # Extract chunk using ffmpeg (direct file-to-file, no memory loading)
                self._extract_chunk_ffmpeg(
                    input_file_path,
                    chunk_path,
                    start_time_seconds,
                    chunk_duration
                )

                # Store chunk information
                chunk_info = {
                    "chunk_id": chunk_index,
                    "start_time": seconds_to_timestamp(start_time_seconds),
                    "end_time": seconds_to_timestamp(end_time_seconds),
                    "duration_seconds": chunk_duration,
                    "file_path": chunk_path,
                    "file_size_bytes": os.path.getsize(chunk_path),
                    "segments": []
                }
                chunks.append(chunk_info)

                logger.debug(f"Created chunk {chunk_index}: {chunk_info['start_time']} - {chunk_info['end_time']}")

                # Move to next chunk start (with overlap consideration)
                if end_time_seconds >= total_duration_seconds:
                    break

                start_time_seconds = end_time_seconds - self.overlap_seconds
                chunk_index += 1

            logger.info(f"Successfully created {len(chunks)} audio chunks in: {output_dir}")
            return chunks

        except Exception as e:
            logger.error(f"Audio chunking failed: {str(e)}", exc_info=True)
            raise AudioChunkerError(f"Audio chunking failed: {str(e)}") from e
    
    def chunk_audio_by_segments(
        self,
        input_file_path: str,
        segments_data: Dict,
        output_dir: Optional[str] = None,
        output_prefix: str = "segment_chunk"
    ) -> List[Dict[str, Any]]:
        """
        Chunk audio file based on provided segments using ffmpeg (no memory loading).

        Args:
            input_file_path: Path to the input audio file
            segments_data: Dictionary containing 'segments' list with 'segment_id', 'start', 'end'
            output_dir: Directory to save chunks (defaults to temp directory)
            output_prefix: Prefix for output chunk files

        Returns:
            List of chunk information dictionaries with grouped segments

        Raises:
            AudioChunkerError: For various audio processing errors
        """
        try:
            input_path = Path(input_file_path)
            if not input_path.exists():
                raise AudioChunkerError(f"Input file not found: {input_file_path}")

            segments = segments_data.get('segments', [])
            if not segments:
                raise AudioChunkerError("No segments provided")

            # Setup output directory
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix="audio_segment_chunks_")
            else:
                os.makedirs(output_dir, exist_ok=True)

            logger.info(f"Starting segment-based audio chunking for: {input_file_path}")
            logger.info(f"Processing {len(segments)} segments")

            # Get audio duration using ffprobe (no memory loading)
            total_duration_seconds = self._get_audio_duration_ffprobe(input_file_path)
            logger.info(f"Audio duration: {total_duration_seconds} seconds")

            # Group segments by target duration
            grouped_segments = group_segments_by_duration(segments, self.chunk_duration_minutes)

            logger.info(f"Created {len(grouped_segments)} chunk groups from {len(segments)} segments")

            # Determine output format
            output_ext = '.m4a'

            chunks = []

            for chunk_index, segment_group in enumerate(grouped_segments):
                # Find the overall start and end times for this group
                group_start_seconds = min(parse_timestamp(seg['start']) for seg in segment_group)
                group_end_seconds = max(parse_timestamp(seg['end']) for seg in segment_group)

                # Add overlap if not the first chunk
                if chunk_index > 0:
                    group_start_seconds = max(0, group_start_seconds - self.overlap_seconds)

                # Add overlap if not the last chunk
                if chunk_index < len(grouped_segments) - 1:
                    group_end_seconds = min(total_duration_seconds, group_end_seconds + self.overlap_seconds)

                chunk_duration = group_end_seconds - group_start_seconds

                # Generate output filename
                chunk_filename = f"{output_prefix}_{chunk_index:03d}{output_ext}"
                chunk_path = os.path.join(output_dir, chunk_filename)

                # Extract chunk using ffmpeg (direct file-to-file, no memory loading)
                self._extract_chunk_ffmpeg(
                    input_file_path,
                    chunk_path,
                    group_start_seconds,
                    chunk_duration
                )

                # Convert segments to relative timestamps (relative to chunk start)
                relative_segments = []
                for seg in segment_group:
                    seg_start_seconds = parse_timestamp(seg['start'])
                    seg_end_seconds = parse_timestamp(seg['end'])

                    # Convert to relative timestamps
                    relative_start = seg_start_seconds - group_start_seconds
                    relative_end = seg_end_seconds - group_start_seconds

                    relative_segment = seg.copy()
                    relative_segment['start'] = seconds_to_timestamp(relative_start)
                    relative_segment['end'] = seconds_to_timestamp(relative_end)
                    relative_segments.append(relative_segment)

                # Store chunk information
                chunk_info = {
                    "chunk_id": chunk_index,
                    "start_time": seconds_to_timestamp(group_start_seconds),
                    "end_time": seconds_to_timestamp(group_end_seconds),
                    "duration_seconds": group_end_seconds - group_start_seconds,
                    "file_path": chunk_path,
                    "file_size_bytes": os.path.getsize(chunk_path),
                    "segments": relative_segments
                }
                chunks.append(chunk_info)

                logger.debug(f"Created chunk {chunk_index}: {chunk_info['start_time']} - {chunk_info['end_time']} "
                           f"({len(segment_group)} segments)")

            logger.info(f"Successfully created {len(chunks)} segment-based audio chunks in: {output_dir}")
            return chunks
            
        except Exception as e:
            logger.error(f"Segment-based audio chunking failed: {str(e)}", exc_info=True)
            raise AudioChunkerError(f"Segment-based audio chunking failed: {str(e)}") from e


def chunk_audio_file(
    input_file_path: str,
    chunk_duration_minutes: float = 10.0,
    overlap_seconds: float = 5.0,
    output_dir: Optional[str] = None,
    output_prefix: str = "chunk"
) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk an audio file.
    
    Args:
        input_file_path: Path to the input audio file
        chunk_duration_minutes: Duration of each chunk in minutes
        overlap_seconds: Overlap duration between chunks in seconds
        output_dir: Directory to save chunks
        output_prefix: Prefix for output chunk files
        
    Returns:
        List of chunk information dictionaries
    """
    chunker = AudioChunker(chunk_duration_minutes, overlap_seconds)
    return chunker.chunk_audio_file(input_file_path, output_dir, output_prefix)


def chunk_audio_by_segments(
    input_file_path: str,
    segments_data: Dict,
    chunk_duration_minutes: float = 10.0,
    overlap_seconds: float = 0.0,
    output_dir: Optional[str] = None,
    output_prefix: str = "segment_chunk"
) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk an audio file based on segments.
    
    Args:
        input_file_path: Path to the input audio file
        segments_data: Dictionary containing segments information
        chunk_duration_minutes: Target duration for each chunk in minutes
        overlap_seconds: Overlap duration between chunks in seconds
        output_dir: Directory to save chunks
        output_prefix: Prefix for output chunk files
        
    Returns:
        List of chunk information dictionaries
    """
    chunker = AudioChunker(chunk_duration_minutes, overlap_seconds)
    return chunker.chunk_audio_by_segments(input_file_path, segments_data, output_dir, output_prefix)