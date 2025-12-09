import logging
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

import audiofile
import soundfile as sf
import numpy as np

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
    """Utility for chunking .m4a audio files into segments with overlap."""
    
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
        
    def chunk_audio_file(
        self,
        input_file_path: str,
        output_dir: Optional[str] = None,
        output_prefix: str = "chunk"
    ) -> List[Dict[str, Any]]:
        """
        Chunk a .m4a audio file into segments with overlap.
        
        Args:
            input_file_path: Path to the input .m4a file
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
            
            if not input_path.suffix.lower() in ['.m4a', '.wav', '.mp3', '.flac']:
                raise AudioChunkerError(f"Unsupported audio format: {input_path.suffix}")
            
            # Setup output directory
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix="audio_chunks_")
            else:
                os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Starting audio chunking for: {input_file_path}")
            
            # Load audio file using audiofile
            audio_data, sample_rate = audiofile.read(input_file_path)
            
            # Convert to mono if stereo
            # audiofile returns shape (channels, samples), so average across channels (axis=0)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            total_duration_seconds = len(audio_data) / sample_rate
            logger.info(f"Audio duration: {total_duration_seconds:.2f} seconds")
            
            # Calculate chunk parameters in samples
            chunk_duration_samples = int(self.chunk_duration_seconds * sample_rate)
            overlap_samples = int(self.overlap_seconds * sample_rate)
            
            chunks = []
            chunk_index = 0
            start_sample = 0
            
            while start_sample < len(audio_data):
                # Calculate end sample for this chunk
                end_sample = min(start_sample + chunk_duration_samples, len(audio_data))
                
                # Extract chunk
                chunk_audio = audio_data[start_sample:end_sample]

                # Generate output filename
                chunk_filename = f"{output_prefix}_{chunk_index:03d}.m4a"
                chunk_path = os.path.join(output_dir, chunk_filename)

                # Export chunk - write to temp WAV first, then convert to M4A
                temp_wav_path = os.path.join(output_dir, f"temp_{chunk_index:03d}.wav")
                sf.write(temp_wav_path, chunk_audio, sample_rate)

                # Convert WAV to M4A using ffmpeg
                subprocess.run([
                    'ffmpeg', '-i', temp_wav_path,
                    '-c:a', 'aac', '-b:a', '128k',
                    '-y',  # Overwrite output file if exists
                    chunk_path
                ], check=True, capture_output=True)

                # Remove temporary WAV file
                os.remove(temp_wav_path)
                
                # Store chunk information
                start_time_seconds = start_sample / sample_rate
                end_time_seconds = end_sample / sample_rate
                chunk_info = {
                    "chunk_id": chunk_index,
                    "start_time": seconds_to_timestamp(start_time_seconds),
                    "end_time": seconds_to_timestamp(end_time_seconds),
                    "duration_seconds": end_time_seconds - start_time_seconds,
                    "file_path": chunk_path,
                    "file_size_bytes": os.path.getsize(chunk_path),
                    "segments": []
                }
                chunks.append(chunk_info)
                
                logger.debug(f"Created chunk {chunk_index}: {chunk_info['start_time']} - {chunk_info['end_time']}")
                
                # Move to next chunk start (with overlap consideration)
                if end_sample >= len(audio_data):
                    break
                
                start_sample = end_sample - overlap_samples
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
        Chunk audio file based on provided segments, grouping them into ~10min chunks.
        
        Args:
            input_file_path: Path to the input .m4a file
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
            
            # Load audio file using audiofile
            audio_data, sample_rate = audiofile.read(input_file_path)
            
            # Convert to mono if stereo
            # audiofile returns shape (channels, samples), so average across channels (axis=0)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            # Group segments by target duration
            grouped_segments = group_segments_by_duration(segments, self.chunk_duration_minutes)
            
            logger.info(f"Created {len(grouped_segments)} chunk groups from segments")
            
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
                    group_end_seconds = group_end_seconds + self.overlap_seconds
                
                # Convert to sample indices
                start_sample = int(group_start_seconds * sample_rate)
                end_sample = int(group_end_seconds * sample_rate)
                
                # Ensure we don't exceed audio bounds
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_data), end_sample)
                
                # Extract chunk
                chunk_audio = audio_data[start_sample:end_sample]

                # Generate output filename
                chunk_filename = f"{output_prefix}_{chunk_index:03d}.m4a"
                chunk_path = os.path.join(output_dir, chunk_filename)

                # Export chunk - write to temp WAV first, then convert to M4A
                temp_wav_path = os.path.join(output_dir, f"temp_{chunk_index:03d}.wav")
                sf.write(temp_wav_path, chunk_audio, sample_rate)

                # Convert WAV to M4A using ffmpeg
                subprocess.run([
                    'ffmpeg', '-i', temp_wav_path,
                    '-c:a', 'aac', '-b:a', '128k',
                    '-y',  # Overwrite output file if exists
                    chunk_path
                ], check=True, capture_output=True)

                # Remove temporary WAV file
                os.remove(temp_wav_path)
                
                # Store chunk information
                chunk_info = {
                    "chunk_id": chunk_index,
                    "start_time": seconds_to_timestamp(group_start_seconds),
                    "end_time": seconds_to_timestamp(group_end_seconds),
                    "duration_seconds": group_end_seconds - group_start_seconds,
                    "file_path": chunk_path,
                    "file_size_bytes": os.path.getsize(chunk_path),
                    "segments": segment_group
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
    overlap_seconds: float = 5.0,
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