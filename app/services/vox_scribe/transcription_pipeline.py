import logging
import os
import tempfile
from typing import Any, Dict, List
import numpy as np
import torch
import torchaudio
import soundfile as sf
from app.services.vox_scribe.transcription_diarization_service import TranscriptionDiarizationService

logger = logging.getLogger(__name__)


def extract_text_from_transcription_result(transcription_result):
    """
    Extract clean text from various transcription result formats.
    
    Args:
        transcription_result: Can be string, dict, list, or nested structures
        
    Returns:
        Clean concatenated text string
    """
    if not transcription_result:
        return ""
    
    text_segments = []
    
    def extract_text_recursive(item):
        """Recursively extract text from nested structures."""
        if isinstance(item, dict):
            # Look for common text keys
            for key in ['text', 'transcript', 'content']:
                if key in item and isinstance(item[key], str):
                    text = item[key].strip()
                    if text:
                        text_segments.append(text)
                        return
            
            # If no direct text found, recurse through values
            for value in item.values():
                extract_text_recursive(value)
                
        elif isinstance(item, list):
            for sub_item in item:
                extract_text_recursive(sub_item)
                
        elif isinstance(item, str):
            text = item.strip()
            if text:
                text_segments.append(text)
    
    extract_text_recursive(transcription_result)
    
    # Join all extracted text segments
    final_text = " ".join(text_segments)
    
    # Clean up the final text
    return final_text.strip()


def transcription_with_timeframes(
    meeting_audio_path: str,
    speaker_timeframes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Transcribe audio using pre-identified speaker timeframes from Google meetings.
    
    Args:
        meeting_audio_path: Path to the merged audio file
        speaker_timeframes: List of speaker segments from google_meetings collection
                           Format: [{"speakerName": "Name", "start": ms, "end": ms}, ...]
    
    Returns:
        List of transcription entries in same format as diarization_pipeline:
        [{"start_time": seconds, "end_time": seconds, "speaker": "Name", 
          "text": "...", "identification_score": 1.0}, ...]
    """
    if not speaker_timeframes:
        logger.warning("No speaker timeframes provided, returning empty result")
        return []
    
    if not os.path.exists(meeting_audio_path):
        logger.error(f"Audio file not found: {meeting_audio_path}")
        return []
    
    try:
        # Validate audio file before processing
        try:
            info = sf.info(meeting_audio_path)
            logger.info(f"Audio file info: {info.frames} frames, {info.samplerate} Hz, {info.channels} channels, {info.duration:.2f}s")
        except Exception as e:
            logger.error(f"Cannot read audio file info: {e}")
            return []
        
        # Load audio using soundfile (better for stereo PCM files)
        logger.info(f"Loading audio file: {meeting_audio_path}")
        target_sample_rate = 16000
        audio_data, sample_rate = sf.read(meeting_audio_path, dtype='float32')
        logger.info(f"Loaded audio: {audio_data.shape} shape, {sample_rate} Hz")
        
        # Handle stereo to mono conversion (since timeframes expect mono)
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            logger.info(f"Converting stereo ({audio_data.shape[1]} channels) to mono")
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sample_rate != target_sample_rate:
            logger.info(f"Resampling from {sample_rate} Hz to {target_sample_rate} Hz") 
            # Convert to tensor for resampling
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, target_sample_rate)
            audio_data = audio_tensor.squeeze().numpy()
            sample_rate = target_sample_rate

        logger.info(f"Final audio: {len(audio_data)} samples at {sample_rate} Hz")
        
        # Check if audio is too short (less than 1 second)
        min_audio_duration = 1.0  # seconds
        audio_duration = len(audio_data) / sample_rate
        if audio_duration < min_audio_duration:
            logger.error(f"Audio file too short: {audio_duration:.3f}s (minimum {min_audio_duration}s required)")
            logger.error(f"This suggests an issue with the audio file: {meeting_audio_path}")
            return []
        
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Initialize transcription service
        transcription_service = TranscriptionDiarizationService()
        
        transcription_results = []
        
        # Process each speaker timeframe
        for i, timeframe in enumerate(speaker_timeframes):
            try:
                speaker_name = timeframe.get("speakerName", f"Speaker_{i}")
                start_ms = timeframe.get("start", 0)
                end_ms = timeframe.get("end", 0)
                
                # Convert milliseconds to seconds
                start_seconds = start_ms / 1000.0
                end_seconds = end_ms / 1000.0
                
                # Skip zero-duration segments
                if start_ms == end_ms:
                    logger.info(f"Skipping timeframe {i+1}/{len(speaker_timeframes)}: "
                               f"{speaker_name} - zero duration segment")
                    continue
                
                logger.info(f"Processing timeframe {i+1}/{len(speaker_timeframes)}: "
                           f"{speaker_name} [{start_seconds:.2f}s - {end_seconds:.2f}s]")
                
                # Convert time to sample indices
                start_sample = int(start_seconds * sample_rate)
                end_sample = int(end_seconds * sample_rate)
                
                # Extract audio segment
                if start_sample >= len(audio_data) or end_sample > len(audio_data):
                    logger.warning(f"Timeframe {i+1} extends beyond audio length, skipping")
                    continue
                
                audio_segment = audio_data[start_sample:end_sample]
                
                # Skip very short segments (less than 0.5 seconds)
                if len(audio_segment) < sample_rate * 0.5:
                    logger.warning(f"Timeframe {i+1} too short ({len(audio_segment)/sample_rate:.2f}s), skipping")
                    continue
                
                # Save segment to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_audio_path = temp_file.name
                    sf.write(temp_audio_path, audio_segment, sample_rate)
                
                try:
                    # Transcribe the segment
                    logger.info(f"Transcribing segment for {speaker_name} (duration: {end_seconds - start_seconds:.2f}s)")
                    transcription_result = transcription_service.transcribe(temp_audio_path)
                    
                    # Log raw transcription result for debugging
                    logger.debug(f"Raw transcription result type: {type(transcription_result)}")
                    logger.debug(f"Raw transcription result: {repr(transcription_result)}")
                    
                    # Extract clean text from transcription result
                    transcription_text = extract_text_from_transcription_result(transcription_result)
                    
                    if transcription_text and transcription_text.strip():
                        transcription_results.append({
                            "start_time": start_seconds,
                            "end_time": end_seconds,
                            "speaker": speaker_name,
                            "text": transcription_text.strip(),
                            "identification_score": 1.0  # High confidence since speaker is pre-identified
                        })
                        logger.info(f"Successfully transcribed: {speaker_name}: '{transcription_text[:50]}...'")
                    else:
                        logger.warning(f"No transcription result for {speaker_name} segment (duration: {end_seconds - start_seconds:.2f}s)")
                        logger.warning(f"Raw transcription result: {repr(transcription_result)}")
                
                except Exception as transcription_error:
                    logger.error(f"Transcription failed for {speaker_name}: {transcription_error}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                        
            except Exception as e:
                logger.error(f"Error processing timeframe {i+1}: {e}")
                continue
        
        # Sort results by start time
        transcription_results.sort(key=lambda x: x["start_time"])
        
        logger.info(f"Transcription complete: {len(transcription_results)} segments processed out of {len(speaker_timeframes)} total timeframes")
        
        if len(transcription_results) == 0:
            logger.warning("No transcription results generated! This will cause analysis to fail.")
            logger.warning("Possible causes: all timeframes were zero-duration, transcription failed, or audio issues")
        
        return transcription_results
        
    except Exception as e:
        logger.error(f"Error in transcription_with_timeframes: {e}")
        return []


def validate_speaker_timeframes(speaker_timeframes: List[Dict[str, Any]]) -> bool:
    """
    Validate the speaker timeframes format.
    
    Args:
        speaker_timeframes: List of timeframe dictionaries
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(speaker_timeframes, list):
        logger.error("speaker_timeframes must be a list")
        return False
    
    for i, timeframe in enumerate(speaker_timeframes):
        if not isinstance(timeframe, dict):
            logger.error(f"Timeframe {i} must be a dictionary")
            return False
        
        required_fields = ["speakerName", "start", "end"]
        for field in required_fields:
            if field not in timeframe:
                logger.error(f"Timeframe {i} missing required field: {field}")
                return False
        
        start = timeframe.get("start")
        end = timeframe.get("end")
        
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            logger.error(f"Timeframe {i} start/end must be numbers")
            return False
        
        if start > end:
            logger.error(f"Timeframe {i} start ({start}) cannot be greater than end ({end})")
            return False
        
        if start == end:
            logger.warning(f"Timeframe {i} has zero duration (start == end: {start}), will be skipped during processing")
            # Don't fail validation for zero-duration segments, just warn
        
        if start < 0:
            logger.error(f"Timeframe {i} start time cannot be negative")
            return False
    
    return True