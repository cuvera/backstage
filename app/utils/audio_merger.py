import logging
import os
import tempfile
import time
from typing import Dict, List, Optional, Any
import asyncio
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import audiofile
import numpy as np
import aiofiles
from app.utils.s3_client import get_s3_client, download_s3_file, upload_to_s3

from app.core.config import settings

logger = logging.getLogger(__name__)


class AudioMergerError(Exception):
    """Custom exception for audio merger operations."""
    pass


async def merge_wav_files_from_s3(
    s3_folder_path: str,
    output_s3_key: str,
    bucket_name: Optional[str] = None,
    file_extension: str = ".wav",
    sort_files: bool = True,
    temp_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Merge all WAV files from an S3 folder into a single audio file.
    
    Args:
        s3_folder_path: S3 folder path (e.g., "audio/recordings/session-123/")
        output_s3_key: Output file S3 key (e.g., "audio/merged/session-123-merged.wav")
        bucket_name: S3 bucket name (uses default from settings if None)
        file_extension: File extension to filter (default: ".wav")
        sort_files: Sort files alphabetically before merging (default: True)
        cleanup_temp: Clean up temporary files after processing (default: True)
    
    Returns:
        Dict containing merge operation results
        
    Raises:
        AudioMergerError: For various audio processing errors
    """
    bucket = bucket_name or settings.S3_BUCKET_NAME
    
    try:
        merge_start_time = time.time()
        logger.info("Starting audio merge operation")
        
        # Get list of WAV files in the S3 folder
        wav_files = await list_wav_files_in_s3_folder(
            s3_folder_path=s3_folder_path,
            bucket_name=bucket,
            file_extension=file_extension
        )
        
        if not wav_files:
            raise AudioMergerError(f"No {file_extension} files found in s3://{bucket}/{s3_folder_path}")
        
        if sort_files:
            wav_files.sort()
        
        logger.info(f"Found {len(wav_files)} audio files for merging")

        meeting_file = next((file for file in wav_files if Path(file).name == "meeting.wav"), None)
        if meeting_file:
            logger.info("Pre-merged meeting file found, downloading directly")
            file_path = os.path.join(temp_dir, "merged_output.wav")
            await download_s3_file(meeting_file, file_path, bucket)

            logger.debug("Meeting file downloaded successfully")
            file_size = os.path.getsize(file_path)

            # Calculate metadata
            signal, sampling_rate = audiofile.read(file_path)
            duration_seconds = len(signal) / sampling_rate

            result = {
                "file": file_path,
                "local_merged_file_path": file_path,
                "merged_file_s3_key": output_s3_key,
                "total_files_merged": len(wav_files),
                "total_duration_seconds": duration_seconds,
                "file_size_bytes": file_size,
                "temp_directory": temp_dir,
                "bucket_name": bucket
            }
            
            processing_time_ms = round((time.time() - merge_start_time) * 1000, 2)
            logger.info(f"Audio processing completed in {processing_time_ms}ms")
            return result
        else:
            # Download all files concurrently
            download_tasks = []
            local_files = []
            
            for i, s3_key in enumerate(wav_files):
                local_file = os.path.join(temp_dir, f"audio_{i:03d}_{Path(s3_key).name}")
                local_files.append(local_file)
                download_tasks.append(download_s3_file(s3_key, local_file, bucket))
            
            download_results = await asyncio.gather(*download_tasks, return_exceptions=True)
            
            # Check for download failures
            failed_downloads = [
                (s3_key, result) for s3_key, result in zip(wav_files, download_results)
                if isinstance(result, Exception) or not result
            ]
            
            if failed_downloads:
                failed_files = [s3_key for s3_key, _ in failed_downloads]
                logger.error(f"Failed to download {len(failed_files)} audio files from S3")
                raise AudioMergerError(f"Failed to download files: {failed_files}")
            
            logger.info(f"All {len(wav_files)} audio files downloaded successfully")
            
            # Validate and merge audio files
            merged_audio, sample_rate = await merge_local_audio_files(local_files)
            
            # Export merged audio to temporary file
            merged_file_path = os.path.join(temp_dir, "merged_output.wav")
            audiofile.write(merged_file_path, merged_audio, sample_rate)
        
            logger.info(f"Merged audio exported to: {merged_file_path}")
        
            # Upload merged file to S3
            if output_s3_key:
                logger.debug("Uploading merged audio file to S3")
                upload_success = await upload_to_s3(merged_file_path, output_s3_key, bucket)
                if not upload_success:
                    logger.error("Failed to upload merged file to S3")
                    raise AudioMergerError(f"Failed to upload merged file to s3://{bucket}/{output_s3_key}")
        
            # Calculate metadata
            file_size = os.path.getsize(merged_file_path)
            duration_seconds = len(merged_audio) / sample_rate
        
            result = {
                "local_merged_file_path": merged_file_path,
                "merged_file_s3_key": output_s3_key,
                "total_files_merged": len(wav_files),
                "total_duration_seconds": duration_seconds,
                "file_size_bytes": file_size,
                "bucket_name": bucket
            }
        
            processing_time_ms = round((time.time() - merge_start_time) * 1000, 2)
            logger.info(f"Audio merge completed successfully in {processing_time_ms}ms - merged {len(wav_files)} files")
            return result
        
    except Exception as e:
        logger.error(f"Audio merge operation failed: {str(e)}", exc_info=True)
        raise AudioMergerError(f"Audio merge operation failed: {str(e)}") from e
    
    finally:
        cleanup_start_time = time.time()
        cleaned_files_count = 0
        
        # Cleanup temporary files (except merged_output.wav which orchestrator handles)
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            try:
                for file in os.listdir(temp_dir):
                    if file != "merged_output.wav":
                        file_path = os.path.join(temp_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            cleaned_files_count += 1
                            logger.debug(f"Cleaned up intermediate file: {file}")
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            cleaned_files_count += 1
                            logger.debug(f"Cleaned up intermediate directory: {file}")
                
                if cleaned_files_count > 0:
                    logger.info(f"Cleaned up {cleaned_files_count} intermediate files")
            except Exception as e:
                logger.warning(f"Failed to cleanup intermediate files: {str(e)}", exc_info=True)


async def list_wav_files_in_s3_folder(
    s3_folder_path: str,
    bucket_name: Optional[str] = None,
    file_extension: str = ".wav"
) -> List[str]:
    """
    List all audio files in an S3 folder.
    
    Args:
        s3_folder_path: S3 folder path
        bucket_name: S3 bucket name
        file_extension: File extension to filter
    
    Returns:
        List of S3 keys for audio files
    """
    bucket = bucket_name or settings.S3_BUCKET_NAME
    
    try:
        s3_client = get_s3_client()
        
        # Ensure folder path ends with '/'
        if not s3_folder_path.endswith('/'):
            s3_folder_path += '/'
        
        logger.debug("Listing files in S3 folder")
        
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=s3_folder_path)
        
        audio_files = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Skip folders and non-audio files
                    if not key.endswith('/') and key.lower().endswith(file_extension.lower()):
                        audio_files.append(key)
        
        logger.info(f"Found {len(audio_files)} {file_extension} files in S3")
        return audio_files
        
    except Exception as e:
        logger.error(f"Failed to list S3 files: {str(e)}", exc_info=True)
        raise AudioMergerError(f"Failed to list S3 files: {str(e)}") from e

async def merge_local_audio_files(file_paths: List[str]) -> tuple[np.ndarray, int]:
    """
    Merge multiple local audio files into a single audio array.
    
    Args:
        file_paths: List of local audio file paths
    
    Returns:
        Tuple of (merged_audio_data, sample_rate)
    """
    if not file_paths:
        logger.info("No files provided for merging")
    
    logger.info(f"Starting local audio file merge - {len(file_paths)} files")
    merge_start_time = time.time()
    
    merged_audio = None
    sample_rate = None
    
    for i, file_path in enumerate(file_paths):
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise AudioMergerError(f"File not found: {file_path}")
            
            # Load audio file
            signal, sr = audiofile.read(file_path)
            
            # Convert to mono if stereo
            if signal.ndim > 1:
                signal = np.mean(signal, axis=1)
            
            # Set sample rate from first file
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                # Resample if needed (basic implementation)
                logger.warning(f"Sample rate mismatch: {sr} vs {sample_rate}. Using first file's rate.")
            
            logger.debug(f"Processed audio file {i+1}/{len(file_paths)}")
            
            # Merge with previous audio
            if merged_audio is None:
                merged_audio = signal
            else:
                merged_audio = np.concatenate([merged_audio, signal])
                
        except Exception as e:
            logger.error(f"Failed to process audio file {file_path}: {str(e)}", exc_info=True)
            raise AudioMergerError(f"Failed to process audio file {file_path}: {str(e)}") from e
    
    if merged_audio is None:
        logger.info("No audio content was merged")
    
    processing_time_ms = round((time.time() - merge_start_time) * 1000, 2)
    total_duration = round(len(merged_audio) / sample_rate, 2)
    logger.info(f"Local audio merge completed in {processing_time_ms}ms - {len(file_paths)} files, {total_duration}s total duration")
    return merged_audio, sample_rate


# if __name__ == "__main__":
    # merge_response = merge_wav_files_from_s3(
    #     s3_folder_path="689ddc0411e4209395942bee/offline/68efa4e55673acfc41f2bad1",
    #     output_s3_key="689ddc0411e4209395942bee/offline/68efa4e55673acfc41f2bad1/meeting.wav"
    # )

    # print(f"merge_response: {merge_response}")