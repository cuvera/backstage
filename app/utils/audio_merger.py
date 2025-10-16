import logging
import os
import tempfile
from typing import Dict, List, Optional, Any
import asyncio
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pydub import AudioSegment
import aiofiles

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
    cleanup_temp: bool = True
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
    temp_dir = None
    
    try:
        print("Starting audio merge for folder: s3://{bucket}/{s3_folder_path}")
        logger.info(f"Starting audio merge for folder: s3://{bucket}/{s3_folder_path}")
        
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
        
        logger.info(f"Found {len(wav_files)} files to merge: {wav_files}")
        
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix="audio_merge_")
        logger.info(f"Created temporary directory: {temp_dir}")
        
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
            raise AudioMergerError(f"Failed to download files: {failed_files}")
        
        logger.info("All files downloaded successfully")
        
        # Validate and merge audio files
        merged_audio = await merge_local_audio_files(local_files)
        
        # Export merged audio to temporary file
        merged_file_path = os.path.join(temp_dir, "merged_output.wav")
        merged_audio.export(merged_file_path, format="wav")
        
        logger.info(f"Merged audio exported to: {merged_file_path}")
        
        # Upload merged file to S3
        upload_success = await upload_to_s3(merged_file_path, output_s3_key, bucket)
        if not upload_success:
            raise AudioMergerError(f"Failed to upload merged file to s3://{bucket}/{output_s3_key}")
        
        # Calculate metadata
        file_size = os.path.getsize(merged_file_path)
        duration_seconds = len(merged_audio) / 1000.0  # pydub uses milliseconds
        
        result = {
            "merged_file_s3_key": output_s3_key,
            "total_files_merged": len(wav_files),
            "total_duration_seconds": duration_seconds,
            "file_size_bytes": file_size,
            "source_files": [Path(f).name for f in wav_files],
            "bucket_name": bucket
        }
        
        logger.info(f"Audio merge completed successfully: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Audio merge failed: {str(e)}")
        raise AudioMergerError(f"Audio merge operation failed: {str(e)}") from e
    
    finally:
        # Cleanup temporary files
        if cleanup_temp and temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")


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
        s3_client = _get_s3_client()
        
        # Ensure folder path ends with '/'
        if not s3_folder_path.endswith('/'):
            s3_folder_path += '/'
        
        logger.info(f"Listing files in s3://{bucket}/{s3_folder_path}")
        
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
        
        logger.info(f"Found {len(audio_files)} {file_extension} files")
        return audio_files
        
    except Exception as e:
        logger.error(f"Failed to list files in s3://{bucket}/{s3_folder_path}: {e}")
        raise AudioMergerError(f"Failed to list S3 files: {str(e)}") from e


async def download_s3_file(
    s3_key: str,
    local_path: str,
    bucket_name: Optional[str] = None
) -> bool:
    """
    Download a file from S3 to local path.
    
    Args:
        s3_key: S3 object key
        local_path: Local file path to save to
        bucket_name: S3 bucket name
    
    Returns:
        True if successful, False otherwise
    """
    bucket = bucket_name or settings.S3_BUCKET_NAME
    
    try:
        s3_client = _get_s3_client()
        
        logger.debug(f"Downloading s3://{bucket}/{s3_key} to {local_path}")
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        s3_client.download_file(bucket, s3_key, local_path)
        
        logger.debug(f"Successfully downloaded {s3_key}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download s3://{bucket}/{s3_key}: {e}")
        return False


async def upload_to_s3(
    local_path: str,
    s3_key: str,
    bucket_name: Optional[str] = None
) -> bool:
    """
    Upload a local file to S3.
    
    Args:
        local_path: Local file path
        s3_key: S3 object key
        bucket_name: S3 bucket name
    
    Returns:
        True if successful, False otherwise
    """
    bucket = bucket_name or settings.S3_BUCKET_NAME
    
    try:
        s3_client = _get_s3_client()
        
        logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
        
        # Set content type for WAV files
        extra_args = {}
        if s3_key.lower().endswith('.wav'):
            extra_args['ContentType'] = 'audio/wav'
        
        s3_client.upload_file(local_path, bucket, s3_key, ExtraArgs=extra_args)
        
        logger.info(f"Successfully uploaded to s3://{bucket}/{s3_key}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload to s3://{bucket}/{s3_key}: {e}")
        return False


async def merge_local_audio_files(file_paths: List[str]) -> AudioSegment:
    """
    Merge multiple local audio files into a single AudioSegment.
    
    Args:
        file_paths: List of local audio file paths
    
    Returns:
        Merged AudioSegment
    """
    if not file_paths:
        raise AudioMergerError("No files provided for merging")
    
    logger.info(f"Merging {len(file_paths)} audio files")
    
    merged_audio = None
    
    for i, file_path in enumerate(file_paths):
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise AudioMergerError(f"File not found: {file_path}")
            
            # Load audio file
            audio = AudioSegment.from_wav(file_path)
            
            # Normalize audio properties (convert to mono, keep original sample rate)
            audio = audio.set_channels(1)  # Convert to mono
            
            logger.debug(f"Loaded audio file {i+1}/{len(file_paths)}: {file_path} "
                        f"(duration: {len(audio)/1000:.2f}s)")
            
            # Merge with previous audio
            if merged_audio is None:
                merged_audio = audio
            else:
                merged_audio += audio
                
        except Exception as e:
            logger.error(f"Failed to process audio file {file_path}: {e}")
            raise AudioMergerError(f"Failed to process audio file {file_path}: {str(e)}") from e
    
    if merged_audio is None:
        raise AudioMergerError("No audio content was merged")
    
    logger.info(f"Successfully merged audio (total duration: {len(merged_audio)/1000:.2f}s)")
    return merged_audio


def _get_s3_client():
    """Get configured S3 client."""
    try:
        # Configure S3 client with settings
        client_config = {
            'region_name': settings.AWS_REGION,
        }
        
        # Add credentials if provided
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            client_config.update({
                'aws_access_key_id': settings.AWS_ACCESS_KEY_ID,
                'aws_secret_access_key': settings.AWS_SECRET_ACCESS_KEY,
            })
        
        # Add custom endpoint if provided (for S3-compatible services)
        if settings.AWS_ENDPOINT:
            client_config['endpoint_url'] = settings.AWS_ENDPOINT
        
        return boto3.client('s3', **client_config)
        
    except NoCredentialsError:
        raise AudioMergerError("AWS credentials not found. Please configure AWS credentials.")
    except Exception as e:
        raise AudioMergerError(f"Failed to create S3 client: {str(e)}") from e


if __name__ == "__main__":
    # merge_response = merge_wav_files_from_s3(
    #     s3_folder_path="689ddc0411e4209395942bee/offline/68efa4e55673acfc41f2bad1",
    #     output_s3_key="689ddc0411e4209395942bee/offline/68efa4e55673acfc41f2bad1/meeting.wav"
    # )

    # print(f"merge_response: {merge_response}")