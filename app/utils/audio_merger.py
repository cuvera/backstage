import logging
import os
import tempfile
import time
import subprocess
import json
from typing import Dict, List, Optional, Any
import asyncio
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import aiofiles
from app.utils.s3_client import get_s3_client, download_s3_file, upload_to_s3

from app.core.config import settings

logger = logging.getLogger(__name__)


class AudioMergerError(Exception):
    """Custom exception for audio merger operations."""
    pass


async def _get_audio_metadata_ffprobe(file_path: str) -> dict:
    """
    Get audio metadata using ffprobe without loading file into memory.

    Args:
        file_path: Path to the audio file

    Returns:
        Dictionary with duration and sample_rate

    Raises:
        AudioMergerError: If ffprobe fails
    """
    try:
        process = await asyncio.create_subprocess_exec(
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration:stream=sample_rate',
            '-of', 'json',
            file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise AudioMergerError(f"ffprobe failed: {stderr.decode()}")

        data = json.loads(stdout.decode())

        duration_str = data.get('format', {}).get('duration')
        if duration_str and duration_str != 'N/A':
            duration = float(duration_str)
        else:
            # Fallback: scan packets for last timestamp (handles WebM files
            # from browser MediaRecorder that lack duration metadata)
            logger.warning(f"Format duration unavailable, falling back to packet scanning: {file_path}")
            pkt_process = await asyncio.create_subprocess_exec(
                'ffprobe', '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'packet=pts_time',
                '-of', 'csv=p=0',
                file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            pkt_stdout, _ = await pkt_process.communicate()
            lines = [l.strip() for l in pkt_stdout.decode().strip().splitlines() if l.strip()]
            if lines:
                duration = float(lines[-1])
            else:
                raise AudioMergerError("Could not determine audio duration: no packets found")

        # Get sample rate from first audio stream
        sample_rate = 44100  # Default fallback
        if 'streams' in data and len(data['streams']) > 0:
            sample_rate = int(data['streams'][0].get('sample_rate', 44100))

        return {'duration': duration, 'sample_rate': sample_rate}
    except AudioMergerError:
        raise
    except Exception as e:
        raise AudioMergerError(f"Failed to get audio metadata: {e}")


async def _merge_audio_files_ffmpeg(
    input_files: List[str],
    output_dir: str,
    output_name: str = "merged_output",
    output_format: str = 'm4a'  # Default to M4A
) -> str:
    """
    Merge audio files using ffmpeg (no memory loading).

    Args:
        input_files: List of input file paths (any audio format)
        output_dir: Directory for output file
        output_name: Output filename without extension
        output_format: Output format ('m4a', 'wav', 'mp3', etc.). Default: 'm4a'

    Returns:
        Path to merged output file

    Raises:
        AudioMergerError: If merge fails
    """
    if not input_files:
        raise AudioMergerError("No input files provided for merging")

    # Normalize format (remove dot if provided)
    output_format = output_format.lstrip('.')

    # Build output path with specified format
    output_path = os.path.join(output_dir, f"{output_name}.{output_format}")

    # Create concat file list
    concat_list_path = output_path + '.concat.txt'
    async with aiofiles.open(concat_list_path, 'w') as f:
        for file_path in input_files:
            escaped_path = file_path.replace("'", "'\\''")
            await f.write(f"file '{escaped_path}'\n")

    try:
        # Check if we can use codec copy (all inputs match output format)
        input_formats = [Path(f).suffix.lower().lstrip('.') for f in input_files]
        can_copy = all(fmt == output_format for fmt in input_formats)

        if can_copy:
            # All inputs match output format: codec copy (fast, no re-encoding)
            logger.info(f"All inputs are {output_format}, using codec copy")
            process = await asyncio.create_subprocess_exec(
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list_path,
                '-c', 'copy',
                '-y',
                output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise AudioMergerError(f"ffmpeg merge failed: {stderr.decode()}")
        else:
            # Need to re-encode to output format
            logger.info(f"Converting inputs to {output_format}")

            # Choose encoder based on output format
            if output_format in ['m4a', 'mp4', 'aac']:
                codec_args = ['-c:a', 'aac', '-b:a', '128k']
            elif output_format == 'mp3':
                codec_args = ['-c:a', 'libmp3lame', '-b:a', '128k']
            elif output_format == 'wav':
                codec_args = ['-c:a', 'pcm_s16le']
            elif output_format in ['flac']:
                codec_args = ['-c:a', 'flac']
            else:
                # Default to AAC for unknown formats
                logger.warning(f"Unknown format '{output_format}', defaulting to AAC")
                codec_args = ['-c:a', 'aac', '-b:a', '128k']

            process = await asyncio.create_subprocess_exec(
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list_path,
                *codec_args,
                '-y',
                output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise AudioMergerError(f"ffmpeg merge failed: {stderr.decode()}")

        logger.info(f"Merged audio file created: {output_path}")
        return output_path

    finally:
        if await asyncio.to_thread(os.path.exists, concat_list_path):
            await asyncio.to_thread(os.remove, concat_list_path)


async def find_files_from_s3(
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

        # Get list of audio files - check for both .wav and .m4a (pre-merged file)
        all_audio_files = await list_files_in_s3_folder(
            s3_folder_path=s3_folder_path,
            bucket_name=bucket,
            file_extension=[".wav", ".m4a"]  # Check multiple formats
        )

        if not all_audio_files:
            raise AudioMergerError(f"No audio files found in s3://{bucket}/{s3_folder_path}")

        # Check for pre-merged file (meeting.wav, meeting.m4a, meeting.mp3, etc.)
        meeting_file = next((file for file in all_audio_files if Path(file).stem == "meeting"), None)

        if meeting_file:
            logger.info("Pre-merged meeting file found, downloading directly")
            # Get format from source file, default to m4a
            source_ext = Path(meeting_file).suffix.lower().lstrip('.')
            output_format = source_ext if source_ext else 'm4a'
            file_path = os.path.join(temp_dir, f"merged_output.{output_format}")
            await download_s3_file(meeting_file, file_path, bucket)

            logger.debug("Meeting file downloaded successfully")
            file_size = await asyncio.to_thread(os.path.getsize, file_path)

            # Calculate metadata using ffprobe (no memory loading)
            metadata = await _get_audio_metadata_ffprobe(file_path)
            duration_seconds = metadata['duration']

            result = {
                "local_merged_file_path": file_path,
                "merged_file_s3_key": output_s3_key,
                "total_files_merged": 1,  # Pre-merged file
                "total_duration_seconds": duration_seconds,
                "file_size_bytes": file_size,
                "temp_directory": temp_dir,
                "bucket_name": bucket
            }

            processing_time_ms = round((time.time() - merge_start_time) * 1000, 2)
            logger.info(f"Audio processing completed in {processing_time_ms}ms (using pre-merged file)")
            return result
        else:
            # No pre-merged file found, merge individual chunks
            # Filter to only .wav files (exclude any other formats)
            chunk_files = [f for f in all_audio_files if f.lower().endswith('.wav')]

            if not chunk_files:
                raise AudioMergerError(f"No .wav chunk files found in s3://{bucket}/{s3_folder_path}")

            if sort_files:
                chunk_files.sort()

            logger.info(f"No pre-merged file found, merging {len(chunk_files)} .wav chunks")

            # Download all files concurrently
            download_tasks = []
            local_files = []

            for i, s3_key in enumerate(chunk_files):
                local_file = os.path.join(temp_dir, f"audio_{i:03d}_{Path(s3_key).name}")
                local_files.append(local_file)
                download_tasks.append(download_s3_file(s3_key, local_file, bucket))
            
            download_results = await asyncio.gather(*download_tasks, return_exceptions=True)

            # Check for download failures
            failed_downloads = [
                (s3_key, result) for s3_key, result in zip(chunk_files, download_results)
                if isinstance(result, Exception) or not result
            ]

            if failed_downloads:
                failed_files = [s3_key for s3_key, _ in failed_downloads]
                logger.error(f"Failed to download {len(failed_files)} audio files from S3")
                raise AudioMergerError(f"Failed to download files: {failed_files}")

            logger.info(f"All {len(chunk_files)} audio files downloaded successfully")

            # Merge audio files using ffmpeg (no memory loading, outputs M4A by default)
            merged_file_path = await _merge_audio_files_ffmpeg(
                local_files,
                temp_dir,
                output_name="merged_output",
                output_format='m4a'  # Default to M4A for efficiency
            )

            logger.info(f"Merged audio exported to: {merged_file_path}")

            # Upload merged file to S3
            if output_s3_key:
                logger.debug("Uploading merged audio file to S3")
                upload_success = await upload_to_s3(merged_file_path, output_s3_key, bucket)
                if not upload_success:
                    logger.error("Failed to upload merged file to S3")
                    raise AudioMergerError(f"Failed to upload merged file to s3://{bucket}/{output_s3_key}")

            # Calculate metadata using ffprobe (no memory loading)
            file_size = await asyncio.to_thread(os.path.getsize, merged_file_path)
            metadata = await _get_audio_metadata_ffprobe(merged_file_path)
            duration_seconds = metadata['duration']
        
            result = {
                "local_merged_file_path": merged_file_path,
                "merged_file_s3_key": output_s3_key,
                "total_files_merged": len(chunk_files),
                "total_duration_seconds": duration_seconds,
                "file_size_bytes": file_size,
                "bucket_name": bucket
            }

            processing_time_ms = round((time.time() - merge_start_time) * 1000, 2)
            logger.info(f"Audio merge completed successfully in {processing_time_ms}ms - merged {len(chunk_files)} files")
            return result
        
    except Exception as e:
        logger.error(f"Audio merge operation failed: {str(e)}", exc_info=True)
        raise AudioMergerError(f"Audio merge operation failed: {str(e)}") from e
    
    finally:
        cleanup_start_time = time.time()
        cleaned_files_count = 0

        # Cleanup temporary files (except merged_output.* which orchestrator handles)
        if temp_dir and await asyncio.to_thread(os.path.exists, temp_dir):
            import shutil
            try:
                files_list = await asyncio.to_thread(os.listdir, temp_dir)
                for file in files_list:
                    # Keep merged output file (any extension: .m4a, .wav, etc.)
                    if not file.startswith("merged_output."):
                        file_path = os.path.join(temp_dir, file)
                        if await asyncio.to_thread(os.path.isfile, file_path):
                            await asyncio.to_thread(os.remove, file_path)
                            cleaned_files_count += 1
                            logger.debug(f"Cleaned up intermediate file: {file}")
                        elif await asyncio.to_thread(os.path.isdir, file_path):
                            await asyncio.to_thread(shutil.rmtree, file_path)
                            cleaned_files_count += 1
                            logger.debug(f"Cleaned up intermediate directory: {file}")

                if cleaned_files_count > 0:
                    logger.info(f"Cleaned up {cleaned_files_count} intermediate files")
            except Exception as e:
                logger.warning(f"Failed to cleanup intermediate files: {str(e)}", exc_info=True)


async def list_files_in_s3_folder(
    s3_folder_path: str,
    bucket_name: Optional[str] = None,
    file_extension: Any = ".wav"
) -> List[str]:
    """
    List all audio files in an S3 folder.

    Args:
        s3_folder_path: S3 folder path
        bucket_name: S3 bucket name
        file_extension: File extension(s) to filter - can be a string or list of strings

    Returns:
        List of S3 keys for audio files
    """
    bucket = bucket_name or settings.S3_BUCKET_NAME

    try:
        s3_client = get_s3_client()

        # Ensure folder path ends with '/'
        if not s3_folder_path.endswith('/'):
            s3_folder_path += '/'

        logger.info(f"Listing files in S3 folder: {bucket}/{s3_folder_path}")

        # Convert file_extension to list if it's a string
        extensions = [file_extension] if isinstance(file_extension, str) else file_extension

        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=s3_folder_path)

        audio_files = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Skip folders and check if file matches any extension
                    if not key.endswith('/') and any(key.lower().endswith(ext.lower()) for ext in extensions):
                        audio_files.append(key)

        ext_str = ', '.join(extensions)
        logger.info(f"Found {len(audio_files)} files with extensions [{ext_str}] in S3")
        return audio_files
        
    except Exception as e:
        logger.error(f"Failed to list S3 files: {str(e)}", exc_info=True)
        raise AudioMergerError(f"Failed to list S3 files: {str(e)}") from e



# if __name__ == "__main__":
    # merge_response = merge_wav_files_from_s3(
    #     s3_folder_path="689ddc0411e4209395942bee/offline/68efa4e55673acfc41f2bad1",
    #     output_s3_key="689ddc0411e4209395942bee/offline/68efa4e55673acfc41f2bad1/meeting.wav"
    # )

    # print(f"merge_response: {merge_response}")