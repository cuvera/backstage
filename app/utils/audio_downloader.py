"""
Audio downloader utility
Downloads audio files from URLs using httpx
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any
import httpx
import aiofiles

from app.utils.audio_chunker import get_base_dir

logger = logging.getLogger(__name__)


async def download_audio(
    audio_url: str,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Download audio from URL to local filesystem using httpx

    Args:
        audio_url: URL to audio file (HTTP/HTTPS, including pre-signed S3/R2 URLs)
        output_dir: Directory to save file (defaults to temp directory)

    Returns:
        Dict with:
            - local_path: Path to downloaded file
            - file_size_bytes: Size of file in bytes
            - mime_type: MIME type of file
            - download_time_ms: Time taken to download

    Raises:
        Exception: If download fails
    """
    start_time = time.time()

    # Create output directory
    if output_dir is None:
        output_dir = get_base_dir()
    else:
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Downloading audio | url={audio_url[:80]}...")

    # Extract filename from URL (without query params)
    filename = Path(audio_url.split("?")[0]).name or f"audio_{int(time.time())}.m4a"
    local_path = os.path.join(output_dir, filename)

    # Download using httpx
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream("GET", audio_url) as response:
            response.raise_for_status()

            async with aiofiles.open(local_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    await f.write(chunk)

    # Get file metadata
    file_size = os.path.getsize(local_path)
    mime_type = _get_mime_type(local_path)
    download_time_ms = (time.time() - start_time) * 1000

    logger.info(f"Download complete | size={file_size}B time={download_time_ms:.0f}ms")

    return {
        "local_path": local_path,
        "file_size_bytes": file_size,
        "mime_type": mime_type,
        "download_time_ms": download_time_ms
    }


def _get_mime_type(file_path: str) -> str:
    """Get MIME type from file extension"""
    import mimetypes

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type

    # Fallback based on extension
    ext = Path(file_path).suffix.lower()
    mime_map = {
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".aac": "audio/aac",
        ".webm": "audio/webm"
    }
    return mime_map.get(ext, "audio/mp4")
