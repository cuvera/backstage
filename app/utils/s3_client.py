import logging
import os
from typing import Optional
import boto3
from botocore.exceptions import NoCredentialsError
from app.core.config import settings
from app.utils.audio_merger import AudioMergerError

logger = logging.getLogger(__name__)

def get_s3_client():
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
        s3_client = get_s3_client()
        
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
        s3_client = get_s3_client()
        
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
