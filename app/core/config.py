from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import logging

class Settings(BaseSettings):
    SERVICE_NAME: Optional[str] = Field(default="backstage", env="SERVICE_NAME")
    
    MONGODB_URL: Optional[str] = Field(default="mongodb://localhost:27017", env="MONGODB_URL")
    DATABASE_NAME: Optional[str] = Field(default="cognitive", env="DATABASE_NAME")
    
    LLM_FALLBACK_CHAIN: Optional[str] = Field(default="gemini-2.5-pro,gemini-3.0-flash-preview,gpt-4.1", env="LLM_FALLBACK_CHAIN")
    LLM_TIMEOUT: Optional[float] = Field(default=600, env="LLM_TIMEOUT")
    LLM_MAX_RETRIES: Optional[int] = Field(default=3, env="LLM_MAX_RETRIES")

    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    AZURE_OPENAI_API_KEY: Optional[str] = Field(default=None, env="AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION: Optional[str] = Field(default="2024-02-15-preview", env="AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = Field(default=None, env="AZURE_OPENAI_DEPLOYMENT_NAME")
    
    GEMINI_API_KEY: Optional[str] = Field(default=None, env="GEMINI_API_KEY") 
    
    LOG_LEVEL: Optional[str] = Field(default="INFO", env="LOG_LEVEL")
    
    RABBITMQ_URL: Optional[str] = Field(env="RABBITMQ_URL")
    RABBITMQ_HEARTBEAT: Optional[int] = Field(default=1800, env="RABBITMQ_HEARTBEAT")
    RABBITMQ_CONNECTION_TIMEOUT: Optional[int] = Field(default=30000, env="RABBITMQ_CONNECTION_TIMEOUT")
    RABBITMQ_PREFETCH_COUNT: Optional[int] = Field(default=2, env="RABBITMQ_PREFETCH_COUNT")
    MAX_CONCURRENT_MEETINGS: Optional[int] = Field(default=2, env="MAX_CONCURRENT_MEETINGS")

    TASK_COMMANDS_QUEUE: Optional[str] = Field(default="task-management.task-commands", env="TASK_COMMANDS_QUEUE")

    # Transcription service configuration
    TRANSCRIPTION_QUEUE: Optional[str] = Field(default="transcription.queue", env="TRANSCRIPTION_QUEUE")
    TRANSCRIPTION_EXCHANGE: Optional[str] = Field(default="transcription.exchange", env="TRANSCRIPTION_EXCHANGE")

    # Temporary file storage configuration
    TEMP_AUDIO_DIR: Optional[str] = Field(default="/data", env="TEMP_AUDIO_DIR")
    TEMP_FILE_MAX_AGE_HOURS: Optional[int] = Field(default=24, env="TEMP_FILE_MAX_AGE_HOURS")
    MAX_AUDIO_FILE_SIZE_MB: Optional[int] = Field(default=1000, env="MAX_AUDIO_FILE_SIZE_MB")
    MIN_FREE_DISK_SPACE_GB: Optional[int] = Field(default=1, env="MIN_FREE_DISK_SPACE_GB")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
