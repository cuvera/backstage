from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import logging

class Settings(BaseSettings):
    SERVICE_NAME: str = Field(default="cognitive-service", env="SERVICE_NAME")
    PROJECT_NAME: str = Field(default="Cognitive Service", env="PROJECT_NAME")
    VERSION: str = Field(default="1.0.0", env="VERSION")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    MONGODB_URL: str = Field(default="mongodb://localhost:27017", env="MONGODB_URL")
    DATABASE_NAME: str = Field(default="intelligence_db", env="DATABASE_NAME")
    
    QDRANT_HOST: str = Field(default="localhost", env="QDRANT_HOST")
    QDRANT_PORT: int = Field(default=6333, env="QDRANT_PORT")
    QDRANT_API_KEY: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = Field(default="voiceprints-v1.1", env="QDRANT_COLLECTION_NAME")
    VECTOR_DIMENSION: int = Field(default=256, env="VECTOR_DIMENSION")
    
    DEFAULT_LLM_PROVIDER: str = Field(default="gemini", env="DEFAULT_LLM_PROVIDER")
    DEFAULT_LLM_MODEL: str = Field(default="gemini-2.5-flash", env="DEFAULT_LLM_MODEL")
    
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    WHISPER_MODEL_SIZE: str = Field(default="tiny", env="WHISPER_MODEL_SIZE")
    WHISPER_DEVICE: str = Field(default="cpu", env="WHISPER_DEVICE")
    WHISPER_COMPUTE_TYPE: str = Field(default="int8", env="WHISPER_COMPUTE_TYPE")
    
    AZURE_OPENAI_API_KEY: Optional[str] = Field(default=None, env="AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION: str = Field(default="2024-02-15-preview", env="AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = Field(default=None, env="AZURE_OPENAI_DEPLOYMENT_NAME")
    
    GEMINI_API_KEY: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = Field(default="us-east-1", env="AWS_REGION")
    S3_BUCKET_NAME: str = Field(default="cognitive-service-storage", env="AWS_BUCKET_NAME")
    AWS_ENDPOINT: Optional[str] = Field(default=None, env="AWS_ENDPOINT")
    
    RABBITMQ_URL: str = Field(env="RABBITMQ_URL")
    RABBITMQ_HEARTBEAT: int = Field(default=60, env="RABBITMQ_HEARTBEAT")
    RABBITMQ_CONNECTION_TIMEOUT: int = Field(default=30000, env="RABBITMQ_CONNECTION_TIMEOUT")
    
    RABBITMQ_VOICEPRINT_REGISTERED_QUEUE: str = Field(env="RABBITMQ_VOICEPRINT_REGISTERED_QUEUE")
    RABBITMQ_MEETING_INSIGHTS_GENERATION_QUEUE: str = Field(env="RABBITMQ_MEETING_INSIGHTS_GENERATION_QUEUE")
    RABBITMQ_MEETING_INSIGHTS_GENERATED_QUEUE: str = Field(env= "RABBITMQ_MEETING_INSIGHTS_GENERATED_QUEUE")

    ENABLE_TRANSCRIPTION_CONSUMER: bool = Field(default=True, env="ENABLE_TRANSCRIPTION_CONSUMER")

    QDRANT_COLLECTION_NAME_RAG: str = Field(env="QDRANT_COLLECTION_NAME_RAG")
    QDRANT_API_KEY_RAG: str = Field(env="QDRANT_API_KEY_RAG")
    QDRANT_URL_RAG: str = Field(env="QDRANT_URL_RAG")

    CHATBOT_NAME: str = Field(env="CHATBOT_NAME")
    CUVERA_CORE_SERVICE: str =Field(env="CUVERA_CORE_SERVICE")
    
    AUTH_SERVICE_URL: str = Field(env="AUTH_SERVICE_URL")

    RABBITMQ_PAINPOINT_CAPTURED_QUEUE: str = Field(env="painpoint.captured.q",)

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
