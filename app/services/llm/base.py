from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from pydantic import BaseModel
from enum import Enum

class LLMProvider(str, Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str

class LLMResponse(BaseModel):
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None

class StreamResponse(BaseModel):
    delta: str
    is_complete: bool = False
    provider: str
    model: str

class BaseLLM(ABC):
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs

    @abstractmethod
    async def generate_response(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[StreamResponse, None]:
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        pass