from typing import List, Optional, AsyncGenerator
from openai import AsyncAzureOpenAI
from app.services.llm.base import BaseLLM, ChatMessage, LLMResponse, StreamResponse, LLMProvider
import logging

logger = logging.getLogger(__name__)

class AzureOpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str = "gpt-35-turbo",
        azure_endpoint: str = None,
        api_key: str = None,
        api_version: str = "2024-02-15-preview",
        deployment_name: str = None,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.deployment_name = deployment_name or model
        self.embedding_deployment = kwargs.get("embedding_deployment", "text-embedding-ada-002")
        
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )

    async def generate_response(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        try:
            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]
            
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=self.get_provider_name(),
                model=self.model,
                usage=usage,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "deployment_name": self.deployment_name
                }
            )
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[StreamResponse, None]:
        try:
            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]
            
            stream = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield StreamResponse(
                        delta=chunk.choices[0].delta.content,
                        provider=self.get_provider_name(),
                        model=self.model
                    )
                
                if chunk.choices[0].finish_reason:
                    yield StreamResponse(
                        delta="",
                        is_complete=True,
                        provider=self.get_provider_name(),
                        model=self.model
                    )
                    
        except Exception as e:
            logger.error(f"Azure OpenAI streaming error: {e}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_deployment,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Azure OpenAI embedding error: {e}")
            raise

    def get_provider_name(self) -> str:
        return LLMProvider.AZURE_OPENAI.value

    async def validate_connection(self) -> bool:
        try:
            test_messages = [ChatMessage(role="user", content="Hello")]
            response = await self.generate_response(test_messages, max_tokens=5)
            return bool(response.content)
        except Exception as e:
            logger.error(f"Azure OpenAI connection validation failed: {e}")
            return False