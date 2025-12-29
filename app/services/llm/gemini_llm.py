from typing import List, Optional, AsyncGenerator, Dict, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from app.services.llm.base import BaseLLM, ChatMessage, LLMResponse, StreamResponse, LLMProvider
import asyncio
import logging

logger = logging.getLogger(__name__)

class GeminiLLM(BaseLLM):
    def __init__(
        self,
        model: str = "gemini-pro",
        api_key: str = None,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model)
        self.embedding_model = kwargs.get("embedding_model", "models/embedding-001")
        
        self.safety_settings = kwargs.get("safety_settings", {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        })

    def _convert_messages_to_gemini_format(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        gemini_messages = []
        
        for message in messages:
            if message.role == "system":
                continue
            
            role = "user" if message.role == "user" else "model"
            gemini_messages.append({
                "role": role,
                "parts": [{"text": message.content}]
            })
        
        return gemini_messages

    def _get_system_instruction(self, messages: List[ChatMessage]) -> Optional[str]:
        for message in messages:
            if message.role == "system":
                return message.content
        return None

    async def generate_response(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        try:
            system_instruction = self._get_system_instruction(messages)
            gemini_messages = self._convert_messages_to_gemini_format(messages)
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )
            
            if system_instruction:
                model_with_system = genai.GenerativeModel(
                    self.model,
                    system_instruction=system_instruction
                )
            else:
                model_with_system = self.model_instance

            if gemini_messages:
                chat = model_with_system.start_chat(history=gemini_messages[:-1])
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: chat.send_message(
                        gemini_messages[-1]["parts"][0]["text"],
                        generation_config=generation_config,
                        safety_settings=self.safety_settings
                    )
                )
            else:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model_with_system.generate_content(
                        "Hello",
                        generation_config=generation_config,
                        safety_settings=self.safety_settings
                    )
                )
            
            usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
            
            return LLMResponse(
                content=response.text,
                provider=self.get_provider_name(),
                model=self.model,
                usage=usage,
                metadata={
                    "finish_reason": response.candidates[0].finish_reason.name if response.candidates else None,
                    "safety_ratings": [
                        {
                            "category": rating.category.name,
                            "probability": rating.probability.name
                        }
                        for rating in response.candidates[0].safety_ratings
                    ] if response.candidates else []
                }
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[StreamResponse, None]:
        try:
            system_instruction = self._get_system_instruction(messages)
            gemini_messages = self._convert_messages_to_gemini_format(messages)
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )
            
            if system_instruction:
                model_with_system = genai.GenerativeModel(
                    self.model,
                    system_instruction=system_instruction
                )
            else:
                model_with_system = self.model_instance

            if gemini_messages:
                chat = model_with_system.start_chat(history=gemini_messages[:-1])
                
                response_stream = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: chat.send_message(
                        gemini_messages[-1]["parts"][0]["text"],
                        generation_config=generation_config,
                        safety_settings=self.safety_settings,
                        stream=True
                    )
                )
            else:
                response_stream = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model_with_system.generate_content(
                        "Hello",
                        generation_config=generation_config,
                        safety_settings=self.safety_settings,
                        stream=True
                    )
                )
            
            for chunk in response_stream:
                if chunk.text:
                    yield StreamResponse(
                        delta=chunk.text,
                        provider=self.get_provider_name(),
                        model=self.model
                    )
            
            yield StreamResponse(
                delta="",
                is_complete=True,
                provider=self.get_provider_name(),
                model=self.model
            )
                    
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="RETRIEVAL_DOCUMENT"
                )
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            raise

    def get_provider_name(self) -> str:
        return LLMProvider.GEMINI.value

    async def validate_connection(self) -> bool:
        try:
            test_messages = [ChatMessage(role="user", content="Hello")]
            response = await self.generate_response(test_messages, max_tokens=5)
            return bool(response.content)
        except Exception as e:
            logger.error(f"Gemini connection validation failed: {e}")
            return False
