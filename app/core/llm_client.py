# =============================
# FILE: app/core/llm_client.py
# PURPOSE:
#   Enhanced LLM client with multi-provider fallback support.
#   Supports Gemini, OpenAI, and Azure OpenAI with automatic fallback
#   and retry logic.
#
# HOW TO USE:
#   from app.core.llm_client import llm_client
#
#   # Provider-level fallback (simplified)
#   response = await llm_client.chat_completion(
#       messages=[{"role": "user", "content": "Hello"}],
#       provider_chain=["gemini", "openai", "azure"],
#       temperature=0.3,
#       max_tokens=65000,
#       timeout=30
#   )
#
#   # Model-level fallback with per-provider settings (granular control)
#   response = await llm_client.chat_completion(
#       messages=[{"role": "user", "content": "Hello"}],
#       provider_chain=[
#           {"provider": "gemini", "model": "gemini-2.5-flash", "timeout": 30, "max_tokens": 4096},
#           {"provider": "openai", "model": "gpt-4o", "timeout": 60, "max_tokens": 8192},
#           {"provider": "gemini", "model": "gemini-3.0-pro", "timeout": 45},
#       ],
#       temperature=0.3
#   )
# =============================

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, AsyncAzureOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Enhanced LLM client with multi-provider fallback support.

    Supports:
    - Gemini (via OpenAI-compatible API)
    - OpenAI
    - Azure OpenAI

    Features:
    - Automatic fallback on provider failure
    - Retry logic with exponential backoff
    - Configurable provider chain (mixed format support)
    - Unified interface across providers

    Chain Formats (all supported in same chain):
    - Model names: "gemini-2.5-flash", "gpt-4o" (auto-detected)
    - Provider names: "gemini", "openai", "azure" (uses default models)
    - Explicit dicts: {"provider": "azure", "model": "my-deployment"}
    - Mixed: ["gemini-2.5-flash", "gpt-4o", {"provider": "azure", "model": "custom"}]
    """

    def __init__(self):
        self.gemini_client = None
        self.openai_client = None
        self.azure_client = None

        # Initialize clients based on available credentials
        self._initialize_clients()

        # Default provider chain from config or hardcoded fallback
        if settings.LLM_FALLBACK_CHAIN:
            self.default_chain = self._parse_fallback_chain(settings.LLM_FALLBACK_CHAIN)
        else:
            self.default_chain = self._default_provider_chain()

        # Timeout and retry settings
        self.timeout = settings.LLM_TIMEOUT
        self.max_retries = settings.LLM_MAX_RETRIES

    def _initialize_clients(self):
        """Initialize available LLM clients based on credentials."""

        # Gemini client (via OpenAI-compatible API)
        if settings.GEMINI_API_KEY:
            try:
                self.gemini_client = AsyncOpenAI(
                    api_key=settings.GEMINI_API_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    max_retries=0,
                    timeout=settings.LLM_TIMEOUT
                )
                logger.info("Gemini client initialized")
            except Exception as e:
                logger.warning(f"Failed to init Gemini client: {e}")

        # OpenAI client
        if settings.OPENAI_API_KEY:
            try:
                self.openai_client = AsyncOpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    max_retries=0,
                    timeout=settings.LLM_TIMEOUT
                )
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"Failed to init OpenAI client: {e}")

        # Azure OpenAI client
        if settings.AZURE_OPENAI_API_KEY and settings.AZURE_OPENAI_ENDPOINT:
            try:
                self.azure_client = AsyncAzureOpenAI(
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    max_retries=0,
                    timeout=settings.LLM_TIMEOUT
                )
                logger.info("Azure client initialized")
            except Exception as e:
                logger.warning(f"Failed to init Azure client: {e}")

    def _parse_fallback_chain(self, chain_str: str) -> List[Dict[str, Optional[str]]]:
        """
        Parse comma-separated fallback chain string and normalize to dict format.

        Args:
            chain_str: Comma-separated chain (e.g., "gemini-2.5-flash,gpt-4o,azure")

        Returns:
            List of normalized chain items as dicts
        """
        if not chain_str:
            return [{"provider": "gemini", "model": None}]

        items = [p.strip() for p in chain_str.split(",") if p.strip()]
        return [self._normalize_chain_item(item) for item in items]

    def _normalize_chain_item(self, item: str | Dict[str, str]) -> Dict[str, Optional[str | int | float]]:
        """
        Normalize chain item to standard dict format.

        Supports:
        - Model names: "gemini-2.5-flash" → {"provider": "gemini", "model": "gemini-2.5-flash"}
        - Provider names: "gemini" → {"provider": "gemini", "model": None}
        - Dict format: {"provider": "azure", "model": "custom", "timeout": 30, "max_tokens": 4096}

        Args:
            item: Chain item in any supported format

        Returns:
            Normalized dict with provider, model, timeout, and max_tokens
        """
        # Already in dict format
        if isinstance(item, dict):
            return {
                "provider": item.get("provider", "").lower(),
                "model": item.get("model"),
                "timeout": item.get("timeout"),
                "max_tokens": item.get("max_tokens"),
                "retries": item.get("retries")
            }

        # String format - auto-detect provider from model name
        if isinstance(item, str):
            item_lower = item.lower().strip()

            # Gemini model names
            if item_lower.startswith("gemini-"):
                return {"provider": "gemini", "model": item, "timeout": None, "max_tokens": None, "retries": None}

            # OpenAI model names
            elif item_lower.startswith("gpt-") or item_lower.startswith("o1-"):
                return {"provider": "openai", "model": item, "timeout": None, "max_tokens": None, "retries": None}

            # Anthropic model names (for future support)
            elif item_lower.startswith("claude-"):
                return {"provider": "anthropic", "model": item, "timeout": None, "max_tokens": None, "retries": None}

            # Provider name (uses default model)
            elif item_lower in ["gemini", "openai", "azure", "anthropic"]:
                return {"provider": item_lower, "model": None, "timeout": None, "max_tokens": None, "retries": None}

            # Unknown format - treat as provider name
            else:
                logger.warning(f"Unknown chain item '{item}', treating as provider")
                return {"provider": item_lower, "model": None, "timeout": None, "max_tokens": None, "retries": None}

        # Fallback
        logger.error(f"Invalid chain item type: {type(item)}")
        return {"provider": "gemini", "model": None, "timeout": None, "max_tokens": None, "retries": None}

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        provider_chain: Optional[List[str | Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """
        Call LLM with automatic fallback across providers.

        Args:
            messages: List of message dicts with 'role' and 'content'
            provider_chain: Ordered list of providers/models to try (supports mixed formats)
                - Model names: ["gemini-2.5-flash", "gpt-4o"]
                - Provider names: ["gemini", "openai", "azure"]
                - Dicts: [{"provider": "azure", "model": "custom", "timeout": 30, "max_tokens": 4096, "retries": 1}]
                - Mixed: ["gemini-2.5-flash", {"provider": "openai", "model": "gpt-4o-mini", "timeout": 60, "retries": 2}]
            model: Global model override (overrides chain-specified models)
            temperature: Sampling temperature
            max_tokens: Global max tokens override (overrides chain-specified max_tokens, defaults to 65000)
            timeout: Global timeout override (overrides chain-specified timeout, defaults to client timeout)
            response_format: Response format specification (e.g., {"type": "json_object"})
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response

        Raises:
            Exception: If all providers in chain fail
        """
        # Normalize chain to dict format
        if provider_chain:
            normalized_chain = [self._normalize_chain_item(item) for item in provider_chain]
        else:
            normalized_chain = self.default_chain

        chain_summary = [f"{c['provider']}:{c['model'] or 'default'}" for c in normalized_chain]
        logger.info(f"Chat completion | chain={chain_summary}")

        last_exception = None
        overall_start = time.time()

        for chain_item in normalized_chain:
            provider = chain_item["provider"]
            chain_model = chain_item["model"]
            chain_timeout = chain_item.get("timeout")
            chain_max_tokens = chain_item.get("max_tokens")
            chain_retries = chain_item.get("retries")

            # Use global overrides if provided, otherwise use chain values, then defaults
            effective_model = model or chain_model
            effective_timeout = timeout or chain_timeout or self.timeout
            effective_max_tokens = max_tokens or chain_max_tokens or 65000
            effective_retries = chain_retries or self.max_retries

            try:
                response = await self._try_provider(
                    provider=provider,
                    messages=messages,
                    model=effective_model,
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                    timeout=effective_timeout,
                    retries=effective_retries,
                    response_format=response_format,
                    **kwargs
                )

                duration_ms = round((time.time() - overall_start) * 1000, 2)
                logger.info(
                    f"Chat completion done | provider={provider} model={effective_model or 'default'} "
                    f"duration={duration_ms}ms"
                )
                return response

            except Exception as e:
                last_exception = e
                logger.warning(f"{provider}:{effective_model or 'default'} failed: {e}, falling back")
                continue

        # All providers failed
        duration_ms = round((time.time() - overall_start) * 1000, 2)
        logger.error(f"All providers failed after {duration_ms}ms | last_error={last_exception}")
        raise Exception("All providers in chain failed") from last_exception

    async def _try_provider(
        self,
        provider: str,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: float,
        retries: int,
        response_format: Optional[Dict],
        **kwargs
    ) -> str:
        """
        Try a single provider with retry logic.

        Args:
            provider: Provider name ("gemini", "openai", "azure")
            messages: Message list
            model: Model override
            temperature: Temperature
            max_tokens: Max tokens
            timeout: Request timeout in seconds
            retries: Number of retry attempts for this provider
            response_format: Response format
            **kwargs: Additional params

        Returns:
            Generated text

        Raises:
            Exception: If provider fails after retries
        """
        # Get client and model for provider
        client, model_name = self._get_provider_client_and_model(provider, model)

        if not client:
            raise Exception(f"Provider {provider} not configured or unavailable")

        # Retry with exponential backoff
        for attempt in range(retries):
            try:
                # Build request params
                request_params = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout,
                    **kwargs
                }

                # Add response_format if specified (not all providers support it)
                if response_format and provider in ["gemini", "openai"]:
                    request_params["response_format"] = response_format

                logger.info(
                    f"Calling {provider} | model={model_name} attempt={attempt + 1}/{retries} "
                    f"timeout={timeout}s max_tokens={max_tokens}"
                )

                call_start = time.time()
                response = await client.chat.completions.create(**request_params)
                call_ms = round((time.time() - call_start) * 1000, 2)

                logger.info(f"{provider}:{model_name} responded in {call_ms}ms")
                return response.choices[0].message.content

            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"{provider} attempt {attempt + 1} failed: {e}, retrying in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # Last attempt failed
                    raise

    def _get_provider_client_and_model(
        self,
        provider: str,
        model_override: Optional[str]
    ) -> tuple[Optional[AsyncOpenAI | AsyncAzureOpenAI], str]:
        """
        Get client and model name for a provider.

        Args:
            provider: Provider name
            model_override: Optional model override

        Returns:
            Tuple of (client, model_name)
        """
        if provider == "gemini":
            return (
                self.gemini_client,
                model_override or settings.DEFAULT_LLM_MODEL
            )

        elif provider == "openai":
            return (
                self.openai_client,
                model_override or "gpt-4o"
            )

        elif provider == "azure":
            return (
                self.azure_client,
                model_override or settings.AZURE_OPENAI_DEPLOYMENT_NAME or "gpt-4"
            )

        else:
            logger.warning(f"Unknown provider: {provider}")
            return (None, "")

    def _default_provider_chain(self):
        return [
          {"provider": "gemini", "model": "gemini-2.5-flash", "timeout": 120, "max_tokens": 24000, "retries": 1},
          {"provider": "gemini", "model": "gemini-3-flash-preview", "timeout": 180, "max_tokens": 48000, "retries": 1},
          {"provider": "gemini", "model": "gemini-2.5-pro", "retries": 2},
      ]

# Singleton instance
llm_client = LLMClient()
