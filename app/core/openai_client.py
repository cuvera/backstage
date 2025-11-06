from app.core.config import settings
import httpx
from openai import AsyncOpenAI

llm_client = AsyncOpenAI(
    api_key=settings.GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=120.0 # Since it's only for background processing which might be require long running tasks
)
httpx_client = httpx.AsyncClient(timeout=10.0)