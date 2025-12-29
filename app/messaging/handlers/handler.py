from typing import Awaitable, Callable, Dict

from aio_pika.abc import AbstractIncomingMessage

from app.messaging.handlers.painpoint_handler import painpoint_handler
from app.messaging.handlers.meeting_handler import meeting_handler
from app.messaging.handlers.meeting_retry_handler import meeting_retry_handler
from app.messaging.handlers.meeting_embedding_handler import meeting_embedding_handler

handler: Dict[str, Callable[[AbstractIncomingMessage], Awaitable[None]]] = {
    "painpoint_handler": painpoint_handler,
    "meeting_handler": meeting_handler,
    "meeting_retry_handler": meeting_retry_handler,
    "meeting_embedding_handler": meeting_embedding_handler,
}
