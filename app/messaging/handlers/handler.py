from typing import Awaitable, Callable, Dict

from aio_pika.abc import AbstractIncomingMessage

from app.messaging.handlers.transcription_handler import transcription_handler

handler: Dict[str, Callable[[AbstractIncomingMessage], Awaitable[None]]] = {
    "transcription_handler": transcription_handler,
    # "painpoint_handler": painpoint_handler,
}
