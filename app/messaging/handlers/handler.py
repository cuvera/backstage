from typing import Awaitable, Callable, Dict

from aio_pika.abc import AbstractIncomingMessage

handler: Dict[str, Callable[[AbstractIncomingMessage], Awaitable[None]]] = {
    # "painpoint_handler": painpoint_handler,
}
