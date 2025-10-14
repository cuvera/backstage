# =============================
# FILE: app/messaging/connection.py
# PURPOSE:
#   Provide a shared RabbitMQ channel for consumers/producers.
#   Handles reconnects and QoS configuration in one place.
# =============================

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aio_pika
from aio_pika.abc import AbstractRobustChannel, AbstractRobustConnection

from app.core.config import settings

logger = logging.getLogger(__name__)

_connection_lock = asyncio.Lock()
_connection: Optional[AbstractRobustConnection] = None
_channel: Optional[AbstractRobustChannel] = None


async def get_channel() -> AbstractRobustChannel:
    """
    Return a shared robust channel to RabbitMQ.

    The channel is cached and re-created on demand if it gets closed.
    A single channel per process keeps consumer setup simple and avoids
    exhausting RabbitMQ connection limits.
    """
    global _connection, _channel

    async with _connection_lock:
        if _channel and not _channel.is_closed:
            return _channel

        if not _connection or _connection.is_closed:
            logger.info("Connecting to RabbitMQ at %s", settings.RABBITMQ_URL)
            _connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)

        logger.info("Opening RabbitMQ channel")
        _channel = await _connection.channel()
        await _channel.set_qos(prefetch_count=10)
        return _channel


async def close():
    """Close the shared channel/connection (mainly for graceful shutdowns)."""
    global _connection, _channel

    async with _connection_lock:
        if _channel and not _channel.is_closed:
            await _channel.close()
        if _connection and not _connection.is_closed:
            await _connection.close()

        _channel = None
        _connection = None
