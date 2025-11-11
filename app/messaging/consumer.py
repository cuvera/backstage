import asyncio
import aio_pika
import logging
from typing import Callable
from app.core.config import settings
from app.messaging.handlers.settings import QUEUE_CONFIG
from app.messaging.handlers.handler import handler

logger = logging.getLogger(__name__)

RABBITMQ_URL = settings.RABBITMQ_URL

class RabbitMQConsumerManager:
    def __init__(self):
        self._tasks = []
        self._connections = []

    async def _consume_queue(self, queue_name: str, handler: Callable, dlq: str = None):
        connection = None
        try:
            logger.info(f"Connecting to RabbitMQ for queue: {queue_name}")
            connection = await aio_pika.connect_robust(
                RABBITMQ_URL,
                timeout=60,
                heartbeat=60,  # Send heartbeat every 60 seconds
                blocked_connection_timeout=300,  # 5 minutes for blocked connections
                client_properties={
                    "connection_name": f"backstage-consumer-{queue_name}"
                }
            )
            self._connections.append(connection)

            channel = await connection.channel()
            # Reduce prefetch to avoid holding too many messages during long processing
            await channel.set_qos(prefetch_count=1)

            # Try to declare queue, if it fails due to precondition, use passive mode
            try:
                queue = await channel.declare_queue(queue_name, durable=True)
            except Exception as e:
                if "PRECONDITION_FAILED" in str(e):
                    logger.warning(f"Queue {queue_name} exists with different config, using passive mode")
                    # Use passive mode to connect to existing queue without changing its config
                    queue = await channel.declare_queue(queue_name, passive=True)
                else:
                    raise

            logger.info(f"Successfully connected and consuming from queue: {queue_name}")
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    logger.debug(f"Received message on queue: {queue_name}")
                    await handler(message)
        except asyncio.CancelledError:
            logger.info(f"Consumer for queue {queue_name} cancelled (normal shutdown)")
            raise
        except Exception as e:
            if "Channel closed by RPC timeout" in str(e) or "ChannelInvalidStateError" in str(e):
                logger.info(f"Consumer for queue {queue_name} disconnected during shutdown")
            else:
                logger.error(f"Error in consumer for queue {queue_name}: {e}", exc_info=True)
            raise

    async def start(self):
        logger.info("Starting RabbitMQ consumers...")
        logger.info(f"Queue config: {QUEUE_CONFIG}")
        logger.info(f"Available handlers: {list(handler.keys())}")
        
        for queue_name, config in QUEUE_CONFIG.items():
            handler_name = config["handler"]
            handler_func = handler.get(handler_name)
            dlq = config.get("dlq")

            logger.info(f"Setting up consumer for queue: {queue_name} with handler: {handler_name}")

            if not handler_func:
                logger.error(f"No handler for queue: {queue_name}, handler: {handler_name}")
                continue

            try:
                task = asyncio.create_task(self._consume_queue(queue_name, handler_func, dlq))
                self._tasks.append(task)
                logger.info(f"Successfully created consumer task for queue: {queue_name}")
            except Exception as e:
                logger.error(f"Failed to create consumer task for queue {queue_name}: {e}")
                continue
            
        logger.info(f"Started {len(self._tasks)} consumer tasks")

    async def stop(self):
        logger.info("Shutting down RabbitMQ consumers...")
        
        # Cancel all consumer tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete with timeout
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Consumer tasks did not shut down within timeout")
        
        # Close connections
        for conn in self._connections:
            try:
                if not conn.is_closed:
                    await conn.close()
            except Exception as e:
                logger.debug(f"Error closing connection during shutdown: {e}")
        
        logger.info("RabbitMQ consumers stopped.")
