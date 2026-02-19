import asyncio
import aio_pika
import logging
from typing import Callable
from app.core.config import settings
from app.messaging.handlers.settings import QUEUE_CONFIG
from app.messaging.handlers.handler import handler
from app.messaging.connection import get_channel, close

logger = logging.getLogger(__name__)

RABBITMQ_URL = settings.RABBITMQ_URL

class RabbitMQConsumerManager:
    def __init__(self):
        self._tasks = []

    async def _consume_queue(
        self,
        queue_name: str,
        handler: Callable,
        dlq: str = None,
        exchange_name: str = None,
        exchange_type: str = "topic",
        routing_keys: list = None,
        durable: bool = True
    ):
        try:
            logger.info(f"Setting up consumer for queue: {queue_name}")

            # Use shared connection and channel
            channel = await get_channel()

            # Set QoS for this consumer
            await channel.set_qos(prefetch_count=settings.RABBITMQ_PREFETCH_COUNT)

            # Declare exchange if specified
            exchange = None
            if exchange_name:
                logger.info(f"Declaring exchange: {exchange_name} (type={exchange_type})")
                exchange = await channel.declare_exchange(
                    exchange_name,
                    type=aio_pika.ExchangeType(exchange_type),
                    durable=durable
                )

            # Try to declare queue, if it fails due to precondition, use passive mode
            try:
                queue = await channel.declare_queue(queue_name, durable=durable)
            except Exception as e:
                if "PRECONDITION_FAILED" in str(e):
                    logger.warning(f"Queue {queue_name} exists with different config, using passive mode")
                    # Use passive mode to connect to existing queue without changing its config
                    queue = await channel.declare_queue(queue_name, passive=True)
                else:
                    raise

            # Bind queue to exchange with routing keys if specified
            if exchange and routing_keys:
                for routing_key in routing_keys:
                    logger.info(f"Binding queue {queue_name} to exchange {exchange_name} with routing key: {routing_key}")
                    await queue.bind(exchange, routing_key=routing_key)

            logger.info(f"Consuming from queue: {queue_name} | prefetch={settings.RABBITMQ_PREFETCH_COUNT}")
            await queue.consume(handler)

            # Keep consumer alive until cancelled
            await asyncio.Future()
        except asyncio.CancelledError:
            logger.info(f"Consumer for queue {queue_name} cancelled")
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
            exchange_name = config.get("exchange")
            exchange_type = config.get("exchange_type", "topic")
            routing_keys = config.get("routing_keys", [])
            durable = config.get("durable", True)

            logger.info(f"Setting up consumer for queue: {queue_name} with handler: {handler_name}")

            if not handler_func:
                logger.error(f"No handler for queue: {queue_name}, handler: {handler_name}")
                continue

            try:
                task = asyncio.create_task(
                    self._consume_queue(
                        queue_name=queue_name,
                        handler=handler_func,
                        dlq=dlq,
                        exchange_name=exchange_name,
                        exchange_type=exchange_type,
                        routing_keys=routing_keys,
                        durable=durable
                    )
                )
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

        # Close shared connection
        await close()

        logger.info("RabbitMQ consumers stopped.")
