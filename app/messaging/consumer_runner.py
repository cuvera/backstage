# =============================
# FILE: app/messaging/consumer_runner.py
# PURPOSE:
#   Start RabbitMQ consumers (for all queues in QUEUE_CONFIG)
# =============================

import asyncio
import logging
from app.messaging.handlers.settings import QUEUE_CONFIG
from app.messaging.connection import get_channel
from app.messaging.utils import import_handler

logger = logging.getLogger(__name__)

async def main():
    channel = await get_channel()

    for queue_name, cfg in QUEUE_CONFIG.items():
        handler_fn = import_handler(cfg["handler"])
        await channel.consume(queue_name, handler_fn)
        logger.info(f"Consuming from queue: {queue_name} with handler {cfg['handler']}")

    logger.info("PainPoint consumer running...")
    await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
