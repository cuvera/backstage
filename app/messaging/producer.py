import aio_pika
import asyncio
import json
import logging
from app.core.config import settings

RABBITMQ_URL = settings.RABBITMQ_URL


logger = logging.getLogger(__name__)

class RabbitMQProducer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RabbitMQProducer, cls).__new__(cls)
            cls._instance.connection = None
            cls._instance.channel = None
            cls._instance._lock = asyncio.Lock()
        return cls._instance

    async def connect(self):
        async with self._lock:
            if self.connection and not self.connection.is_closed:
                return  # already connected

            self.connection = await aio_pika.connect_robust(RABBITMQ_URL)
            self.channel = await self.connection.channel()
            logger.info("RabbitMQ Producer connected")

    async def send(self, queue_name: str, message: any):
        if not self.channel or self.channel.is_closed:
            raise RuntimeError("Producer is not connected. Call connect() first.")

        # Serialize message to JSON bytes if it's not already bytes
        if isinstance(message, (dict, list)):
            message_body = json.dumps(message, ensure_ascii=False).encode('utf-8')
        elif isinstance(message, str):
            message_body = message.encode('utf-8')
        else:
            message_body = message

        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=message_body,
                content_type='application/json'
            ),
            routing_key=queue_name
        )
        logger.info(f"Sent message to {queue_name}")

    async def publish_to_exchange(self, exchange_name: str, routing_key: str, message: any):
        """Publish message to a specific exchange with routing key."""
        if not self.channel or self.channel.is_closed:
            raise RuntimeError("Producer is not connected. Call connect() first.")

        # Serialize message to JSON bytes if it's not already bytes
        if isinstance(message, (dict, list)):
            message_body = json.dumps(message, ensure_ascii=False).encode('utf-8')
        elif isinstance(message, str):
            message_body = message.encode('utf-8')
        else:
            message_body = message

        # Get or declare exchange
        exchange = await self.channel.get_exchange(exchange_name)

        await exchange.publish(
            aio_pika.Message(
                body=message_body,
                content_type='application/json'
            ),
            routing_key=routing_key
        )
        logger.info(f"Published message to exchange={exchange_name} routing_key={routing_key}")

    async def close(self):
        async with self._lock:
            if self.channel and not self.channel.is_closed:
                await self.channel.close()
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
            logger.info("RabbitMQ Producer connection closed")

init_producer = None
def get_producer():
    global init_producer
    if init_producer is None:
        init_producer = RabbitMQProducer()    
    return init_producer

producer = get_producer()