from app.core.config import settings
from dotenv import load_dotenv

load_dotenv()

# PAINPOINTS_ROUTING_KEY = settings.RABBITMQ_PAINPOINT_CAPTURED_QUEUE.strip()

QUEUE_CONFIG = {
    # Transcription service queue
    settings.TRANSCRIPTION_QUEUE: {
        "handler": "transcription_handler",
        "exchange": settings.TRANSCRIPTION_EXCHANGE,
        "exchange_type": "topic",
        "routing_keys": ["meeting.completed", "interview.completed"],
        "durable": True
    },
    # PainPoint Agent queue
    # PAINPOINTS_ROUTING_KEY: {
    #     "handler": "painpoint_handler"
    # }
}
