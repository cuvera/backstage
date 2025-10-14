from app.core.config import settings
from dotenv import load_dotenv

load_dotenv()

PAINPOINTS_ROUTING_KEY = settings.RABBITMQ_PAINPOINT_CAPTURED_QUEUE.strip()

QUEUE_CONFIG = {

      #PainPoint Agent queue
    PAINPOINTS_ROUTING_KEY: {
        "handler": "painpoint_handler"
    },
}
