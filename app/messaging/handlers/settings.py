from app.core.config import settings
from dotenv import load_dotenv

load_dotenv()

PAINPOINTS_ROUTING_KEY = settings.RABBITMQ_PAINPOINT_CAPTURED_QUEUE.strip()
MEETING_PROCESSING_ROUTING_KEY = settings.RABBITMQ_MEETING_PROCESSING_QUEUE.strip()
MEETING_RETRY_ROUTING_KEY = settings.RABBITMQ_BACKSTAGE_RETRY_QUEUE.strip()

QUEUE_CONFIG = {

      #PainPoint Agent queue
    PAINPOINTS_ROUTING_KEY: {
        "handler": "painpoint_handler"
    },
    
    # Meeting Processing queue
    MEETING_PROCESSING_ROUTING_KEY: {
        "handler": "meeting_handler"
    },
    
    # Meeting Retry queue
    MEETING_RETRY_ROUTING_KEY: {
        "handler": "meeting_retry_handler"
    },
}
