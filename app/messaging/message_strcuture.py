import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
from typing import Any, Dict
load_dotenv()

def generate_message(metadata: dict[str, str], payload: any) -> dict:
    return {
        "metadata": {
            "tenantId": metadata.get("tenantId"),
            "messageId": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "cognitive-service",
            "version": "v1.0",
            "sourceVersion": os.getenv("SERVICE_VERSION"),
            "sourceRegion": "us-east-1",
            "eventType": metadata.get("eventType"),
            "correlationId": metadata.get("correlationId") or generate_correlation_id(),
            "causationId": generate_correlation_id()
        },
        "payload": payload,
        "headers": {
            "content-type": "application/json",
             "schema-version": "1.0"
        }
    }


# Generate correlationId
def generate_correlation_id() -> str:
    return str(uuid.uuid4())
