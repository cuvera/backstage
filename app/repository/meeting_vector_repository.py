import logging
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from app.core.config import settings

logger = logging.getLogger(__name__)

class MeetingVectorRepository:
    """
    Repository for managing meeting transcript embeddings in Qdrant.
    Handles collection management and point upserts.
    """

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or settings.QDRANT_MEETING_COLLECTION_NAME
        self.client = QdrantClient(
            url=settings.QDRANT_MEETING_URL or settings.QDRANT_URL,
            api_key=settings.QDRANT_MEETING_API_KEY or settings.QDRANT_API_KEY
        )

    def ensure_collection(self, vector_size: int = 768, distance: str = "Cosine"):
        """
        Ensures the collection exists with the specified configuration.
        Default vector size 768 matches Gemini embedding-001 and text-embedding-004.
        """
        try:
            self.client.get_collection(self.collection_name)
            logger.debug(f"[Qdrant] Collection {self.collection_name} exists.")
            # Ensure index exists even if collection was already there
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="meetingId",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="tenantId",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except (UnexpectedResponse, Exception) as e:
            # Check if it's a 404
            if hasattr(e, 'status_code') and e.status_code == 404 or "Not Found" in str(e):
                logger.info(f"[Qdrant] Collection {self.collection_name} not found. Creating...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE if distance.lower() == "cosine" else models.Distance.EUCLID
                    )
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="meetingId",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="tenantId",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.info(f"[Qdrant] Created collection {self.collection_name}, meetingId index, and tenantId index")
            else:
                logger.error(f"[Qdrant] Error checking/creating collection: {e}")
                raise

    def upsert_points(self, points: List[models.PointStruct]):
        """
        Upserts points into the collection.
        """
        if not points:
            return
            
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            logger.info(f"[Qdrant] Upserted {len(points)} points to {self.collection_name}")
        except Exception as e:
            logger.error(f"[Qdrant] Upsert failed: {e}")
            raise

    def delete_by_meeting_id(self, meeting_id: str, tenant_id: str):
        """
        Deletes points matching a specific meetingId and tenantId.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="meetingId", match=models.MatchValue(value=meeting_id)),
                        models.FieldCondition(key="tenantId", match=models.MatchValue(value=tenant_id))
                    ]
                )
            )
        )
