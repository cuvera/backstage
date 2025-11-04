from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import ASCENDING

from app.schemas.meeting_analysis import MeetingAnalysis

logger = logging.getLogger(__name__)


class CallAnalysisService:
    """
    Minimal persistence layer for call analyses.
    Stores one document per (tenant_id, session_id) in MongoDB.
    """

    COLLECTION = "meeting_analyses"

    def __init__(
        self,
        *,
        db: Optional[AsyncIOMotorDatabase] = None,
        collection_name: str = COLLECTION,
    ) -> None:
        self._db = db
        self._collection_name = collection_name
        self._collection: Optional[AsyncIOMotorCollection] = None

    @classmethod
    async def from_default(cls, collection_name: str = COLLECTION) -> "CallAnalysisService":
        from app.db.mongodb import get_database  # lazy import to avoid circular deps

        db = await get_database()
        service = cls(db=db, collection_name=collection_name)
        await service.ensure_indexes()
        return service

    async def ensure_indexes(self) -> None:
        collection = await self._ensure_collection()
        await collection.create_index(
            [("tenant_id", ASCENDING), ("session_id", ASCENDING)],
            unique=True,
            name="ux_tenant_session",
        )
        await collection.create_index(
            [("created_at", ASCENDING)],
            name="ix_created_at",
        )

    async def save_analysis(self, analysis: MeetingAnalysis) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        doc: Dict[str, Any] = analysis.dict(exclude_none=True)
        doc.setdefault("created_at", now)
        doc["updated_at"] = now

        collection = await self._ensure_collection()
        key = {"tenant_id": analysis.tenant_id, "session_id": analysis.session_id}

        logger.info(
            "[CallAnalysisService] Upserting analysis for tenant=%s session=%s",
            analysis.tenant_id,
            analysis.session_id,
        )

        result = await collection.update_one(key, {"$set": doc}, upsert=True)
        stored = await collection.find_one(key, {"_id": 1})

        mongo_id = stored.get("_id") if stored else result.upserted_id
        return {
            "document_id": str(mongo_id) if mongo_id else None,
            "matched": result.matched_count,
            "upserted": bool(result.upserted_id),
            "collection": self._collection_name,
        }

    async def get_analysis(self, *, tenant_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        collection = await self._ensure_collection()
        record = await collection.find_one(
            {"tenant_id": tenant_id, "session_id": session_id},
            {"_id": 0},
        )
        return record

    async def get_analyses_by_session_ids(
        self, 
        *, 
        session_ids: List[str], 
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get multiple analyses by session IDs.
        
        Args:
            session_ids: List of session identifiers to fetch analyses for
            tenant_id: Optional tenant filter for security
            
        Returns:
            List of analysis documents
        """
        if not session_ids:
            logger.warning("[CallAnalysisService] No session IDs provided for batch fetch")
            return []
        
        collection = await self._ensure_collection()
        
        try:
            # Build query
            query = {"session_id": {"$in": session_ids}}
            if tenant_id:
                query["tenant_id"] = tenant_id
            
            # Execute query
            cursor = collection.find(query, {"_id": 0}).sort("created_at", -1)
            docs = await cursor.to_list(length=len(session_ids) * 2)  # Allow for potential duplicates
            
            logger.info(
                "[CallAnalysisService] Fetched %d analyses for %d session IDs",
                len(docs),
                len(session_ids)
            )
            return docs
            
        except Exception as exc:
            logger.error(
                "[CallAnalysisService] Error fetching analyses for session_ids %s: %s",
                session_ids,
                exc
            )
            return []

    async def _ensure_collection(self) -> AsyncIOMotorCollection:
        if self._collection is None:
            if self._db is None:
                from app.db.mongodb import get_database

                self._db = await get_database()
            self._collection = self._db[self._collection_name]
        return self._collection
