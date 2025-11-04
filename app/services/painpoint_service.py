from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from app.schemas.painpoint import PainPointCreate

logger = logging.getLogger(__name__)


class PainPointsService:
    """
    Handles CRUD and index setup for the 'pain_points' Mongo collection.

    This service mirrors the async patterns used across the codebase: collections are
    lazy-initialised and all DB operations are awaited to play nicely with Motor.
    """

    COLLECTION = "pain_points"

    def __init__(
        self,
        db: Optional[AsyncIOMotorDatabase] = None,
        collection_name: str = COLLECTION,
    ):
        self._db = db
        self._collection_name = collection_name
        self._collection: Optional[AsyncIOMotorCollection] = None

    # ----------------------------
    # Factory method for quick use
    # ----------------------------
    @classmethod
    async def from_default(cls, collection_name: str = COLLECTION) -> "PainPointsService":
        """
        Helper for convenience — imports the global DB connection and returns
        a ready-to-use service instance. Used directly by the handler.
        """
        from app.db.mongodb import get_database  # local import to avoid circular deps

        db = await get_database()
        service = cls(db=db, collection_name=collection_name)
        await service.ensure_indexes()
        return service

    async def _ensure_collection(self) -> AsyncIOMotorCollection:
        if self._collection is None:
            if self._db is None:
                from app.db.mongodb import get_database

                self._db = await get_database()
            self._collection = self._db[self._collection_name]
        return self._collection

    # ----------------------------
    # Index setup
    # ----------------------------
    async def ensure_indexes(self) -> None:
        """
        Creates indexes for performance and idempotency.
        - message_id: unique (idempotent upserts)
        - tenant_id + created_at: for efficient filtering
        - category + severity: for reporting queries
        """
        logger.info("[PainPointsService] Ensuring Mongo indexes for '%s'", self.COLLECTION)

        collection = await self._ensure_collection()
        await collection.create_index([("message_id", ASCENDING)], unique=True, name="ux_message_id")
        await collection.create_index(
            [("tenant_id", ASCENDING), ("created_at", DESCENDING)],
            name="ix_tenant_created",
        )
        await collection.create_index(
            [("enriched.category", ASCENDING), ("enriched.severity", ASCENDING)],
            name="ix_cat_sev",
        )

    # ----------------------------
    # Core method: Create Pain Point
    # ----------------------------
    async def create_painpoint(self, payload: PainPointCreate) -> Dict[str, Any]:
        """
        Insert a new Pain Point document into MongoDB.
        Uses message_id as a unique key for idempotency.

        Returns the inserted (or existing) document as a dict.
        """
        now = datetime.now(timezone.utc).isoformat()
        doc: Dict[str, Any] = payload.dict(by_alias=True)
        doc["created_at"] = now
        doc["updated_at"] = now

        collection = await self._ensure_collection()
        logger.info(
            "[PainPointsService] Inserting pain point | tenant=%s user=%s msg=%s category=%s severity=%s",
            doc.get("tenant_id"),
            doc.get("user_id"),
            doc.get("message_id"),
            doc.get("enriched", {}).get("category"),
            doc.get("enriched", {}).get("severity"),
        )

        try:
            res = await collection.insert_one(doc)
            logger.info(
                "[PainPointsService] Inserted _id=%s for message_id=%s",
                str(res.inserted_id),
                doc.get("message_id"),
            )
            doc["_id"] = str(res.inserted_id)
            return doc

        except DuplicateKeyError:
            # This means the same message_id was processed before
            logger.warning(
                "[PainPointsService] Duplicate message_id=%s (idempotent insert)",
                doc.get("message_id"),
            )

            existing = await collection.find_one({"message_id": doc.get("message_id")})
            if existing:
                existing["_id"] = str(existing["_id"])  # normalize for JSON
                logger.info(
                    "[PainPointsService] Returning existing record _id=%s for message_id=%s",
                    existing["_id"],
                    doc.get("message_id"),
                )
                return existing

            # Edge case: race between index creation and insert visibility
            logger.warning(
                "[PainPointsService] DuplicateKeyError but no existing doc found, retrying insert"
            )
            res = await collection.insert_one(doc)
            doc["_id"] = str(res.inserted_id)
            logger.info("[PainPointsService] Inserted after retry _id=%s", str(res.inserted_id))
            return doc

    async def get_painpoint_analytics(self, period: int) -> list[dict[str, Any]]:
        """
        Fetches and aggregates pain point analytics across all departments.
        """
        from datetime import datetime, timedelta, timezone

        collection = await self._ensure_collection()
        analytics_collection = collection.database["daily_dept_painpoint_top10"]

        start_date = datetime.now(timezone.utc) - timedelta(days=period)

        pipeline = [
            {
                "$match": {
                    "bucket_start": {"$gte": start_date}
                }
            },
            {"$unwind": "$top10"},
            {
                "$match": {
                    "top10.count": {"$gt": 0}  # include all valid pain points
                }
            },
            {
                "$group": {
                    "_id": {
                        "title": "$top10.title",
                        "category": "$top10.category",
                        "severity": "$top10.severity",
                        "department": "$meta.department"
                    },
                    "total_count": {"$sum": "$top10.count"}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "title": "$_id.title",
                    "department_name": "$_id.department",
                    "category": "$_id.category",
                    "severity": "$_id.severity",
                    "count": "$total_count"
                }
            },
            {"$sort": {"count": -1}}
        ]

        logger.info(f"[PainPointsService] Aggregating pain points across all departments for last {period} days")
        results = await analytics_collection.aggregate(pipeline).to_list(None)
        return results
