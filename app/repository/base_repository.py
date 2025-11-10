from typing import Optional
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase


class BaseRepository:
    """Base repository class with common MongoDB patterns."""

    def __init__(
        self,
        *,
        db: Optional[AsyncIOMotorDatabase] = None,
        collection_name: str,
    ) -> None:
        self._db = db
        self._collection_name = collection_name
        self._collection: Optional[AsyncIOMotorCollection] = None

    async def ensure_indexes(self) -> None:
        """Create MongoDB indexes for optimal query performance. Override in subclasses."""
        pass

    async def _ensure_collection(self) -> AsyncIOMotorCollection:
        """Ensure the MongoDB collection is available."""
        if self._collection is None:
            if self._db is None:
                from app.db.mongodb import get_database

                self._db = await get_database()
            self._collection = self._db[self._collection_name]
        return self._collection