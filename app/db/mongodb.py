from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class MongoDB:
    client: AsyncIOMotorClient = None
    database = None

db = MongoDB()

async def get_database():
    return db.database

async def connect_to_mongo():
    logger.info("Connecting to MongoDB...")
    db.client = AsyncIOMotorClient(settings.MONGODB_URL, tlsAllowInvalidCertificates=True)
    db.database = db.client[settings.DATABASE_NAME]
    logger.info("Connected to MongoDB")

async def close_mongo_connection():
    logger.info("Closing MongoDB connection...")
    db.client.close()
    logger.info("MongoDB connection closed")
