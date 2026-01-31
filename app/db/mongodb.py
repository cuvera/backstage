from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Dict
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class MongoDB:
    """Multi-tenant MongoDB connection manager."""
    client: AsyncIOMotorClient = None
    database = None  # Legacy - kept for backward compatibility
    _tenant_databases: Dict[str, AsyncIOMotorDatabase] = {}

    def get_tenant_database(self, tenant_id: str) -> AsyncIOMotorDatabase:
        """
        Get or create database connection for a specific tenant.

        Uses lazy initialization and caching for efficient connection management.
        Each tenant gets a separate database named: {TENANT_DATABASE_PREFIX}_{tenant_id}

        Args:
            tenant_id: The tenant identifier

        Returns:
            AsyncIOMotorDatabase instance for the tenant

        Example:
            >>> db = MongoDB()
            >>> tenant_db = db.get_tenant_database("tenant_abc")
            >>> # Returns database named "backstage_tenant_abc"
        """
        if tenant_id not in self._tenant_databases:
            db_name = f"{settings.TENANT_DATABASE_PREFIX}_{tenant_id}"
            self._tenant_databases[tenant_id] = self.client[db_name]
            logger.info(f"Created database connection | tenant={tenant_id} db={db_name}")

        return self._tenant_databases[tenant_id]

    def clear_tenant_cache(self, tenant_id: str = None) -> None:
        """
        Clear cached tenant database connections.

        Args:
            tenant_id: Specific tenant to clear, or None to clear all

        Note:
            This is primarily for testing and administrative purposes.
            In production, connections are cached for the lifetime of the application.
        """
        if tenant_id:
            if tenant_id in self._tenant_databases:
                del self._tenant_databases[tenant_id]
                logger.info(f"Cleared database cache for tenant={tenant_id}")
        else:
            self._tenant_databases.clear()
            logger.info("Cleared all tenant database cache")

db = MongoDB()

async def get_database() -> AsyncIOMotorDatabase:
    """
    Get database for the current tenant context.

    This function reads the tenant ID from the current async context
    and returns the appropriate tenant-specific database.

    Returns:
        AsyncIOMotorDatabase instance for the current tenant

    Raises:
        TenantContextError: If tenant context is not set

    Example:
        >>> from app.core.tenant_context import set_tenant_context
        >>> set_tenant_context("tenant_abc")
        >>> db = await get_database()
        >>> # Returns database "backstage_tenant_abc"
    """
    from app.core.tenant_context import require_tenant_context

    tenant_id = require_tenant_context()
    return db.get_tenant_database(tenant_id)

async def get_database_for_tenant(tenant_id: str) -> AsyncIOMotorDatabase:
    """
    Get database for a specific tenant (bypasses context).

    Use this when you need to explicitly specify a tenant
    rather than using the current context.

    Args:
        tenant_id: The tenant identifier

    Returns:
        AsyncIOMotorDatabase instance for the specified tenant

    Example:
        >>> db = await get_database_for_tenant("tenant_xyz")
        >>> # Returns database "backstage_tenant_xyz"
    """
    return db.get_tenant_database(tenant_id)

async def connect_to_mongo():
    logger.info("Connecting to MongoDB...")
    db.client = AsyncIOMotorClient(settings.MONGODB_URL, tlsAllowInvalidCertificates=True)
    db.database = db.client[settings.DATABASE_NAME]
    logger.info("Connected to MongoDB")

async def close_mongo_connection():
    logger.info("Closing MongoDB connection...")
    db.client.close()
    logger.info("MongoDB connection closed")
