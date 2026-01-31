"""
Tenant middleware for extracting and validating tenant-id from HTTP headers.

This middleware:
1. Extracts tenant-id from request headers
2. Validates the tenant against the allowlist
3. Sets tenant context for the request lifecycle
4. Clears context after request completes
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from app.core.tenant_context import set_tenant_context, clear_tenant_context
from app.core.config import validate_tenant_id
import logging

logger = logging.getLogger(__name__)


class TenantMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract and validate tenant-id from HTTP headers.

    Extracts tenant ID from 'tenant-id' or 'x-tenant-id' headers,
    validates it, and sets it in the async context for the request.

    Paths in EXCLUDED_PATHS bypass tenant validation.
    """

    # Paths that don't require tenant-id header
    EXCLUDED_PATHS = [
        "/health",
        "/docs",
    ]

    async def dispatch(self, request: Request, call_next):
        """
        Process request and set tenant context.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler in the chain

        Returns:
            Response with x-tenant-id header added
        """
        # Skip validation for excluded paths (health checks, docs, etc.)
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        # Extract tenant-id from headers (support both formats)
        tenant_id = request.headers.get("tenant-id") or request.headers.get("x-tenant-id")

        if not tenant_id:
            logger.warning(
                f"Missing tenant-id header | "
                f"path={request.url.path} method={request.method} "
                f"client={request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Missing tenant-id header",
                    "message": "All API requests must include 'tenant-id' or 'x-tenant-id' header",
                    "detail": f"Request to {request.url.path} requires tenant identification"
                }
            )

        # Validate tenant ID against allowlist
        if not validate_tenant_id(tenant_id):
            logger.warning(
                f"Invalid tenant-id | "
                f"tenant_id={tenant_id} path={request.url.path} "
                f"client={request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Invalid tenant",
                    "message": f"Tenant '{tenant_id}' is not authorized",
                    "detail": "The provided tenant ID is not in the allowlist"
                }
            )

        # Set tenant context for the request lifecycle
        set_tenant_context(tenant_id)
        logger.debug(
            f"Tenant context set | "
            f"tenant_id={tenant_id} path={request.url.path} method={request.method}"
        )

        try:
            # Process the request
            response = await call_next(request)

            # Add tenant-id to response headers for debugging/tracking
            response.headers["x-tenant-id"] = tenant_id

            return response

        finally:
            # Always clear context after request completes
            # This prevents context leakage between requests
            clear_tenant_context()
