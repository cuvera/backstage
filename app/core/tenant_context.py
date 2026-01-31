"""
Tenant context management using contextvars for multi-tenant support.

This module provides thread-safe and async-safe tenant context management
using Python's contextvars. The context automatically propagates through
async call chains without requiring explicit parameter passing.
"""

from contextvars import ContextVar
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Context variable for storing current tenant ID
# Each async task gets its own isolated copy
_tenant_context: ContextVar[Optional[str]] = ContextVar('tenant_id', default=None)


class TenantContextError(Exception):
    """Raised when tenant context is required but not set."""
    pass


class InvalidTenantError(Exception):
    """Raised when tenant ID is invalid or unauthorized."""
    pass


def set_tenant_context(tenant_id: str) -> None:
    """
    Set tenant ID in current async context.

    Args:
        tenant_id: The tenant identifier to set in context

    Example:
        >>> set_tenant_context("tenant_abc")
        >>> # All subsequent async calls will have access to this tenant_id
    """
    if not tenant_id:
        raise ValueError("tenant_id cannot be empty")

    _tenant_context.set(tenant_id)
    logger.debug(f"Tenant context set: {tenant_id}")


def get_tenant_context() -> Optional[str]:
    """
    Get tenant ID from current async context.

    Returns:
        The current tenant ID, or None if not set

    Example:
        >>> tenant_id = get_tenant_context()
        >>> if tenant_id:
        ...     print(f"Current tenant: {tenant_id}")
    """
    return _tenant_context.get()


def clear_tenant_context() -> None:
    """
    Clear tenant context.

    Should be called in finally blocks to prevent context leakage
    between requests/messages.

    Example:
        >>> try:
        ...     set_tenant_context("tenant_abc")
        ...     # Process request
        ... finally:
        ...     clear_tenant_context()
    """
    tenant_id = _tenant_context.get()
    if tenant_id:
        logger.debug(f"Clearing tenant context: {tenant_id}")
    _tenant_context.set(None)


def require_tenant_context() -> str:
    """
    Get tenant ID from context or raise if not set.

    Returns:
        The current tenant ID

    Raises:
        TenantContextError: If tenant context is not set

    Example:
        >>> tenant_id = require_tenant_context()  # Raises if not set
        >>> db = get_database_for_tenant(tenant_id)
    """
    tenant_id = _tenant_context.get()
    if not tenant_id:
        raise TenantContextError(
            "Tenant context is not set. Ensure tenant context is set before "
            "accessing tenant-scoped resources."
        )
    return tenant_id


def tenant_context_required(func):
    """
    Decorator to ensure tenant context is set before function execution.

    Args:
        func: The async function to wrap

    Raises:
        TenantContextError: If tenant context is not set when function is called

    Example:
        >>> @tenant_context_required
        ... async def process_data():
        ...     # This function requires tenant context to be set
        ...     pass
    """
    async def wrapper(*args, **kwargs):
        require_tenant_context()  # Raises if not set
        return await func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
