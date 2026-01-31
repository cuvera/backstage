import logging
import sys
from app.core.config import settings


class TenantContextFilter(logging.Filter):
    """
    Logging filter that adds tenant_id to all log records.

    This filter automatically extracts the current tenant ID from
    the async context and adds it to every log record, making it
    easy to filter and debug tenant-specific issues.
    """

    def filter(self, record):
        """
        Add tenant_id attribute to log record.

        Args:
            record: LogRecord to modify

        Returns:
            True (always process the record)
        """
        from app.core.tenant_context import get_tenant_context

        tenant_id = get_tenant_context()
        record.tenant_id = tenant_id if tenant_id else "NO_TENANT"
        return True


def setup_logging():
    """
    Setup logging configuration with tenant context support.

    Configures:
    - Log level from settings
    - Log format with tenant_id field
    - Tenant context filter for all handlers
    - Suppresses noisy third-party loggers
    """
    # Create tenant filter
    tenant_filter = TenantContextFilter()

    # Configure basic logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(asctime)s - %(name)s - [tenant:%(tenant_id)s] - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Add tenant filter to all handlers
    for handler in logging.root.handlers:
        handler.addFilter(tenant_filter)

    # Suppress noisy loggers
    logging.getLogger("motor").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logging.getLogger(__name__)
