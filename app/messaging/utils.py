# =============================
# FILE: app/messaging/utils.py
# PURPOSE:
#   Helper utilities for dynamically resolving message handlers.
# =============================

from __future__ import annotations

import importlib
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def import_handler(handler_name: str) -> Callable:
    """
    Resolve a handler function by name.

    Handlers are primarily registered in `app.messaging.handlers.handler`.
    If a handler is not in the registry, fallback to importing it directly
    from `app.messaging.handlers.<handler_name>`.
    """
    from app.messaging.handlers.handler import handler as handler_registry  # lazy import

    func = handler_registry.get(handler_name)
    if func:
        return func

    module_path, attr = _deduce_module_and_attr(handler_name)

    try:
        module = importlib.import_module(module_path)
        func = getattr(module, attr)
        logger.debug("Imported handler %s from %s", attr, module_path)
        return func
    except (ModuleNotFoundError, AttributeError) as exc:
        raise ImportError(f"Handler '{handler_name}' not found") from exc


def _deduce_module_and_attr(handler_name: str) -> tuple[str, str]:
    """
    Determine import path (module, attribute) for a handler name.

    Accepts both dotted paths (e.g. "module.handler") and short names.
    Short names map to modules under `app.messaging.handlers`.
    """
    if "." in handler_name:
        module_path, attr = handler_name.rsplit(".", maxsplit=1)
        return module_path, attr

    module_path = f"app.messaging.handlers.{handler_name}"
    attr = handler_name
    return module_path, attr
