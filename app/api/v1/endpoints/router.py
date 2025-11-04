from fastapi import APIRouter
from .health import router as health_router
from .painpoint_analytics import router as painpoint_analytics_router

api_router = APIRouter()
api_router.include_router(health_router, prefix="/health", tags=["health"])
api_router.include_router(painpoint_analytics_router, tags=["Painpoint Analytics"])
