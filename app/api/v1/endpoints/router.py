from fastapi import APIRouter
from .health import router as health_router
from .meeting_prep import router as meeting_prep_router

api_router = APIRouter()
api_router.include_router(health_router, prefix="/health", tags=["health"])
api_router.include_router(meeting_prep_router, prefix="/meeting-prep", tags=["meeting-prep"])
