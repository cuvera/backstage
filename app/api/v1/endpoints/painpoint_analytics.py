from fastapi import APIRouter, HTTPException

from app.schemas.painpoint_analytics import PainPointAnalyticsResponse
from app.services.painpoint_service import PainPointsService

router = APIRouter()

@router.get("/painpoint-analytics", response_model=PainPointAnalyticsResponse)
async def get_painpoint_analytics(period: int = 30):
    """
    Retrieve aggregated pain point analytics for a specified period.

    - **period**: Time period in days to look back (must be 7, 30, or 90).
    """
    if period not in [7, 30, 90]:
        raise HTTPException(status_code=400, detail="Period must be 7, 30, or 90")
    try:
        service = await PainPointsService.from_default()
        painpoints_data = await service.get_painpoint_analytics(period=period)
        return PainPointAnalyticsResponse(painpoints=painpoints_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
