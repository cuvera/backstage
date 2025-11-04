from pydantic import BaseModel
from typing import List, Optional

class PainPointAnalytics(BaseModel):
    title: Optional[str] = None
    department_name: Optional[str] = None
    category: Optional[str] = None
    severity: Optional[str] = None
    count: Optional[int] = None

class PainPointAnalyticsResponse(BaseModel):
    painpoints: List[PainPointAnalytics]
