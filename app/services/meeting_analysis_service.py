from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.schemas.meeting_analysis import MeetingAnalysis
from app.repository import MeetingAnalysisRepository

logger = logging.getLogger(__name__)


class MeetingAnalysisService:
    """
    Service layer for meeting analysis operations.
    Delegates data operations to MeetingAnalysisRepository.
    """

    def __init__(
        self,
        *,
        repository: Optional[MeetingAnalysisRepository] = None,
    ) -> None:
        self._repository = repository

    @classmethod
    async def from_default(cls) -> "MeetingAnalysisService":
        repository = await MeetingAnalysisRepository.from_default()
        return cls(repository=repository)

    async def save_analysis(self, analysis: MeetingAnalysis) -> Dict[str, Any]:
        """Save analysis using repository."""
        if not self._repository:
            raise RuntimeError("Repository not initialized")
        return await self._repository.save_analysis(analysis)

    async def get_analysis(self, *, tenant_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis using repository."""
        if not self._repository:
            raise RuntimeError("Repository not initialized")
        return await self._repository.get_analysis(tenant_id=tenant_id, session_id=session_id)

    async def get_analyses_by_session_ids(
        self, 
        *, 
        session_ids: List[str], 
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get multiple analyses using repository."""
        if not self._repository:
            raise RuntimeError("Repository not initialized")
        return await self._repository.get_analyses_by_session_ids(
            session_ids=session_ids, tenant_id=tenant_id
        )
