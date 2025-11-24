from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.schemas.meeting_analysis import MeetingAnalysis, MeetingPrepPack
from app.services.agents.meeting_prep_agent import MeetingPrepAgent, MeetingPrepAgentError
from app.services.meeting_analysis_service import MeetingAnalysisService
from app.repository import MeetingPrepRepository, MeetingMetadataRepository
from app.utils.meeting_metadata import extract_recurring_meeting_id

logger = logging.getLogger(__name__)


class MeetingPrepCuratorServiceError(Exception):
    """Raised when the meeting prep curator service cannot complete its task."""


class MeetingPrepCuratorService:
    """
    Service layer for meeting preparation packs.
    Orchestrates MeetingPrepAgent and delegates data operations to repository.
    """

    def __init__(
        self,
        *,
        repository: Optional[MeetingPrepRepository] = None,
        analysis_service: Optional[MeetingAnalysisService] = None,
        metadata_repository: Optional[MeetingMetadataRepository] = None,
    ) -> None:
        self._repository = repository
        self._agent = MeetingPrepAgent()
        self._analysis_service = analysis_service
        self._metadata_repository = metadata_repository

    @classmethod
    async def from_default(cls) -> "MeetingPrepCuratorService":
        repository = await MeetingPrepRepository.from_default()
        analysis_service = await MeetingAnalysisService.from_default()
        metadata_repository = await MeetingMetadataRepository.from_default()
        
        service = cls(
            repository=repository, 
            analysis_service=analysis_service,
            metadata_repository=metadata_repository
        )
        return service

    async def generate_and_save_prep_pack(
        self,
        next_meeting_id: str,
        meeting_analysis: MeetingAnalysis,
        meeting_metadata: Dict[str, Any],
        platform: str,
        *,
        recurring_meeting_id: Optional[str] = None,
        previous_meeting_counts: Optional[int] = 2,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a meeting prep pack using the agent and save it to MongoDB.
        
        Args:
            meeting_id: ID of the upcoming meeting
            platform: Meeting platform (e.g., 'google', 'zoom', 'offline')
            recurring_meeting_id: Optional override for recurring meeting ID
            previous_meeting_counts: Number of previous meetings to analyze
            context: Additional context for processing
            
        Returns:
            Dictionary with save result and prep pack data
        """
        try:            
            # Fetch previous meetings and their analyses based on platform
            previous_meetings = await self._get_previous_meetings(
                recurring_meeting_id=recurring_meeting_id, 
                count = previous_meeting_counts, 
                platform=platform,
                to_date=meeting_metadata.get("start_time")
            )
            previous_analyses: List[MeetingAnalysis] = await self._get_meeting_analyses(previous_meetings)

            # Add current meeting to previous meetings and it's analysis
            previous_meetings.append(meeting_metadata)
            previous_analyses.append(meeting_analysis)

            # Generate prep pack using agent
            prep_pack = await self._agent.generate_prep_pack(
                meeting_metadata=meeting_metadata,
                previous_analyses=previous_analyses,
                recurring_meeting_id=recurring_meeting_id,
                previous_meetings=previous_meetings,
                context=context,
            )
            
            # Save to MongoDB using repository
            await self.save_prep_pack(prep_pack, next_meeting_id)
            
        except MeetingPrepAgentError as exc:
            logger.error("[MeetingPrepCuratorService] Agent error: %s", exc)
            raise MeetingPrepCuratorServiceError(f"Failed to generate prep pack: {exc}") from exc
        except Exception as exc:
            logger.exception("[MeetingPrepCuratorService] Unexpected error: %s", exc)
            raise MeetingPrepCuratorServiceError(f"Unexpected error: {exc}") from exc

    async def save_prep_pack(self, prep_pack: MeetingPrepPack, meeting_id: str) -> Dict[str, Any]:
        """Save a meeting prep pack using repository."""
        if not self._repository:
            raise RuntimeError("Repository not initialized")
        return await self._repository.save_prep_pack(prep_pack, meeting_id)

    async def get_prep_pack_by_meeting_id(
        self, *, tenant_id: str, meeting_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a prep pack by meeting ID using repository."""
        if not self._repository:
            raise RuntimeError("Repository not initialized")
        return await self._repository.get_prep_pack_by_meeting_id(
            tenant_id=tenant_id, meeting_id=meeting_id
        )

    async def get_prep_pack_by_recurring_meeting_id(
        self, *, tenant_id: str, recurring_meeting_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get the latest prep pack by recurring meeting ID using repository."""
        if not self._repository:
            raise RuntimeError("Repository not initialized")
        return await self._repository.get_prep_pack_by_recurring_meeting_id(
            tenant_id=tenant_id, recurring_meeting_id=recurring_meeting_id
        )

    async def get_prep_packs_by_recurring_meeting_id(
        self, *, tenant_id: str, recurring_meeting_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get multiple prep packs for a recurring meeting series using repository."""
        if not self._repository:
            raise RuntimeError("Repository not initialized")
        return await self._repository.get_prep_packs_by_recurring_meeting_id(
            tenant_id=tenant_id, recurring_meeting_id=recurring_meeting_id, limit=limit
        )

    async def delete_prep_pack(
        self, *, tenant_id: str, recurring_meeting_id: str
    ) -> Dict[str, Any]:
        """Delete a prep pack by tenant and recurring meeting ID using repository."""
        if not self._repository:
            raise RuntimeError("Repository not initialized")
        return await self._repository.delete_prep_pack(
            tenant_id=tenant_id, recurring_meeting_id=recurring_meeting_id
        )

    def _resolve_recurring_meeting_id(
        self, meeting_metadata: Optional[Dict[str, Any]], recurring_meeting_id: Optional[str] = None
    ) -> str:
        """Resolve recurring meeting ID from meeting details if not provided."""
        if recurring_meeting_id:
            return recurring_meeting_id
        
        # Handle case where meeting_metadata is None
        if meeting_metadata is None:
            logger.warning("Meeting metadata is None, cannot extract recurring_meeting_id")
            return None
        
        # Use utility function to extract recurring meeting ID
        resolved_id = extract_recurring_meeting_id(meeting_metadata)
        
        if not resolved_id:
            logger.warning("No recurring_meeting_id found in meeting metadata")
            return None

        return resolved_id

    async def _get_meeting_metadata(self, meeting_id: str, platform: str) -> Dict[str, Any]:
        """
        Fetch meeting metadata using repository.
        
        Args:
            meeting_id: Individual meeting identifier
            platform: Meeting platform (e.g., 'google', 'zoom', 'offline')
            
        Returns:
            Meeting metadata dictionary
        """
        # Only fetch from MongoDB for online platforms
        if platform != "offline" and platform == "google":
            logger.info("Fetching meeting metadata for platform=%s, meeting_id=%s", platform, meeting_id)
            
            if self._metadata_repository:
                meeting_data = await self._metadata_repository.get_meeting_metadata(meeting_id, platform)
                
                if meeting_data:
                    return meeting_data
                
                logger.warning("Failed to fetch meeting metadata, using placeholder data")
        
        # Return None for offline or when MongoDB query fails
        return None

    async def _get_previous_meetings(
        self, 
        recurring_meeting_id: str, 
        count: int, 
        platform: str = "google",
        from_date: Optional[str] = None, 
        to_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch previous meetings for the recurring meeting series from MongoDB.
        
        Args:
            recurring_meeting_id: Recurring meeting identifier
            count: Number of previous meetings to fetch
            platform: Meeting platform (e.g., 'google', 'zoom', 'offline')
            from_date: Start date filter (ISO format)
            to_date: End date filter (ISO format)
            
        Returns:
            List of previous meeting data
        """        
        if not self._metadata_repository:
            logger.warning("No metadata repository available for fetching previous meetings")
            return []

        meetings_data = await self._metadata_repository.get_meetings_by_recurring_id(
            recurring_meeting_id,
            platform=platform,
            limit=count,
            start=from_date,
            end=to_date,
        )
        
        if meetings_data and isinstance(meetings_data, list):
            logger.info("Fetched %d previous meetings", len(meetings_data))
            return meetings_data[:count]  # Ensure we don't exceed requested count
        
        logger.warning("Failed to fetch previous meetings or no meetings found")
        return []

    async def _get_meeting_analyses(self, meetings: List[Dict[str, Any]]) -> List[MeetingAnalysis]:
        """
        Fetch meeting analyses for the given meetings from MongoDB using MeetingAnalysisService.
        
        Args:
            meetings: List of meeting data with session_id fields
            
        Returns:
            List of MeetingAnalysis objects
        """
        if not meetings:
            logger.warning("[MeetingPrepCuratorService] No meetings provided for analysis fetch")
            return []
        
        # Extract session IDs and tenant_id from meetings
        session_ids = []
        tenant_id = None
        
        for meeting in meetings:
            session_id = meeting.get("id")
            if session_id:
                session_ids.append(session_id)
            
            # Extract tenant_id for security filtering (should be same for all meetings)
            if not tenant_id:
                tenant_id = meeting.get("tenant_id")
        
        if not session_ids:
            logger.warning("[MeetingPrepCuratorService] No valid session_ids found in meetings data")
            return []
        
        # Fetch analyses using MeetingAnalysisService
        try:
            analysis_docs = await self._analysis_service.get_analyses_by_session_ids(
                session_ids=session_ids,
                tenant_id=tenant_id
            )
            
            # Convert documents to MeetingAnalysis objects
            analyses = []
            for doc in analysis_docs:
                try:
                    # The document should already be in the correct format for MeetingAnalysis
                    analysis = MeetingAnalysis(**doc)
                    analyses.append(analysis)
                except Exception as e:
                    logger.warning(
                        "[MeetingPrepCuratorService] Failed to parse analysis for session %s: %s",
                        doc.get("session_id"),
                        e
                    )
                    continue
            
            logger.info(
                "[MeetingPrepCuratorService] Retrieved %d analyses for %d meetings",
                len(analyses),
                len(meetings)
            )
            return analyses
            
        except Exception as exc:
            logger.error(
                "[MeetingPrepCuratorService] Error fetching analyses: %s",
                exc
            )
            return []

    async def _find_immediate_next_meeting(
        self,
        current_meeting_metadata: Dict[str, Any],
        recurring_meeting_id: str,
        platform: str = "google"
    ) -> Optional[str]:
        """
        Find the immediate next scheduled meeting after the current meeting ends.
        Delegates to metadata repository method.
        """
        if not self._metadata_repository:
            logger.warning("[MeetingPrepCuratorService] No metadata repository available for finding next meeting")
            return None
        
        return await self._metadata_repository.find_immediate_next_meeting(
            current_meeting_metadata, recurring_meeting_id, platform
        )