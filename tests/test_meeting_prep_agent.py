import unittest
from unittest.mock import AsyncMock, MagicMock
import json
from datetime import datetime, timedelta
from app.services.agents.meeting_prep_agent import MeetingPrepAgent, MeetingPrepError
from app.schemas.meeting_analysis import MeetingAnalysis, MeetingPrepPack

class TestMeetingPrepAgent(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        self.mock_llm = MagicMock()
        self.agent = MeetingPrepAgent(llm=self.mock_llm)
        
        self.valid_metadata = {
            "title": "Weekly Sync",
            "scheduled_datetime": (datetime.now() + timedelta(days=1)).isoformat(),
            "attendees": ["Alice", "Bob"],
            "tenant_id": "tenant_123",
            "locale": "en-US",
            "timezone": "UTC"
        }
        
        self.valid_history = [
            MeetingAnalysis(
                tenant_id="tenant_123",
                session_id="s1",
                summary="Previous meeting summary.",
                key_points=["Point 1"],
                decisions=[],
                action_items=[],
                created_at=(datetime.now() - timedelta(days=7)).isoformat()
            )
        ]
        
        self.mock_llm_response = {
            "title": "Weekly Sync",
            "purpose": "Strategic alignment.",
            "expected_outcomes": [{"description": "Decision X", "owner": "Alice", "type": "decision"}],
            "blocking_items": [{"title": "Blocker Y", "owner": "Bob", "eta": "2023-01-01", "impact": "High", "severity": "high", "status": "open"}],
            "key_points": ["**Topic:** Status"],
            "open_questions": ["Question 1?"]
        }

    async def test_generate_prep_pack_success(self):
        # Setup mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(self.mock_llm_response)))]
        self.mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Run generation
        result = await self.agent.generate_prep_pack(
            meeting_metadata=self.valid_metadata,
            previous_analyses=self.valid_history,
            recurring_meeting_id="rec_123"
        )
        
        # Verify result
        self.assertIsInstance(result, MeetingPrepPack)
        self.assertEqual(result.title, "Weekly Sync")
        self.assertEqual(result.purpose, "Strategic alignment.")
        self.assertEqual(len(result.expected_outcomes), 1)
        self.assertEqual(result.expected_outcomes[0].type, "decision")
        self.assertEqual(len(result.blocking_items), 1)
        self.assertEqual(result.blocking_items[0].severity, "high")

    async def test_generate_prep_pack_no_history(self):
        # Setup mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(self.mock_llm_response)))]
        self.mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Run generation with empty history
        result = await self.agent.generate_prep_pack(
            meeting_metadata=self.valid_metadata,
            previous_analyses=[],
            recurring_meeting_id="rec_123"
        )
        
        self.assertIsInstance(result, MeetingPrepPack)

    async def test_generate_prep_pack_llm_failure(self):
        # Setup mock LLM to raise exception
        self.mock_llm.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        
        with self.assertRaises(MeetingPrepError) as cm:
            await self.agent.generate_prep_pack(
                meeting_metadata=self.valid_metadata,
                previous_analyses=self.valid_history,
                recurring_meeting_id="rec_123"
            )
        self.assertIn("llm call failed", str(cm.exception))

    async def test_generate_prep_pack_invalid_json(self):
        # Setup mock LLM to return invalid JSON
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Not JSON"))]
        self.mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)
        
        with self.assertRaises(MeetingPrepError) as cm:
            await self.agent.generate_prep_pack(
                meeting_metadata=self.valid_metadata,
                previous_analyses=self.valid_history,
                recurring_meeting_id="rec_123"
            )
        self.assertIn("llm response was not valid JSON", str(cm.exception))

if __name__ == '__main__':
    unittest.main()
