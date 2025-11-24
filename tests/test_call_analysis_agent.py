import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import json
from app.services.agents.call_analysis_agent import CallAnalysisAgent, CallAnalysisAgentError
from app.schemas.meeting_analysis import MeetingAnalysis

class TestCallAnalysisAgent(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        self.mock_llm = MagicMock()
        self.agent = CallAnalysisAgent(llm=self.mock_llm)
        
        self.valid_payload = {
            "tenant_id": "tenant_123",
            "session_id": "session_456",
            "conversation": [
                {"speaker": "Alice", "text": "Hello", "start_time": 0, "end_time": 1, "duration": 1},
                {"speaker": "Bob", "text": "Hi there", "start_time": 2, "end_time": 3, "duration": 1}
            ],
            "sentiments": {
                "overall": "positive",
                "participant": [
                    {"name": "Alice", "sentiment": "positive"},
                    {"name": "Bob", "sentiment": "neutral"}
                ]
            }
        }
        
        self.mock_llm_response = {
            "summary": "A brief summary.",
            "key_points": ["Point 1", "Point 2"],
            "decisions": [{"title": "Decision 1", "owner": "Alice", "due_date": "2023-01-01", "references": [0]}],
            "action_items": [{"task": "Task 1", "owner": "Bob", "due_date": "2023-01-02", "priority": "High", "references": [1]}],
            "call_scoring": {
                "score": 8.5,
                "grade": "A",
                "reasons": [{"reason": "Good flow", "reference": {"start": "00:00", "end": "00:01"}}],
                "summary": "Great call."
            }
        }

    async def test_analyze_success(self):
        # Setup mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(self.mock_llm_response)))]
        self.mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Run analysis
        result = await self.agent.analyze(self.valid_payload)
        
        # Verify result
        self.assertIsInstance(result, MeetingAnalysis)
        self.assertEqual(result.tenant_id, "tenant_123")
        self.assertEqual(result.session_id, "session_456")
        self.assertEqual(result.summary, "A brief summary.")
        self.assertEqual(len(result.decisions), 1)
        self.assertEqual(result.decisions[0].title, "Decision 1")
        self.assertEqual(len(result.action_items), 1)
        self.assertEqual(result.action_items[0].task, "Task 1")
        self.assertEqual(result.call_scoring.score, 8.5)
        self.assertEqual(result.call_scoring.grade, "A")

    async def test_analyze_missing_ids(self):
        payload = self.valid_payload.copy()
        del payload["tenant_id"]
        del payload["session_id"]
        
        with self.assertRaises(CallAnalysisAgentError) as cm:
            await self.agent.analyze(payload)
        self.assertIn("tenant_id and session_id are required", str(cm.exception))

    async def test_analyze_empty_conversation(self):
        payload = self.valid_payload.copy()
        payload["conversation"] = []
        
        with self.assertRaises(CallAnalysisAgentError) as cm:
            await self.agent.analyze(payload)
        self.assertIn("conversation must contain at least one turn", str(cm.exception))

    async def test_analyze_llm_failure(self):
        # Setup mock LLM to raise exception
        self.mock_llm.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        
        with self.assertRaises(CallAnalysisAgentError) as cm:
            await self.agent.analyze(self.valid_payload)
        self.assertIn("llm call failed", str(cm.exception))

    async def test_analyze_invalid_json_response(self):
        # Setup mock LLM to return invalid JSON
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Not JSON"))]
        self.mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)
        
        with self.assertRaises(CallAnalysisAgentError) as cm:
            await self.agent.analyze(self.valid_payload)
        self.assertIn("llm response was not valid JSON", str(cm.exception))

if __name__ == '__main__':
    unittest.main()
