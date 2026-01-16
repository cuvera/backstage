# Integration Guide: CallAnalysisCoordinator

This guide explains how to migrate the existing `MeetingAnalysisOrchestrator` from the monolithic `CallAnalysisAgent` to the new modular `CallAnalysisCoordinator`.

## Location
The new agent ensemble is located at:
`app/services/agents/call_analysis/`

## Step 1: Import the Coordinator
In `app/services/meeting_analysis_orchestrator.py`, replace the import:

```diff
-from app.services.agents import TranscriptionAgent, CallAnalysisAgent
+from app.services.agents import TranscriptionAgent
+from app.services.agents.call_analysis import CallAnalysisCoordinator
```

## Step 2: Initialize
Optionally initialize in the `__init__` or directly in the `analyze_meeting` method.

```python
# Inside MeetingAnalysisOrchestrator
self.analysis_coordinator = CallAnalysisCoordinator()
```

## Step 3: Replace the Agent Call
Replace the existing Step 2 block (around line 255) with the coordinator call:

```python
# REPLACE THIS:
# call_analysis_agent = CallAnalysisAgent()
# analysis = await call_analysis_agent.analyze(...)

# WITH THIS:
analysis_data = await self.analysis_coordinator.analyze_meeting(
    meeting_id=meeting_id,
    tenant_id=tenant_id,
    v2_transcript=transcription,  # The V2 payload
    metadata={
        "title": meeting_metadata.get("summary"),
        "platform": platform,
        "participants": attendees_list
    }
)
analysis = MeetingAnalysis(**analysis_data)
```

## Why this is better:
1.  **Parallel Execution**: Agents run concurrently (except Agenda inference), reducing total processing time.
2.  **Modular Logic**: Prompt changes for one section (e.g., Summary) won't break others (e.g., Action Items).
3.  **Structured Key Points**: The `KeyPointsAgent` now returns structured objects (Topic, Point, References) instead of flat strings, enabling more rich UI presentations.
4.  **Strategic Decisions**: Decisions are now strictly strategic and include time-based references for easy verification.
5.  **High-Fidelity Summary**: The new `SummaryAgent` provides a much more detailed and professional overview.
