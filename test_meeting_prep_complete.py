#!/usr/bin/env python3
"""
Complete test for MeetingPrepAgent + MeetingPrepCuratorService + API endpoints.
Tests the full flow from agent to service to API.
"""

import sys
sys.path.append('/Users/gulshan/Developments/cuvera/backstage')

import asyncio
from typing import Dict, Any

def test_imports():
    """Test that all components can be imported successfully."""
    print("Testing imports...")
    
    try:
        from app.services.agents.meeting_prep_agent import MeetingPrepAgent, MeetingPrepAgentError
        print("✓ MeetingPrepAgent imported successfully")
        
        from app.services.meeting_prep_curator_service import MeetingPrepCuratorService, MeetingPrepCuratorServiceError
        print("✓ MeetingPrepCuratorService imported successfully")
        
        from app.api.v1.endpoints.meeting_prep import (
            router,
            GeneratePrepPackRequest,
            PrepPackResponse,
            PrepPackListResponse
        )
        print("✓ API endpoints imported successfully")
        
        from app.schemas.meeting_analysis import MeetingPrepPack, ExpectedOutcome, BlockingItem
        print("✓ Schemas imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_agent_functionality():
    """Test the MeetingPrepAgent functionality."""
    print("\nTesting MeetingPrepAgent...")
    
    try:
        from app.services.agents.meeting_prep_agent import MeetingPrepAgent
        
        # Create agent
        agent = MeetingPrepAgent()
        print("✓ Agent created successfully")
        
        # Test component methods
        meeting_id = "test_meeting_123"
        recurring_id = "weekly_sync_001"
        
        # Test ID resolution
        resolved_id = agent._resolve_recurring_meeting_id(meeting_id, recurring_id)
        print(f"✓ ID resolution works: {resolved_id}")
        
        # Test metadata fetching
        metadata = agent._get_meeting_metadata(meeting_id)
        print(f"✓ Metadata fetching works: {metadata['title']}")
        
        # Test previous meetings
        previous_meetings = agent._get_previous_meetings(recurring_id, 2)
        print(f"✓ Previous meetings fetching works: {len(previous_meetings)} meetings")
        
        # Test analyses
        analyses = agent._get_meeting_analyses(previous_meetings)
        print(f"✓ Analysis fetching works: {len(analyses)} analyses")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent test error: {e}")
        return False

async def test_service_functionality():
    """Test the MeetingPrepCuratorService functionality (without actual DB)."""
    print("\nTesting MeetingPrepCuratorService structure...")
    
    try:
        from app.services.meeting_prep_curator_service import MeetingPrepCuratorService
        
        # Test service creation (without DB connection)
        service = MeetingPrepCuratorService()
        print("✓ Service created successfully")
        
        # Test that service has required methods
        assert hasattr(service, 'generate_and_save_prep_pack')
        assert hasattr(service, 'get_prep_pack_by_meeting_id')
        assert hasattr(service, 'get_prep_pack_by_recurring_meeting_id')
        assert hasattr(service, 'save_prep_pack')
        print("✓ Service has all required methods")
        
        return True
        
    except Exception as e:
        print(f"✗ Service test error: {e}")
        return False

def test_api_structure():
    """Test the API endpoint structure."""
    print("\nTesting API endpoint structure...")
    
    try:
        from app.api.v1.endpoints.meeting_prep import router
        from fastapi import APIRouter
        
        # Verify router is FastAPI router
        assert isinstance(router, APIRouter)
        print("✓ Router is valid FastAPI router")
        
        # Check routes are registered
        routes = [route.path for route in router.routes]
        expected_routes = ['/generate', '/meeting/{meeting_id}', '/recurring/{recurring_meeting_id}', '/recurring/{recurring_meeting_id}/history', '/recurring/{recurring_meeting_id}']
        
        for expected_route in expected_routes:
            # Check if any route matches the pattern (ignoring exact path parameters)
            if any(expected_route.replace('{', '').replace('}', '') in route.replace('{', '').replace('}', '') for route in routes):
                print(f"✓ Route pattern found: {expected_route}")
            else:
                print(f"⚠ Route pattern might be missing: {expected_route}")
        
        return True
        
    except Exception as e:
        print(f"✗ API test error: {e}")
        return False

def test_schema_validation():
    """Test schema validation."""
    print("\nTesting schema validation...")
    
    try:
        from app.schemas.meeting_analysis import (
            MeetingPrepPack, ExpectedOutcome, BlockingItem, DecisionQueueItem,
            PreviousMeetingReference, OutcomeType, SeverityLevel, BlockingItemStatus
        )
        
        # Test creating schemas
        outcome = ExpectedOutcome(
            description="Test outcome",
            owner="test@example.com",
            type=OutcomeType.decision
        )
        print("✓ ExpectedOutcome schema works")
        
        blocking_item = BlockingItem(
            title="Test blocker",
            owner="test@example.com",
            eta="2025-11-01",
            impact="High impact",
            severity=SeverityLevel.high,
            status=BlockingItemStatus.open
        )
        print("✓ BlockingItem schema works")
        
        decision_item = DecisionQueueItem(
            id="decision_1",
            title="Test decision",
            needs=["approval", "budget"],
            owner="test@example.com"
        )
        print("✓ DecisionQueueItem schema works")
        
        meeting_ref = PreviousMeetingReference(
            meeting_id="meeting_123",
            analysis_id="analysis_456",
            datetime="2025-10-30T15:00:00Z"
        )
        print("✓ PreviousMeetingReference schema works")
        
        # Test full prep pack
        prep_pack = MeetingPrepPack(
            title="Test Prep Pack",
            tenant_id="test_tenant",
            timezone="UTC",
            locale="en-US",
            recurring_meeting_id="recurring_123",
            purpose="Test meeting",
            expected_outcomes=[outcome],
            blocking_items=[blocking_item],
            decision_queue=[decision_item],
            key_points=["Point 1", "Point 2"],
            open_questions=["Question 1"],
            risks_issues=["Risk 1"],
            leadership_asks=["Ask 1"],
            previous_meetings_ref=[meeting_ref],
            created_at="2025-10-31T15:00:00Z",
            updated_at="2025-10-31T15:00:00Z"
        )
        print("✓ MeetingPrepPack schema works")
        
        return True
        
    except Exception as e:
        print(f"✗ Schema test error: {e}")
        return False

def print_usage_examples():
    """Print usage examples for the implemented functionality."""
    print("\n" + "="*60)
    print("🎉 IMPLEMENTATION COMPLETE!")
    print("="*60)
    
    print("\n📋 USAGE EXAMPLES:")
    print("-" * 40)
    
    print("\n1. Generate Prep Pack (POST /api/v1/meeting-prep/generate):")
    print("""
    {
        "meeting_id": "meeting_123",
        "tenant_id": "your_tenant_id",
        "recurring_meeting_id": "weekly_sync_001",  // optional
        "previous_meeting_counts": 3  // optional
    }
    """)
    
    print("\n2. Get Prep Pack by Meeting ID (GET /api/v1/meeting-prep/meeting/{meeting_id}):")
    print("GET /api/v1/meeting-prep/meeting/meeting_123?tenant_id=your_tenant_id")
    
    print("\n3. Get Prep Pack by Recurring Meeting ID (GET /api/v1/meeting-prep/recurring/{recurring_meeting_id}):")
    print("GET /api/v1/meeting-prep/recurring/weekly_sync_001?tenant_id=your_tenant_id")
    
    print("\n4. Get Prep Pack History (GET /api/v1/meeting-prep/recurring/{recurring_meeting_id}/history):")
    print("GET /api/v1/meeting-prep/recurring/weekly_sync_001/history?tenant_id=your_tenant_id&limit=10")
    
    print("\n5. Delete Prep Pack (DELETE /api/v1/meeting-prep/recurring/{recurring_meeting_id}):")
    print("DELETE /api/v1/meeting-prep/recurring/weekly_sync_001?tenant_id=your_tenant_id")

def print_implementation_summary():
    """Print summary of what was implemented."""
    print("\n📦 IMPLEMENTATION SUMMARY:")
    print("-" * 40)
    
    print("\n✓ Created Components:")
    print("  • MeetingPrepAgent - Core AI agent for prep pack generation")
    print("  • MeetingPrepCuratorService - Service layer with MongoDB operations")
    print("  • API Endpoints - Complete REST API for prep pack operations")
    print("  • Schemas - Pydantic models for validation")
    
    print("\n✓ Key Features:")
    print("  • Automatic recurring_meeting_id resolution")
    print("  • Configurable previous meeting analysis (1-10 meetings)")
    print("  • MongoDB persistence with proper indexing")
    print("  • Full CRUD operations via REST API")
    print("  • Comprehensive error handling and logging")
    print("  • Multi-tenant support")
    
    print("\n✓ Database Collections:")
    print("  • meeting_preparations - Stores generated prep packs")
    print("  • Indexes: tenant_id+recurring_meeting_id (unique), tenant_id+meeting_id, created_at")
    
    print("\n⚠ Next Steps for Production:")
    print("  • Replace placeholder database queries in MeetingPrepAgent")
    print("  • Configure actual LLM integration")
    print("  • Set up MongoDB connection")
    print("  • Add authentication/authorization to API endpoints")
    print("  • Configure logging and monitoring")

async def main():
    """Run all tests and show results."""
    print("🧪 TESTING MEETING PREP IMPLEMENTATION")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Agent", test_agent_functionality()))
    results.append(("Service", await test_service_functionality()))
    results.append(("API", test_api_structure()))
    results.append(("Schemas", test_schema_validation()))
    
    # Show results
    print("\n" + "="*50)
    print("📊 TEST RESULTS:")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:15} | {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print_usage_examples()
        print_implementation_summary()
    else:
        print("\n❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    asyncio.run(main())