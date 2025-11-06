#!/usr/bin/env python3
"""
Test script to directly invoke meeting prep pack generation.
Uses hardcoded meeting IDs to test the meeting prep service.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import from app
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from app.services.meeting_prep_curator_service import MeetingPrepCuratorService
# from app.db.mongodb import connect_to_mongo, close_mongo_connection
# from app.db.mongodb import get_database  # local import to avoid circular deps


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcoded test values
TEST_MEETING_ID = "69034a94bd25edb427558ce0"
TEST_RECURRING_MEETING_ID = "5gtgbtq4uu7u3nd9fe8c9r7ldl"
TEST_TENANT_ID = "689ddc0411e4209395942bee"

async def test_meeting_prep():
    """Test meeting prep pack generation using hardcoded meeting IDs."""
    
    try:
        logger.info("=== STARTING MEETING PREP TEST ===")
        logger.info(f"Meeting ID: {TEST_MEETING_ID}")
        logger.info(f"Recurring Meeting ID: {TEST_RECURRING_MEETING_ID}")
        logger.info(f"Tenant ID: {TEST_TENANT_ID}")
        
        # Connect to MongoDB first
        logger.info("Connecting to MongoDB...")
        db = await get_database()
        logger.info("✅ Connected to MongoDB successfully")
        
        # Initialize MeetingPrepCuratorService
        logger.info("Initializing MeetingPrepCuratorService...")
        prep_service = await MeetingPrepCuratorService.from_default()
        logger.info("✅ MeetingPrepCuratorService initialized successfully")
        
        # Generate and save prep pack
        logger.info("Generating prep pack...")
        result = await prep_service.generate_and_save_prep_pack(
            meeting_id=TEST_MEETING_ID,
            platform="google",
            recurring_meeting_id=TEST_RECURRING_MEETING_ID,
            previous_meeting_counts=5,  # Analyze last 5 meetings
            context={"test_mode": True, "source": "test_script"}
        )
        
        # Analyze results
        logger.info("\n=== PREP PACK GENERATION RESULTS ===")
        
        if result.get("status") == "skipped":
            logger.warning("⚠️  Prep pack generation was skipped")
            logger.warning(f"Reason: {result}")
            return result
        
        success = result.get('prep_pack') is not None
        logger.info(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        if result.get("prep_pack"):
            prep_pack = result["prep_pack"]
            logger.info("📋 PREP PACK DETAILS:")
            logger.info(f"  • Title: {prep_pack.get('title', 'N/A')}")
            logger.info(f"  • Tenant ID: {prep_pack.get('tenant_id', 'N/A')}")
            logger.info(f"  • Recurring Meeting ID: {prep_pack.get('recurring_meeting_id', 'N/A')}")
            logger.info(f"  • Purpose: {prep_pack.get('purpose', 'N/A')}")
            logger.info(f"  • Confidence: {prep_pack.get('confidence', 'N/A')}")
            logger.info(f"  • Key Points: {len(prep_pack.get('key_points', []))} items")
            logger.info(f"  • Expected Outcomes: {len(prep_pack.get('expected_outcomes', []))} items")
            logger.info(f"  • Blocking Items: {len(prep_pack.get('blocking_items', []))} items")
            logger.info(f"  • Decision Queue: {len(prep_pack.get('decision_queue', []))} items")
            logger.info(f"  • Open Questions: {len(prep_pack.get('open_questions', []))} items")
            logger.info(f"  • Risks/Issues: {len(prep_pack.get('risks_issues', []))} items")
            logger.info(f"  • Leadership Asks: {len(prep_pack.get('leadership_asks', []))} items")
            logger.info(f"  • Previous Meetings Ref: {len(prep_pack.get('previous_meetings_ref', []))} items")
            
            # Show sample content if available
            if prep_pack.get('key_points'):
                logger.info("  • Sample Key Points:")
                for i, point in enumerate(prep_pack['key_points'][:3], 1):
                    logger.info(f"    {i}. {point}")
            
            if prep_pack.get('expected_outcomes'):
                logger.info("  • Sample Expected Outcomes:")
                for i, outcome in enumerate(prep_pack['expected_outcomes'][:2], 1):
                    logger.info(f"    {i}. {outcome.get('description', 'N/A')} (Owner: {outcome.get('owner', 'N/A')})")
        
        if result.get("save_result"):
            save_result = result["save_result"]
            logger.info("\n💾 SAVE OPERATION:")
            logger.info(f"  • Collection: {save_result.get('collection', 'N/A')}")
            logger.info(f"  • Document ID: {save_result.get('document_id', 'N/A')}")
            logger.info(f"  • Operation: {'INSERT' if save_result.get('upserted', False) else 'UPDATE'}")
            logger.info(f"  • Matched Count: {save_result.get('matched', 0)}")
        
        # Test retrieval by recurring meeting ID
        logger.info("\n=== TESTING RETRIEVAL ===")
        try:
            retrieved = await prep_service.get_prep_pack_by_recurring_meeting_id(
                tenant_id=TEST_TENANT_ID,
                recurring_meeting_id=TEST_RECURRING_MEETING_ID
            )
            
            if retrieved:
                logger.info("✅ Successfully retrieved prep pack")
                logger.info(f"  • Retrieved Title: {retrieved.get('title', 'N/A')}")
                logger.info(f"  • Created At: {retrieved.get('created_at', 'N/A')}")
                logger.info(f"  • Updated At: {retrieved.get('updated_at', 'N/A')}")
                logger.info(f"  • Meeting ID: {retrieved.get('meeting_id', 'N/A')}")
            else:
                logger.warning("⚠️  Could not retrieve prep pack")
                
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
        
        # Test retrieval by meeting ID
        logger.info("\n=== TESTING RETRIEVAL BY MEETING ID ===")
        try:
            retrieved_by_meeting = await prep_service.get_prep_pack_by_meeting_id(
                tenant_id=TEST_TENANT_ID,
                meeting_id=TEST_MEETING_ID
            )
            
            if retrieved_by_meeting:
                logger.info("✅ Successfully retrieved prep pack by meeting ID")
                logger.info(f"  • Title: {retrieved_by_meeting.get('title', 'N/A')}")
            else:
                logger.warning("⚠️  Could not retrieve prep pack by meeting ID")
                
        except Exception as e:
            logger.error(f"❌ Retrieval by meeting ID failed: {e}")
        
        logger.info("\n=== TEST COMPLETED SUCCESSFULLY! ===")
        return result
        
    except Exception as e:
        logger.error(f"❌ Meeting prep test failed: {e}", exc_info=True)
        raise
    # finally:
        # Close MongoDB connection
        # try:
        #     await close_mongo_connection()
        #     logger.info("✅ MongoDB connection closed")
        # except Exception as e:
        #     logger.warning(f"⚠️  Error closing MongoDB connection: {e}")

async def test_multiple_scenarios():
    """Test multiple scenarios with the same meeting ID."""
    
    try:
        # Connect to MongoDB first
        # await connect_to_mongo()
        logger.info("✅ Connected to MongoDB for multiple scenarios test")
        
        scenarios = [
            {
                "name": "Standard Test",
                "previous_meeting_counts": 5,
                "context": {"test_mode": True}
            },
            {
                "name": "Limited Previous Meetings",
                "previous_meeting_counts": 2,
                "context": {"test_mode": True, "limited": True}
            },
            {
                "name": "Max Previous Meetings",
                "previous_meeting_counts": 10,
                "context": {"test_mode": True, "comprehensive": True}
            }
        ]
        
        prep_service = await MeetingPrepCuratorService.from_default()
        
        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"SCENARIO {i}: {scenario['name']}")
            logger.info(f"{'='*60}")
            
            try:
                result = await prep_service.generate_and_save_prep_pack(
                    meeting_id=TEST_MEETING_ID,
                    platform="google",
                    recurring_meeting_id=TEST_RECURRING_MEETING_ID,
                    previous_meeting_counts=scenario["previous_meeting_counts"],
                    context=scenario["context"]
                )
                
                success = result.get('prep_pack') is not None
                logger.info(f"✅ {scenario['name']}: {'SUCCESS' if success else 'NO PREP PACK'}")
                
                if success and result.get('prep_pack'):
                    prep_pack = result['prep_pack']
                    logger.info(f"  • Key Points: {len(prep_pack.get('key_points', []))}")
                    logger.info(f"  • Expected Outcomes: {len(prep_pack.get('expected_outcomes', []))}")
                    
            except Exception as e:
                logger.error(f"❌ {scenario['name']}: FAILED - {e}")
                
    except Exception as e:
        logger.error(f"❌ Multiple scenarios test failed: {e}", exc_info=True)
        raise
    # finally:
    #     try:
    #         await close_mongo_connection()
    #         logger.info("✅ MongoDB connection closed")
    #     except Exception as e:
    #         logger.warning(f"⚠️  Error closing MongoDB connection: {e}")

async def quick_test():
    """Quick minimal test - just the essentials."""
    
    try:
        logger.info("=== QUICK TEST ===")
        
        # Connect to MongoDB
        # await connect_to_mongo()
        
        prep_service = await MeetingPrepCuratorService.from_default()
        
        result = await prep_service.generate_and_save_prep_pack(
            meeting_id=TEST_MEETING_ID,
            platform="google",
            recurring_meeting_id=TEST_RECURRING_MEETING_ID
        )
        
        success = result.get('prep_pack') is not None
        print(f"\n🎯 QUICK RESULTS:")
        print(f"Success: {success}")
        
        if success:
            prep_pack = result.get('prep_pack', {})
            print(f"Title: {prep_pack.get('title', 'N/A')}")
            print(f"Key Points: {len(prep_pack.get('key_points', []))}")
            print(f"Saved: {result.get('save_result', {}).get('document_id', 'N/A')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}", exc_info=True)
        raise
    finally:
        try:
            await close_mongo_connection()
        except Exception as e:
            logger.warning(f"⚠️  Error closing MongoDB connection: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test: python test_meeting_prep.py quick
        asyncio.run(quick_test())
    elif len(sys.argv) > 1 and sys.argv[1] == "scenarios":
        # Multiple scenarios: python test_meeting_prep.py scenarios
        asyncio.run(test_multiple_scenarios())
    else:
        # Default comprehensive test: python test_meeting_prep.py
        asyncio.run(test_meeting_prep())