#!/usr/bin/env python3
"""
Simplified test script for meeting prep - bypasses SSL verification for testing.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcoded test values
TEST_MEETING_ID = "69034a94bd25edb427558ce0"
TEST_RECURRING_MEETING_ID = "5gtgbtq4uu7u3nd9fe8c9r7ldl"
TEST_TENANT_ID = "689ddc0411e4209395942bee"

async def test_meeting_prep_simple():
    """Simple test that creates minimal service without full database setup."""
    
    try:
        logger.info("=== STARTING SIMPLE MEETING PREP TEST ===")
        logger.info(f"Meeting ID: {TEST_MEETING_ID}")
        logger.info(f"Recurring Meeting ID: {TEST_RECURRING_MEETING_ID}")
        logger.info(f"Tenant ID: {TEST_TENANT_ID}")
        
        # Import here to avoid SSL issues during import
        from motor.motor_asyncio import AsyncIOMotorClient
        from app.services.meeting_prep_curator_service import MeetingPrepCuratorService
        from app.core.config import settings
        
        # Create MongoDB client with SSL verification disabled for testing
        logger.info("Connecting to MongoDB with SSL verification disabled...")
        
        # Modify MongoDB URL to disable SSL verification for testing
        mongodb_url = settings.MONGODB_URL
        
        # Connect to MongoDB using modified URL
        client = AsyncIOMotorClient(
            mongodb_url,
            serverSelectionTimeoutMS=5000  # 5 second timeout
        )
        
        # Get database
        db = client[settings.DATABASE_NAME]
        
        # Test connection
        await db.list_collection_names()
        logger.info("✅ Connected to MongoDB successfully")
        
        # Create MeetingPrepCuratorService manually without index creation
        logger.info("Creating MeetingPrepCuratorService...")
        prep_service = MeetingPrepCuratorService(db=db)
        
        # Initialize analysis service
        from app.services.meeting_analysis_service import MeetingAnalysisService
        prep_service._analysis_service = MeetingAnalysisService(db=db)
        
        logger.info("✅ MeetingPrepCuratorService created successfully")
        
        # Test the prep pack generation
        logger.info("Generating prep pack...")
        result = await prep_service.generate_and_save_prep_pack(
            meeting_id=TEST_MEETING_ID,
            platform="google",
            recurring_meeting_id=TEST_RECURRING_MEETING_ID,
            previous_meeting_counts=3,
            context={"test_mode": True, "ssl_disabled": True}
        )
        
        # Analyze results
        logger.info("\n=== PREP PACK GENERATION RESULTS ===")
        
        if result.get("status") == "skipped":
            logger.warning("⚠️  Prep pack generation was skipped")
            logger.warning(f"Reason: Platform not supported or no meeting data found")
            return result
        
        success = result.get('prep_pack') is not None
        logger.info(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        if result.get("prep_pack"):
            prep_pack = result["prep_pack"]
            logger.info("📋 PREP PACK DETAILS:")
            logger.info(f"  • Title: {prep_pack.get('title', 'N/A')}")
            logger.info(f"  • Tenant ID: {prep_pack.get('tenant_id', 'N/A')}")
            logger.info(f"  • Recurring Meeting ID: {prep_pack.get('recurring_meeting_id', 'N/A')}")
            logger.info(f"  • Confidence: {prep_pack.get('confidence', 'N/A')}")
            logger.info(f"  • Key Points: {len(prep_pack.get('key_points', []))} items")
            logger.info(f"  • Expected Outcomes: {len(prep_pack.get('expected_outcomes', []))} items")
            logger.info(f"  • Blocking Items: {len(prep_pack.get('blocking_items', []))} items")
            
            # Show some sample content
            if prep_pack.get('key_points'):
                logger.info("  • Sample Key Points:")
                for i, point in enumerate(prep_pack['key_points'][:2], 1):
                    logger.info(f"    {i}. {point}")
        else:
            logger.warning("No prep pack was generated")
            logger.info("This might be because:")
            logger.info("  - Meeting metadata not found in database")
            logger.info("  - No previous meetings found")
            logger.info("  - Platform not supported")
        
        if result.get("save_result"):
            save_result = result["save_result"]
            logger.info("\n💾 SAVE OPERATION:")
            logger.info(f"  • Collection: {save_result.get('collection', 'N/A')}")
            logger.info(f"  • Document ID: {save_result.get('document_id', 'N/A')}")
            logger.info(f"  • Operation: {'INSERT' if save_result.get('upserted', False) else 'UPDATE'}")
        
        logger.info("\n=== TEST COMPLETED! ===")
        return result
        
    except Exception as e:
        logger.error(f"❌ Simple meeting prep test failed: {e}", exc_info=True)
        
        # Show some diagnostic info
        logger.info("\n🔍 DIAGNOSTIC INFORMATION:")
        logger.info(f"  • Meeting ID: {TEST_MEETING_ID}")
        logger.info(f"  • Recurring ID: {TEST_RECURRING_MEETING_ID}")
        logger.info("  • Check if these IDs exist in your database")
        logger.info("  • Verify MongoDB connection string in settings")
        
        return {"error": str(e), "success": False}
    
    finally:
        # Close MongoDB connection
        try:
            if 'client' in locals():
                client.close()
                logger.info("✅ MongoDB connection closed")
        except Exception as e:
            logger.warning(f"⚠️  Error closing MongoDB connection: {e}")

async def test_database_connection():
    """Test just the database connection."""
    
    try:
        logger.info("=== TESTING DATABASE CONNECTION ===")
        
        from motor.motor_asyncio import AsyncIOMotorClient
        from app.core.config import settings
        
        logger.info("Attempting to connect to MongoDB...")
        
        # Modify MongoDB URL to disable SSL verification for testing
        mongodb_url = settings.MONGODB_URL
        
        # Add SSL verification bypass parameters to the connection string
        if "ssl=true" in mongodb_url.lower():
            # Replace ssl=true with ssl=false for testing
            mongodb_url = mongodb_url.replace("ssl=true", "ssl=false").replace("ssl=True", "ssl=false")
        elif "tls=true" in mongodb_url.lower():
            # Replace tls=true with tls=false for testing
            mongodb_url = mongodb_url.replace("tls=true", "tls=false").replace("tls=True", "tls=false")
        
        # If no SSL params in URL, add our own to disable SSL
        if "ssl=" not in mongodb_url.lower() and "tls=" not in mongodb_url.lower():
            separator = "&" if "?" in mongodb_url else "?"
            mongodb_url += f"{separator}ssl=false"
        
        logger.info("Modified MongoDB URL to disable SSL for testing")
        
        client = AsyncIOMotorClient(
            mongodb_url,
            serverSelectionTimeoutMS=5000
        )
        
        # Test connection
        db = client[settings.DATABASE_NAME]
        collections = await db.list_collection_names()
        
        logger.info("✅ Database connection successful!")
        logger.info(f"Database name: {settings.DATABASE_NAME}")
        logger.info(f"Collections found: {len(collections)}")
        logger.info(f"Sample collections: {collections[:5]}")
        
        # Check for our specific collections
        required_collections = ["google_meetings", "meeting_analyses", "meeting_preparations"]
        for collection in required_collections:
            if collection in collections:
                count = await db[collection].count_documents({})
                logger.info(f"  ✅ {collection}: {count} documents")
            else:
                logger.warning(f"  ⚠️  {collection}: Not found")
        
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "dbtest":
        # Test database connection only
        asyncio.run(test_database_connection())
    else:
        # Run the meeting prep test
        asyncio.run(test_meeting_prep_simple())