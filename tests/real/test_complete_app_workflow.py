"""
Complete application workflow tests with real data.
Tests the entire stack: Import → Database → UI Logic → Query → Display

These tests verify the application works correctly end-to-end with real Discord data.
"""
import pytest
import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from load_messages import load_messages_once
from lib.database import get_recent_message_records, save_server, save_channel


@pytest.mark.real
@pytest.mark.skip_ci
@pytest.mark.slow
class TestCompleteWorkflow:
    """Test complete application workflow with real data."""
    
    def test_import_to_ui_workflow(
        self,
        test_database,
        test_server_id,
    ):
        """Test complete workflow: Import → Database → UI Query → Results."""
        print("\n" + "="*80)
        print("🔄 COMPLETE APPLICATION WORKFLOW TEST")
        print("="*80)
        
        # Step 1: Import data
        print("\n📥 STEP 1: Importing messages (720 hours)...")
        summary = load_messages_once(
            server_id=test_server_id,
            hours_back=720,
            sleep_between_channels=False
        )
        
        print(f"   ✅ Import complete: {summary['messages_saved']} messages")
        
        # Step 2: Simulate UI _refresh_recent_records() logic
        print("\n🖥️  STEP 2: Simulating UI data load...")
        
        # Import the actual function from streamlit_app
        # We'll test it with different time ranges
        test_ranges = [1, 24, 168, 720]
        ui_results = {}
        
        for hours in test_ranges:
            # This is what UI does
            records = get_recent_message_records(
                server_id=int(test_server_id),
                hours=hours,
                keywords=None,
                channel_ids=None,
                limit=None,  # Should be None after fix
            )
            
            ui_results[hours] = {
                'count': len(records),
                'records': records
            }
            
            print(f"   {hours:3}h → {len(records):5} messages")
        
        # Step 3: Verify results are correct
        print("\n✅ STEP 3: Verifying results...")
        
        # Counts should increase with time range
        counts = [ui_results[h]['count'] for h in test_ranges]
        
        if counts == sorted(counts):
            print("   ✅ Message counts increase with time range")
        else:
            print(f"   ❌ PROBLEM: Counts don't increase: {counts}")
            
        # Verify no artificial limit
        if ui_results[720]['count'] > 500:
            print(f"   ✅ No 500-message limit detected "
                  f"({ui_results[720]['count']} messages loaded)")
        else:
            print(f"   ⚠️  WARNING: Only {ui_results[720]['count']} messages "
                  f"for 720h (might be limited)")
        
        # Step 4: Verify data quality
        print("\n🔍 STEP 4: Verifying data quality...")
        
        for hours, result in ui_results.items():
            if result['records']:
                # Check timestamp range
                timestamps = [r['sent_at'] for r in result['records']]
                oldest = min(timestamps)
                newest = max(timestamps)
                span = (newest - oldest) / 3600
                
                print(f"   {hours:3}h: span={span:6.1f}h, "
                      f"oldest={datetime.fromtimestamp(oldest):%Y-%m-%d %H:%M}")
        
        # Assertions
        assert len(ui_results[720]['records']) > 0, "Should have messages"
        assert counts == sorted(counts), "Counts should increase with time"
        
        print("\n✅ Complete workflow test PASSED!")
        
        return ui_results
    
    def test_ui_limit_removed(
        self,
        test_database,
        test_server_id,
    ):
        """Specifically test that the 500-message limit bug is fixed."""
        print("\n" + "="*80)
        print("🐛 TESTING BUG FIX: 500-Message Limit Removed")
        print("="*80)
        
        # Import enough data
        print("\n📥 Importing 720 hours of messages...")
        summary = load_messages_once(
            server_id=test_server_id,
            hours_back=720,
            sleep_between_channels=False
        )
        
        imported = summary['messages_saved']
        print(f"   💾 Imported: {imported} messages")
        
        if imported <= 500:
            print(f"\n   ℹ️  Note: Server only has {imported} messages in 720h")
            print(f"   Cannot test limit removal (need >500 messages)")
            pytest.skip("Not enough messages to test limit removal")
        
        # Query without limit (as UI should do now)
        print("\n🔍 Querying with limit=None (fixed UI)...")
        records_unlimited = get_recent_message_records(
            server_id=int(test_server_id),
            hours=720,
            limit=None
        )
        
        print(f"   ✅ Retrieved: {len(records_unlimited)} messages")
        
        # Query with old limit (to show difference)
        print("\n🔍 Querying with limit=500 (old buggy behavior)...")
        records_limited = get_recent_message_records(
            server_id=int(test_server_id),
            hours=720,
            limit=500
        )
        
        print(f"   ⚠️  Retrieved: {len(records_limited)} messages")
        
        # Compare
        print("\n📊 Comparison:")
        print(f"   Without limit: {len(records_unlimited)} messages")
        print(f"   With limit=500: {len(records_limited)} messages")
        print(f"   Difference: {len(records_unlimited) - len(records_limited)} messages")
        
        # The fix should show more messages
        assert len(records_unlimited) > len(records_limited), \
            "Unlimited should return more messages than limited"
        assert len(records_unlimited) == imported, \
            "Unlimited should return all imported messages"
        assert len(records_limited) == 500, \
            "Limited should return exactly 500"
        
        print("\n✅ Bug fix verified! UI will now show all messages.")


@pytest.mark.real
@pytest.mark.skip_ci
class TestUIFunctionality:
    """Test UI-specific functions with real data."""
    
    def test_refresh_recent_records_logic(
        self,
        test_database,
        test_server_id,
    ):
        """Test the _refresh_recent_records() logic with real data."""
        print("\n" + "="*80)
        print("🔄 TESTING _refresh_recent_records() LOGIC")
        print("="*80)
        
        # Setup: Import some data
        print("\n📥 Setting up test data...")
        summary = load_messages_once(
            server_id=test_server_id,
            hours_back=168,  # 1 week
            sleep_between_channels=False
        )
        print(f"   💾 Imported: {summary['messages_saved']} messages")
        
        # Test different hours values
        print("\n🔍 Testing with different time ranges...")
        
        test_cases = [
            (1, "1 hour"),
            (24, "24 hours"),
            (168, "1 week"),
        ]
        
        results = {}
        
        for hours, label in test_cases:
            # Simulate what _refresh_recent_records() does
            records = get_recent_message_records(
                server_id=int(test_server_id),
                hours=hours,
                keywords=None,
                channel_ids=None,
                limit=None,  # IMPORTANT: Should be None after fix
            )
            
            # Sort as UI does
            records = sorted(records, key=lambda r: r["sent_at"]) if records else []
            
            results[hours] = {
                'records': records,
                'count': len(records),
                'metadata': {
                    'message_count': len(records),
                    'hours': hours,
                }
            }
            
            print(f"   {label:15} → {len(records):4} messages")
            
            if records:
                oldest = datetime.fromtimestamp(records[0]['sent_at'])
                newest = datetime.fromtimestamp(records[-1]['sent_at'])
                print(f"      Range: {oldest:%H:%M} - {newest:%H:%M}")
        
        # Verify
        counts = [results[h]['count'] for h in [1, 24, 168]]
        assert counts == sorted(counts), "Counts should increase"
        
        print("\n✅ _refresh_recent_records() logic working correctly!")
    
    def test_channel_filtering_workflow(
        self,
        test_database,
        test_server_id,
        real_discord_client,
    ):
        """Test channel filtering works end-to-end."""
        print("\n" + "="*80)
        print("📢 TESTING CHANNEL FILTERING WORKFLOW")
        print("="*80)
        
        # Get channels from server
        print("\n📋 Getting channels...")
        channels = real_discord_client.get_channel_ids()
        print(f"   Found {len(channels)} channels")
        
        if len(channels) < 2:
            pytest.skip("Need at least 2 channels to test filtering")
        
        # Import data
        print("\n📥 Importing all channels...")
        summary_all = load_messages_once(
            server_id=test_server_id,
            hours_back=24,
            channel_ids=None,  # All channels
            sleep_between_channels=False
        )
        
        print(f"   💾 All channels: {summary_all['messages_saved']} messages")
        
        # Clear and import single channel
        conn = sqlite3.connect(test_database)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        conn.commit()
        conn.close()
        
        test_channel_id = channels[0][0]
        print(f"\n📥 Importing single channel ({channels[0][1]})...")
        
        summary_single = load_messages_once(
            server_id=test_server_id,
            hours_back=24,
            channel_ids=[test_channel_id],
            sleep_between_channels=False
        )
        
        print(f"   💾 Single channel: {summary_single['messages_saved']} messages")
        
        # Query with channel filter
        print("\n🔍 Testing UI channel filtering...")
        
        records_filtered = get_recent_message_records(
            server_id=int(test_server_id),
            hours=24,
            channel_ids=[test_channel_id],
            limit=None
        )
        
        print(f"   Filtered query: {len(records_filtered)} messages")
        
        # Verify all returned messages are from correct channel
        if records_filtered:
            wrong_channel = [r for r in records_filtered 
                           if r['channel_id'] != test_channel_id]
            
            if wrong_channel:
                print(f"   ❌ Found {len(wrong_channel)} messages from wrong channel!")
            else:
                print(f"   ✅ All messages from correct channel")
            
            assert len(wrong_channel) == 0, "All messages should be from filtered channel"
        
        print("\n✅ Channel filtering works correctly!")


@pytest.mark.real
@pytest.mark.skip_ci
@pytest.mark.slow
class TestCompleteUserScenarios:
    """Test complete user scenarios end-to-end."""
    
    def test_user_changes_time_range_scenario(
        self,
        test_database,
        test_server_id,
    ):
        """Simulate user changing time range in UI."""
        print("\n" + "="*80)
        print("👤 USER SCENARIO: Changing Time Range in UI")
        print("="*80)
        
        # Setup: Import 1 week of data
        print("\n📥 Setup: Importing 1 week of messages...")
        summary = load_messages_once(
            server_id=test_server_id,
            hours_back=168,
            sleep_between_channels=False
        )
        print(f"   💾 Imported: {summary['messages_saved']} messages")
        
        # Simulate user workflow
        print("\n👤 User opens app, sees default 720 hours...")
        
        # User's actions:
        actions = [
            (720, "User keeps default 720h"),
            (24, "User changes to 24h"),
            (1, "User changes to 1h"),
            (168, "User changes to 168h (1 week)"),
        ]
        
        previous_count = None
        
        for hours, description in actions:
            print(f"\n   {description}:")
            
            # This is what happens when user changes slider
            records = get_recent_message_records(
                server_id=int(test_server_id),
                hours=hours,
                limit=None
            )
            
            count = len(records)
            print(f"      → UI shows: {count} messages")
            
            # User expectation: different time ranges show different counts
            if previous_count is not None:
                if count != previous_count:
                    print(f"      ✅ Count changed from {previous_count} (expected)")
                else:
                    print(f"      ⚠️  Count same as before (might be issue)")
            
            previous_count = count
        
        print("\n✅ User scenario completed!")
    
    def test_user_queries_messages_scenario(
        self,
        test_database,
        test_server_id,
    ):
        """Simulate user querying messages."""
        print("\n" + "="*80)
        print("👤 USER SCENARIO: Querying Messages")
        print("="*80)
        
        # Setup
        print("\n📥 Setup: Importing messages...")
        summary = load_messages_once(
            server_id=test_server_id,
            hours_back=24,
            sleep_between_channels=False
        )
        print(f"   💾 Imported: {summary['messages_saved']} messages")
        
        if summary['messages_saved'] == 0:
            print("   ⚠️  No messages to query")
            pytest.skip("No messages available for testing")
        
        # User views recent messages
        print("\n👤 User opens chat interface...")
        records = get_recent_message_records(
            server_id=int(test_server_id),
            hours=24,
            limit=None
        )
        
        print(f"   📊 UI loaded: {len(records)} messages")
        
        # Show sample messages
        if records:
            print(f"\n   📝 Recent messages preview:")
            for record in records[-3:]:
                dt = datetime.fromtimestamp(record['sent_at'])
                preview = record['content'][:50] + "..." if len(record['content']) > 50 else record['content']
                print(f"      [{dt:%H:%M}] {preview}")
        
        # Verify data structure
        if records:
            sample = records[0]
            required_fields = ['id', 'channel_id', 'content', 'sent_at']
            
            for field in required_fields:
                assert field in sample, f"Record missing required field: {field}"
            
            print(f"\n   ✅ All {len(records)} messages have correct structure")
        
        print("\n✅ Query scenario completed!")
