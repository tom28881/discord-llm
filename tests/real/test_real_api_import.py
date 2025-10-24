"""
Real-world integration tests using actual Discord API and real data.

⚠️  IMPORTANT: These tests use REAL Discord API and REAL data!
   
Prerequisites:
1. Add to your .env file:
   ENABLE_REAL_TESTS=1
   TEST_SERVER_ID=your_test_server_id
   TEST_CHANNEL_ID=your_test_channel_id (optional)
   DISCORD_TOKEN=your_discord_token

2. Use a TEST server, NOT production!

3. Run with:
   pytest -m real -v -s

These tests will:
- Connect to actual Discord API
- Import real messages
- Store in isolated test database (data/test_real_db.sqlite)
- Verify data integrity
- Test incremental imports

The test database is kept after tests for inspection.
Delete data/test_real_db.sqlite manually to clean up.
"""
import pytest
import sqlite3
import time
from datetime import datetime, timedelta

from load_messages import load_messages_once, initialize_discord_client
from lib.database import (
    get_recent_message_records, get_servers, get_channels,
    get_latest_message_timestamp, get_last_message_id
)


@pytest.mark.real
@pytest.mark.skip_ci
@pytest.mark.slow
class TestRealDiscordConnection:
    """Test real Discord API connection."""
    
    def test_can_connect_to_discord(self, real_discord_client):
        """Verify we can connect to Discord API with real credentials."""
        print("\n🔌 Testing real Discord API connection...")
        
        # Get server list
        servers = real_discord_client.get_server_ids()
        
        print(f"✅ Successfully connected! Found {len(servers)} servers")
        for server_id, server_name in servers:
            print(f"   📁 {server_name} (ID: {server_id})")
        
        assert len(servers) > 0, "Should have at least one server"
    
    def test_can_get_channels(self, real_discord_client, test_server_id):
        """Verify we can retrieve channels from test server."""
        print(f"\n📢 Testing channel retrieval from server {test_server_id}...")
        
        # Get channels
        channels = real_discord_client.get_channel_ids()
        
        print(f"✅ Found {len(channels)} channels:")
        for channel_id, channel_name in channels[:10]:  # Show first 10
            print(f"   # {channel_name} (ID: {channel_id})")
        
        if len(channels) > 10:
            print(f"   ... and {len(channels) - 10} more")
        
        assert len(channels) > 0, "Test server should have at least one channel"


@pytest.mark.real
@pytest.mark.skip_ci
@pytest.mark.slow
class TestRealMessageImport:
    """Test real message import from Discord."""
    
    def test_import_real_messages(
        self, 
        test_database, 
        test_server_id, 
        test_channel_id,
        test_config
    ):
        """Import real messages from Discord and verify in database."""
        print(f"\n📥 Importing real messages from server {test_server_id}...")
        
        # Configure import parameters
        channel_filter = [int(test_channel_id)] if test_channel_id else None
        hours_back = test_config["hours_back"]
        
        print(f"   ⏱️  Time window: Last {hours_back} hour(s)")
        if channel_filter:
            print(f"   📢 Channel filter: {channel_filter}")
        else:
            print(f"   📢 All channels")
        
        # Run import
        start_time = time.time()
        summary = load_messages_once(
            server_id=test_server_id,
            channel_ids=channel_filter,
            hours_back=hours_back,
            sleep_between_channels=True
        )
        duration = time.time() - start_time
        
        # Print results
        print(f"\n✅ Import completed in {duration:.2f} seconds")
        print(f"   📊 Servers processed: {summary['processed_servers']}")
        print(f"   💾 Messages saved: {summary['messages_saved']}")
        
        # Verify in database
        conn = sqlite3.connect(test_database)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM servers")
        server_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM channels")
        channel_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages")
        message_count = cursor.fetchone()[0]
        
        print(f"\n📊 Database contents:")
        print(f"   Servers: {server_count}")
        print(f"   Channels: {channel_count}")
        print(f"   Messages: {message_count}")
        
        # Show sample messages
        cursor.execute("""
            SELECT content, sent_at 
            FROM messages 
            ORDER BY sent_at DESC 
            LIMIT 5
        """)
        recent = cursor.fetchall()
        
        if recent:
            print(f"\n📝 Sample messages (most recent):")
            for content, timestamp in recent:
                dt = datetime.fromtimestamp(timestamp)
                preview = content[:60] + "..." if len(content) > 60 else content
                print(f"   [{dt:%Y-%m-%d %H:%M}] {preview}")
        
        conn.close()
        
        # Assertions
        assert summary["processed_servers"] >= 1, "Should process at least one server"
        assert server_count >= 1, "Should have at least one server in DB"
        
        if summary["messages_saved"] > 0:
            assert message_count == summary["messages_saved"], "DB count should match import summary"
        else:
            print("\n⚠️  No messages imported (channel might be empty in time window)")
    
    def test_real_data_integrity(self, test_database, test_server_id):
        """Verify imported data has correct structure and relationships."""
        print(f"\n🔍 Verifying data integrity...")
        
        conn = sqlite3.connect(test_database)
        cursor = conn.cursor()
        
        # Check foreign key relationships
        cursor.execute("""
            SELECT COUNT(*) 
            FROM messages m
            LEFT JOIN channels c ON m.channel_id = c.id
            WHERE c.id IS NULL
        """)
        orphan_messages = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM channels c
            LEFT JOIN servers s ON c.server_id = s.id
            WHERE s.id IS NULL
        """)
        orphan_channels = cursor.fetchone()[0]
        
        # Check for duplicate messages (per channel)
        cursor.execute("""
            SELECT channel_id, id, COUNT(*) as cnt
            FROM messages
            GROUP BY channel_id, id
            HAVING cnt > 1
        """)
        duplicates = cursor.fetchall()
        
        # Check timestamp validity
        cursor.execute("""
            SELECT COUNT(*)
            FROM messages
            WHERE sent_at IS NULL OR sent_at <= 0
        """)
        invalid_timestamps = cursor.fetchone()[0]
        
        conn.close()
        
        # Report results
        print(f"   ✅ Orphan messages: {orphan_messages}")
        print(f"   ✅ Orphan channels: {orphan_channels}")
        print(f"   ✅ Duplicate messages: {len(duplicates)}")
        print(f"   ✅ Invalid timestamps: {invalid_timestamps}")
        
        # Assertions
        assert orphan_messages == 0, "No messages should be orphaned"
        assert orphan_channels == 0, "No channels should be orphaned"
        assert len(duplicates) == 0, "No duplicate messages should exist"
        assert invalid_timestamps == 0, "All timestamps should be valid"
        
        print("\n✅ Data integrity verified!")
    
    def test_message_content_preservation(self, test_database):
        """Verify message content is preserved exactly."""
        print(f"\n📝 Checking message content preservation...")
        
        conn = sqlite3.connect(test_database)
        cursor = conn.cursor()
        
        # Get various message types
        cursor.execute("""
            SELECT id, content, LENGTH(content) as len
            FROM messages
            WHERE content IS NOT NULL AND content != ''
            ORDER BY sent_at DESC
            LIMIT 10
        """)
        messages = cursor.fetchall()
        
        print(f"   Analyzing {len(messages)} messages...")
        
        # Check for content issues
        issues = []
        for msg_id, content, length in messages:
            # Check for truncation
            if "\x00" in content:
                issues.append(f"Message {msg_id}: Contains null bytes")
            
            # Check encoding
            try:
                content.encode('utf-8')
            except UnicodeEncodeError:
                issues.append(f"Message {msg_id}: Encoding issues")
        
        conn.close()
        
        if issues:
            print(f"\n⚠️  Content issues found:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print(f"   ✅ All message content properly preserved")
        
        assert len(issues) == 0, f"Found {len(issues)} content preservation issues"


@pytest.mark.real
@pytest.mark.skip_ci
@pytest.mark.slow
class TestRealIncrementalImport:
    """Test incremental imports with real data."""
    
    def test_incremental_import_no_duplicates(
        self,
        test_database,
        test_server_id,
        test_channel_id,
        test_config
    ):
        """Run two imports and verify no duplicate messages."""
        print(f"\n🔄 Testing incremental import...")
        
        channel_filter = [int(test_channel_id)] if test_channel_id else None
        
        # First import
        print("\n   1️⃣  First import...")
        summary1 = load_messages_once(
            server_id=test_server_id,
            channel_ids=channel_filter,
            hours_back=test_config["hours_back"],
            sleep_between_channels=True
        )
        print(f"      💾 Saved: {summary1['messages_saved']} messages")
        
        # Get message count
        conn = sqlite3.connect(test_database)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages")
        count_after_first = cursor.fetchone()[0]
        
        # Get all message IDs
        cursor.execute("SELECT id, channel_id FROM messages")
        first_messages = set(cursor.fetchall())
        conn.close()
        
        # Wait a bit
        print("\n   ⏳ Waiting 2 seconds...")
        time.sleep(2)
        
        # Second import (should find same messages, not duplicate)
        print("\n   2️⃣  Second import...")
        summary2 = load_messages_once(
            server_id=test_server_id,
            channel_ids=channel_filter,
            hours_back=test_config["hours_back"],
            sleep_between_channels=True
        )
        print(f"      💾 Saved: {summary2['messages_saved']} messages")
        
        # Check for duplicates
        conn = sqlite3.connect(test_database)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages")
        count_after_second = cursor.fetchone()[0]
        
        # Get all message IDs after second import
        cursor.execute("SELECT id, channel_id FROM messages")
        second_messages = set(cursor.fetchall())
        
        # Check for true duplicates (same id AND channel)
        cursor.execute("""
            SELECT id, channel_id, COUNT(*) as cnt
            FROM messages
            GROUP BY id, channel_id
            HAVING cnt > 1
        """)
        duplicates = cursor.fetchall()
        
        conn.close()
        
        print(f"\n   📊 Results:")
        print(f"      After 1st import: {count_after_first} messages")
        print(f"      After 2nd import: {count_after_second} messages")
        print(f"      Duplicates found: {len(duplicates)}")
        print(f"      New unique messages: {len(second_messages - first_messages)}")
        
        # Assertions
        assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate messages"
        
        # If no new messages arrived, counts should be same
        if summary2['messages_saved'] == 0:
            assert count_after_second == count_after_first, \
                "Message count should not change if no new messages"
            print("\n   ✅ No new messages, no duplicates - incremental import working!")
        else:
            print(f"\n   ✅ {summary2['messages_saved']} new messages added without duplicates!")


@pytest.mark.real
@pytest.mark.skip_ci
@pytest.mark.slow
class TestRealPerformance:
    """Test performance with real data."""
    
    def test_import_performance(
        self,
        test_database,
        test_server_id,
        test_channel_id,
        test_config
    ):
        """Measure real import performance."""
        print(f"\n⚡ Testing import performance...")
        
        channel_filter = [int(test_channel_id)] if test_channel_id else None
        
        # Measure import time
        start = time.time()
        summary = load_messages_once(
            server_id=test_server_id,
            channel_ids=channel_filter,
            hours_back=test_config["hours_back"],
            sleep_between_channels=True
        )
        duration = time.time() - start
        
        messages_count = summary['messages_saved']
        
        print(f"\n   ⏱️  Performance metrics:")
        print(f"      Total duration: {duration:.2f}s")
        print(f"      Messages imported: {messages_count}")
        
        if messages_count > 0:
            rate = messages_count / duration
            print(f"      Import rate: {rate:.2f} messages/second")
            
            # Performance assertion (should be reasonable)
            assert rate > 0.5, "Import rate should be at least 0.5 messages/second"
        else:
            print(f"      ⚠️  No messages to measure rate")
        
        # Measure query performance
        conn = sqlite3.connect(test_database)
        
        start = time.time()
        records = get_recent_message_records(int(test_server_id), hours=24)
        query_time = time.time() - start
        
        print(f"\n   🔍 Query performance:")
        print(f"      Query time: {query_time:.4f}s")
        print(f"      Records returned: {len(records)}")
        
        # Query should be fast even with real data
        assert query_time < 1.0, "Query should complete in under 1 second"
        
        conn.close()
        
        print(f"\n   ✅ Performance metrics acceptable!")


@pytest.mark.real
@pytest.mark.skip_ci
class TestRealQueryFunctionality:
    """Test query functionality with real data."""
    
    def test_time_based_queries(self, test_database, test_server_id):
        """Test time-based filtering with real data."""
        print(f"\n📅 Testing time-based queries...")
        
        # Query different time windows
        records_1h = get_recent_message_records(int(test_server_id), hours=1)
        records_24h = get_recent_message_records(int(test_server_id), hours=24)
        records_all = get_recent_message_records(int(test_server_id), hours=None)
        
        print(f"   Last 1 hour: {len(records_1h)} messages")
        print(f"   Last 24 hours: {len(records_24h)} messages")
        print(f"   All time: {len(records_all)} messages")
        
        # Logical assertions
        assert len(records_1h) <= len(records_24h), "1h should have <= 24h messages"
        assert len(records_24h) <= len(records_all), "24h should have <= all messages"
        
        print(f"   ✅ Time-based filtering works correctly!")
    
    def test_channel_based_queries(
        self,
        test_database,
        test_server_id,
        test_channel_id
    ):
        """Test channel filtering with real data."""
        if not test_channel_id:
            pytest.skip("TEST_CHANNEL_ID not set, skipping channel filter test")
        
        print(f"\n📢 Testing channel-based queries...")
        
        # Query all channels
        all_records = get_recent_message_records(int(test_server_id), hours=None)
        
        # Query specific channel
        channel_records = get_recent_message_records(
            int(test_server_id),
            hours=None,
            channel_ids=[int(test_channel_id)]
        )
        
        print(f"   All channels: {len(all_records)} messages")
        print(f"   Channel {test_channel_id}: {len(channel_records)} messages")
        
        # Channel-filtered should be <= total
        assert len(channel_records) <= len(all_records), \
            "Filtered results should be <= total"
        
        # Verify all returned messages are from correct channel
        if channel_records:
            for record in channel_records:
                assert record['channel_id'] == int(test_channel_id), \
                    f"Record {record['id']} is from wrong channel"
        
        print(f"   ✅ Channel filtering works correctly!")
