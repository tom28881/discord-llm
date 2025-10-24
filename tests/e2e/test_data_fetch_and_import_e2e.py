"""
End-to-end tests for complete data fetch and import pipeline.
Tests the entire flow from Discord API through to database queries.
"""
import pytest
import sqlite3
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import threading

from lib.database import (
    init_db, get_recent_message_records, get_servers, get_channels,
    get_recent_messages, get_latest_message_timestamp
)
from load_messages import load_messages_once


@pytest.mark.e2e
@pytest.mark.slow
class TestFullImportAndQueryCycle:
    """Test complete cycle: API → Import → Database → Query → Verify."""
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_complete_user_workflow(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test complete user workflow from import to query."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        # Setup mock Discord API with realistic data
        server_id = 123456789012345678
        channel1_id = 111111111111111111
        channel2_id = 222222222222222222
        
        now = datetime.now()
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [
            (server_id, "My Discord Server")
        ]
        mock_client.get_channel_ids.return_value = [
            (channel1_id, "general"),
            (channel2_id, "announcements"),
        ]
        
        # Realistic message data
        def fetch_messages(channel_id, *args, **kwargs):
            if channel_id == channel1_id:
                return [
                    (1001, "Hey everyone! Group buy for keyboards closing tomorrow!", 
                     int((now - timedelta(hours=2)).timestamp())),
                    (1002, "Count me in! What's the price?", 
                     int((now - timedelta(hours=1, minutes=50)).timestamp())),
                    (1003, "$120 for the full set with shipping", 
                     int((now - timedelta(hours=1, minutes=45)).timestamp())),
                ]
            else:  # announcements channel
                return [
                    (2001, "Server maintenance scheduled for tonight at 10 PM", 
                     int((now - timedelta(hours=3)).timestamp())),
                    (2002, "Maintenance completed successfully!", 
                     int((now - timedelta(minutes=30)).timestamp())),
                ]
        
        mock_client.fetch_messages.side_effect = fetch_messages
        mock_init_client.return_value = mock_client
        
        # Step 1: Import messages
        import_summary = load_messages_once()
        
        # Verify import summary
        assert import_summary["processed_servers"] == 1
        assert import_summary["messages_saved"] == 5
        
        # Step 2: Verify data structure
        servers = get_servers()
        assert len(servers) == 1
        assert servers[0][1] == "My Discord Server"
        
        channels = get_channels(server_id)
        assert len(channels) == 2
        channel_names = [ch[1] for ch in channels]
        assert "general" in channel_names
        assert "announcements" in channel_names
        
        # Step 3: Verify import summary
        assert import_summary["messages_saved"] == 5
        assert import_summary["processed_servers"] == 1
        
        # Step 4: Verify servers and channels created
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM servers WHERE id = ?", (server_id,))
        assert cursor.fetchone()[0] == 1
        
        cursor.execute("SELECT COUNT(*) FROM channels WHERE server_id = ?", (server_id,))
        assert cursor.fetchone()[0] == 2
        
        cursor.execute("SELECT COUNT(*) FROM messages WHERE server_id = ?", (server_id,))
        assert cursor.fetchone()[0] == 5
        
        # Verify message content
        cursor.execute("SELECT content FROM messages WHERE server_id = ? AND content LIKE ?", 
                      (server_id, "%keyboard%"))
        keyboard_msg = cursor.fetchone()
        assert keyboard_msg is not None
        assert "Group buy" in keyboard_msg[0]
        
        conn.close()
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_multi_import_cycle(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test multiple import cycles to verify incremental updates."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [(server_id, "Test Server")]
        mock_client.get_channel_ids.return_value = [(channel_id, "general")]
        
        # Cycle 1: Initial import
        mock_client.fetch_messages.return_value = [
            (1001, "Message 1", 1703030400),
            (1002, "Message 2", 1703030500),
        ]
        mock_init_client.return_value = mock_client
        
        summary1 = load_messages_once()
        assert summary1["messages_saved"] == 2
        
        # Cycle 2: New messages arrive
        mock_client.fetch_messages.return_value = [
            (1003, "Message 3", 1703030600),
        ]
        
        summary2 = load_messages_once()
        assert summary2["messages_saved"] == 1
        
        # Cycle 3: No new messages
        mock_client.fetch_messages.return_value = []
        
        summary3 = load_messages_once()
        assert summary3["messages_saved"] == 0
        
        # Verify total messages imported across all cycles
        total_imported = summary1["messages_saved"] + summary2["messages_saved"] + summary3["messages_saved"]
        assert total_imported == 3


@pytest.mark.e2e
@pytest.mark.slow
class TestDataIntegrity:
    """Test data integrity throughout the import process."""
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_message_content_preservation(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test that message content is preserved exactly."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        # Test various message formats
        test_messages = [
            (1001, "Simple message", 1703030400),
            (1002, "Message with emoji 🎉🚀", 1703030401),
            (1003, "Message with\nmultiple\nlines", 1703030402),
            (1004, "Message with special chars: @user #channel <:emoji:123>", 1703030403),
            (1005, "Very long message " + "x" * 1000, 1703030404),
            (1006, "", 1703030405),  # Empty message
        ]
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [(server_id, "Test Server")]
        mock_client.get_channel_ids.return_value = [(channel_id, "general")]
        mock_client.fetch_messages.return_value = test_messages
        mock_init_client.return_value = mock_client
        
        # Import
        summary = load_messages_once()
        
        # Verify all messages were imported
        assert summary["messages_saved"] == len(test_messages)
        
        # Verify exact content preservation by querying database directly
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        for original_msg in test_messages:
            msg_id, original_content, _ = original_msg
            cursor.execute("SELECT content FROM messages WHERE id = ? AND channel_id = ?", 
                          (msg_id, channel_id))
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == original_content
        
        conn.close()
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_timestamp_accuracy(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test that timestamps are stored and retrieved accurately."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        # Use specific timestamps
        timestamp1 = int(datetime(2024, 1, 15, 10, 0, 0).timestamp())
        timestamp2 = int(datetime(2024, 1, 15, 11, 30, 45).timestamp())
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [(server_id, "Test Server")]
        mock_client.get_channel_ids.return_value = [(channel_id, "general")]
        mock_client.fetch_messages.return_value = [
            (1001, "Message 1", timestamp1),
            (1002, "Message 2", timestamp2),
        ]
        mock_init_client.return_value = mock_client
        
        load_messages_once()
        
        # Verify timestamps
        records = get_recent_message_records(server_id, hours=None)
        
        msg1 = next(r for r in records if r["id"] == 1001)
        msg2 = next(r for r in records if r["id"] == 1002)
        
        assert msg1["sent_at"] == timestamp1
        assert msg2["sent_at"] == timestamp2
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_foreign_key_relationships(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test that foreign key relationships are maintained correctly."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        server_id = 123456789012345678
        channel1_id = 111111111111111111
        channel2_id = 222222222222222222
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [(server_id, "Test Server")]
        mock_client.get_channel_ids.return_value = [
            (channel1_id, "general"),
            (channel2_id, "random"),
        ]
        
        def fetch_messages(channel_id, *args, **kwargs):
            if channel_id == channel1_id:
                return [(1001, "Message in general", 1703030400)]
            else:
                return [(2001, "Message in random", 1703030500)]
        
        mock_client.fetch_messages.side_effect = fetch_messages
        mock_init_client.return_value = mock_client
        
        load_messages_once()
        
        # Verify relationships in database
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # Check that messages reference correct channels
        cursor.execute("""
            SELECT m.id, m.channel_id, c.name 
            FROM messages m
            JOIN channels c ON m.channel_id = c.id
        """)
        results = cursor.fetchall()
        
        assert len(results) == 2
        
        msg1 = next(r for r in results if r[0] == 1001)
        msg2 = next(r for r in results if r[0] == 2001)
        
        assert msg1[1] == channel1_id
        assert msg1[2] == "general"
        assert msg2[1] == channel2_id
        assert msg2[2] == "random"
        
        conn.close()


@pytest.mark.e2e
@pytest.mark.slow
class TestPerformanceAndScaling:
    """Test performance with realistic data volumes."""
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_large_batch_import(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test importing a large batch of messages."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        # Generate 1000 messages
        large_batch = [
            (i, f"Message {i}", 1703030400 + i)
            for i in range(1000, 2000)
        ]
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [(server_id, "Test Server")]
        mock_client.get_channel_ids.return_value = [(channel_id, "general")]
        mock_client.fetch_messages.return_value = large_batch
        mock_init_client.return_value = mock_client
        
        # Measure import time
        start_time = time.time()
        summary = load_messages_once()
        import_duration = time.time() - start_time
        
        # Verify all messages imported
        assert summary["messages_saved"] == 1000
        
        # Verify in database
        records = get_recent_message_records(server_id, hours=None)
        assert len(records) == 1000
        
        # Performance assertion (should complete in reasonable time)
        assert import_duration < 10.0  # Should take less than 10 seconds
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_query_performance(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test query performance with significant data."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        # Import 500 messages
        messages = [
            (i, f"Message {i}", 1703030400 + i * 60)
            for i in range(1000, 1500)
        ]
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [(server_id, "Test Server")]
        mock_client.get_channel_ids.return_value = [(channel_id, "general")]
        mock_client.fetch_messages.return_value = messages
        mock_init_client.return_value = mock_client
        
        load_messages_once()
        
        # Test various query patterns
        start = time.time()
        records_all = get_recent_message_records(server_id, hours=None)
        query1_time = time.time() - start
        
        start = time.time()
        records_24h = get_recent_message_records(server_id, hours=24)
        query2_time = time.time() - start
        
        start = time.time()
        records_channel = get_recent_message_records(
            server_id, 
            hours=None,
            channel_ids=[channel_id]
        )
        query3_time = time.time() - start
        
        # All queries should be fast
        assert query1_time < 0.5
        assert query2_time < 0.5
        assert query3_time < 0.5
        
        assert len(records_all) == 500


@pytest.mark.e2e
@pytest.mark.slow
class TestConcurrentOperations:
    """Test concurrent import and query operations."""
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_concurrent_queries_during_import(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test that queries work correctly during imports."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        # Initial data
        initial_messages = [
            (i, f"Initial message {i}", 1703030400 + i)
            for i in range(1000, 1050)
        ]
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [(server_id, "Test Server")]
        mock_client.get_channel_ids.return_value = [(channel_id, "general")]
        mock_client.fetch_messages.return_value = initial_messages
        mock_init_client.return_value = mock_client
        
        # Initial import
        load_messages_once()
        
        # Track query results during concurrent operations
        query_results = []
        errors = []
        
        def concurrent_query():
            try:
                for _ in range(5):
                    records = get_recent_message_records(server_id, hours=None)
                    query_results.append(len(records))
                    time.sleep(0.1)
            except Exception as e:
                errors.append(e)
        
        def concurrent_import():
            try:
                new_messages = [
                    (i, f"New message {i}", 1703030400 + i)
                    for i in range(1050, 1060)
                ]
                mock_client.fetch_messages.return_value = new_messages
                load_messages_once()
            except Exception as e:
                errors.append(e)
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(concurrent_query)
            time.sleep(0.05)  # Small delay
            future2 = executor.submit(concurrent_import)
            
            future1.result()
            future2.result()
        
        # Should complete without errors
        assert len(errors) == 0
        
        # Queries should have returned valid results
        assert len(query_results) > 0
        assert all(r >= 50 for r in query_results)  # At least initial messages


@pytest.mark.e2e
@pytest.mark.slow
class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_typical_daily_usage(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Simulate typical daily usage pattern."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        server_id = 123456789012345678
        channels = [
            (111111111111111111, "general"),
            (222222222222222222, "announcements"),
            (333333333333333333, "tech-talk"),
        ]
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [(server_id, "My Server")]
        mock_client.get_channel_ids.return_value = channels
        
        now = datetime.now()
        
        # Morning: Initial import with overnight messages
        def morning_fetch(channel_id, *args, **kwargs):
            base = int(channel_id)
            return [
                (base + 1, f"Good morning from {channel_id}", 
                 int((now - timedelta(hours=8)).timestamp())),
                (base + 2, f"Morning update in {channel_id}", 
                 int((now - timedelta(hours=7)).timestamp())),
            ]
        
        mock_client.fetch_messages.side_effect = morning_fetch
        mock_init_client.return_value = mock_client
        
        summary1 = load_messages_once(hours_back=12)
        assert summary1["messages_saved"] == 6  # 2 messages × 3 channels
        
        # Afternoon: Check for new messages
        def afternoon_fetch(channel_id, *args, **kwargs):
            base = int(channel_id)
            return [
                (base + 3, f"Afternoon message in {channel_id}", 
                 int((now - timedelta(hours=2)).timestamp())),
            ]
        
        mock_client.fetch_messages.side_effect = afternoon_fetch
        
        summary2 = load_messages_once(hours_back=4)
        assert summary2["messages_saved"] == 3  # 1 message × 3 channels
        
        # Evening: Query messages from the day
        all_today = get_recent_message_records(server_id, hours=12)
        assert len(all_today) == 9  # All messages from today
        
        # Query specific channel
        general_messages = get_recent_message_records(
            server_id,
            hours=12,
            channel_ids=[111111111111111111]
        )
        assert len(general_messages) == 3  # Only general channel messages
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_weekend_catch_up(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Simulate catching up after a weekend away."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        now = datetime.now()
        
        # Generate messages from the last 48 hours
        weekend_messages = []
        for hour in range(48, 0, -1):
            for msg_num in range(3):  # 3 messages per hour
                msg_id = 1000 + (48 - hour) * 3 + msg_num
                timestamp = int((now - timedelta(hours=hour)).timestamp())
                weekend_messages.append(
                    (msg_id, f"Message from {hour}h ago", timestamp)
                )
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [(server_id, "Test Server")]
        mock_client.get_channel_ids.return_value = [(channel_id, "general")]
        mock_client.fetch_messages.return_value = weekend_messages
        mock_init_client.return_value = mock_client
        
        # Import all weekend messages
        summary = load_messages_once(hours_back=48)
        
        total_messages = len(weekend_messages)
        assert summary["messages_saved"] == total_messages
        
        # Query recent messages
        last_24h = get_recent_message_records(server_id, hours=24)
        last_12h = get_recent_message_records(server_id, hours=12)
        last_6h = get_recent_message_records(server_id, hours=6)
        
        # Verify time-based filtering works correctly
        assert len(last_24h) > len(last_12h)
        assert len(last_12h) > len(last_6h)
        
        # Approximately 72 messages in 24 hours (3 per hour)
        assert 60 <= len(last_24h) <= 80
