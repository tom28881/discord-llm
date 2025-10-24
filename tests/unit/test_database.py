"""
Unit tests for database operations.
"""
import pytest
import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

from lib.database import (
    init_db, save_server, save_channel, save_messages,
    get_last_message_id, get_recent_messages, get_unique_server_ids,
    get_servers, get_channels, get_recent_message_records,
    get_latest_message_timestamp
)


@pytest.mark.unit
@pytest.mark.database
class TestDatabaseOperations:
    """Test database operations."""

    def test_init_db(self, temp_db):
        """Test database initialization."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert "servers" in tables
        assert "channels" in tables
        assert "messages" in tables
        
        conn.close()

    def test_save_server(self, temp_db):
        """Test saving server information."""
        server_id = 123456789012345678
        server_name = "Test Server"
        
        save_server(server_id, server_name)
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM servers WHERE id = ?", (server_id,))
        result = cursor.fetchone()
        conn.close()
        
        assert result is not None
        assert result[0] == server_id
        assert result[1] == server_name

    def test_save_channel(self, temp_db):
        """Test saving channel information."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        channel_name = "general"
        
        # First save the server
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, channel_name)
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT id, server_id, name FROM channels WHERE id = ?", (channel_id,))
        result = cursor.fetchone()
        conn.close()
        
        assert result is not None
        assert result[0] == channel_id
        assert result[1] == server_id
        assert result[2] == channel_name

    def test_save_messages(self, temp_db):
        """Test saving messages."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        # Setup server and channel
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        
        messages = [
            (server_id, channel_id, 1001, "Hello world!", 1703030400),
            (server_id, channel_id, 1002, "How are you?", 1703030500),
        ]
        
        save_messages(messages)
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages WHERE server_id = ?", (server_id,))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 2

    def test_get_last_message_id(self, temp_db):
        """Test getting the last message ID."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        # Setup
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        
        messages = [
            (server_id, channel_id, 1001, "First message", 1703030400),
            (server_id, channel_id, 1002, "Second message", 1703030500),
            (server_id, channel_id, 1003, "Latest message", 1703030600),
        ]
        save_messages(messages)
        
        last_id = get_last_message_id(server_id, channel_id)
        assert last_id == 1003

    def test_get_recent_messages(self, temp_db):
        """Test getting recent messages."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        # Setup
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        
        now = datetime.now()
        old_timestamp = int((now - timedelta(hours=25)).timestamp())
        recent_timestamp = int((now - timedelta(hours=1)).timestamp())
        
        messages = [
            (server_id, channel_id, 1001, "Old message", old_timestamp),
            (server_id, channel_id, 1002, "Recent message", recent_timestamp),
        ]
        save_messages(messages)
        
        recent_messages = get_recent_messages(server_id, hours=24)
        assert len(recent_messages) == 1
        assert "Recent message" in recent_messages[0]

    def test_get_recent_messages_with_keywords(self, temp_db):
        """Test getting recent messages with keyword filtering."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        # Setup
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        
        now = datetime.now()
        recent_timestamp = int((now - timedelta(hours=1)).timestamp())
        
        messages = [
            (server_id, channel_id, 1001, "Group buy for keyboards", recent_timestamp),
            (server_id, channel_id, 1002, "Just a regular message", recent_timestamp),
            (server_id, channel_id, 1003, "Another group buy announcement", recent_timestamp),
        ]
        save_messages(messages)
        
        filtered_messages = get_recent_messages(server_id, hours=24, keywords=["group buy"])
        assert len(filtered_messages) == 2

    def test_get_unique_server_ids(self, temp_db):
        """Test getting unique server IDs."""
        server_ids = [123456789012345678, 987654321098765432]
        
        for server_id in server_ids:
            save_server(server_id, f"Server {server_id}")
            save_channel(111111111111111111 + server_id, server_id, "general")
            save_messages([(server_id, 111111111111111111 + server_id, 1001, "Test", 1703030400)])
        
        unique_ids = get_unique_server_ids()
        assert set(unique_ids) == set(server_ids)

    def test_get_servers(self, temp_db):
        """Test getting all servers."""
        servers = [
            (123456789012345678, "Test Server 1"),
            (987654321098765432, "Test Server 2"),
        ]
        
        for server_id, server_name in servers:
            save_server(server_id, server_name)
        
        retrieved_servers = get_servers()
        assert len(retrieved_servers) == 2
        assert (123456789012345678, "Test Server 1") in retrieved_servers

    def test_get_channels(self, temp_db):
        """Test getting channels for a server."""
        server_id = 123456789012345678
        save_server(server_id, "Test Server")
        
        channels = [
            (111111111111111111, "general"),
            (222222222222222222, "random"),
        ]
        
        for channel_id, channel_name in channels:
            save_channel(channel_id, server_id, channel_name)
        
        retrieved_channels = get_channels(server_id)
        assert len(retrieved_channels) == 2
        assert (111111111111111111, "general") in retrieved_channels

    def test_database_error_handling(self, temp_db):
        """Test database error handling."""
        # Test saving invalid data
        with patch('lib.database.logger') as mock_logger:
            # Try to save a message without proper server/channel setup
            save_messages([(999999999999999999, 888888888888888888, 1001, "Test", 1703030400)])
            # Should not raise exception but should log error
            assert mock_logger.error.called

    def test_concurrent_access(self, temp_db):
        """Test concurrent database access."""
        import threading
        import time
        
        server_id = 123456789012345678
        save_server(server_id, "Test Server")
        save_channel(111111111111111111, server_id, "general")
        
        def insert_messages(start_id):
            messages = [
                (server_id, 111111111111111111, start_id + i, f"Message {start_id + i}", 1703030400 + i)
                for i in range(10)
            ]
            save_messages(messages)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=insert_messages, args=(i * 10,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all messages were saved
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 50  # 5 threads * 10 messages each

    def test_recent_message_records_multi_channel(self, temp_db):
        """Test fetching recent message records with multi-channel filtering."""
        server_id = 555
        general_id = 100
        random_id = 200

        save_server(server_id, "Test Server")
        save_channel(general_id, server_id, "general")
        save_channel(random_id, server_id, "random")

        base_ts = int(datetime.now().timestamp())
        messages = [
            (server_id, general_id, 1, "General info", base_ts - 120),
            (server_id, random_id, 2, "Random chat", base_ts - 60),
            (server_id, general_id, 3, "General follow up", base_ts - 30),
        ]
        save_messages(messages)

        records = get_recent_message_records(
            server_id,
            hours=24,
            channel_ids=[general_id],
        )

        assert [record["channel_id"] for record in records] == [general_id, general_id]
        assert all("General" in record["content"] for record in records)

        keyword_records = get_recent_message_records(
            server_id,
            hours=24,
            keywords=["follow"],
            channel_ids=[general_id, random_id],
        )

        assert len(keyword_records) == 1
        assert keyword_records[0]["id"] == 3

    def test_get_latest_message_timestamp(self, temp_db):
        """Test retrieving the latest message timestamp for selected channels."""
        server_id = 777
        channel_ids = [300, 400]

        save_server(server_id, "Timestamp Server")
        for idx, channel_id in enumerate(channel_ids):
            save_channel(channel_id, server_id, f"channel-{idx}")

        base_ts = int(datetime.now().timestamp())
        save_messages([
            (server_id, channel_ids[0], 10, "Older", base_ts - 300),
            (server_id, channel_ids[1], 11, "Newer", base_ts - 60),
        ])

        latest_all = get_latest_message_timestamp(server_id)
        latest_filtered = get_latest_message_timestamp(server_id, channel_ids=[channel_ids[0]])

        assert latest_all == base_ts - 60
        assert latest_filtered == base_ts - 300

    def test_get_recent_messages_with_channel_ids(self, temp_db):
        """Ensure legacy helper supports channel_ids parameter."""
        server_id = 888
        channel_primary = 500
        channel_secondary = 600

        save_server(server_id, "Legacy Server")
        save_channel(channel_primary, server_id, "primary")
        save_channel(channel_secondary, server_id, "secondary")

        base_ts = int(datetime.now().timestamp())
        save_messages([
            (server_id, channel_primary, 21, "Primary message", base_ts - 10),
            (server_id, channel_secondary, 22, "Secondary message", base_ts - 5),
        ])

        messages = get_recent_messages(
            server_id,
            hours=1,
            channel_ids=[channel_primary],
        )

        assert messages == ["Primary message"]


@pytest.mark.unit
@pytest.mark.database
@pytest.mark.performance
class TestDatabasePerformance:
    """Test database performance."""

    def test_bulk_insert_performance(self, temp_db, performance_baseline):
        """Test bulk insert performance."""
        import time
        
        server_id = 123456789012345678
        save_server(server_id, "Test Server")
        save_channel(111111111111111111, server_id, "general")
        
        # Create 1000 test messages
        messages = [
            (server_id, 111111111111111111, i, f"Test message {i}", 1703030400 + i)
            for i in range(1000)
        ]
        
        start_time = time.time()
        save_messages(messages)
        end_time = time.time()
        
        execution_time = end_time - start_time
        messages_per_second = len(messages) / execution_time
        
        # Should be faster than baseline
        assert messages_per_second >= performance_baseline["message_import_rate"]

    def test_query_performance(self, database_with_sample_data, performance_baseline):
        """Test query performance."""
        import time
        
        start_time = time.time()
        messages = get_recent_messages(123456789012345678, hours=24)
        end_time = time.time()
        
        query_time = end_time - start_time
        assert query_time <= performance_baseline["query_response_time"]
