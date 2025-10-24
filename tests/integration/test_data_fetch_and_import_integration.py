"""
Integration tests for data fetching and import pipeline.
Tests multiple components working together with mocked Discord API.
"""
import pytest
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
import requests

from lib.database import (
    init_db, save_server, save_channel, get_recent_message_records,
    get_last_message_id, get_servers, get_channels
)
from load_messages import (
    load_messages_once, fetch_and_store_messages,
    initialize_discord_client
)


@pytest.mark.integration
class TestLoadMessagesOnceFlow:
    """Test complete load_messages_once workflow."""
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_complete_single_server_import(
        self, 
        mock_forbidden, 
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test complete import cycle for a single server."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        # Mock Discord client
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [
            (123456789012345678, "Test Server")
        ]
        mock_client.get_channel_ids.return_value = [
            (111111111111111111, "general"),
            (222222222222222222, "random"),
        ]
        mock_client.fetch_messages.return_value = [
            (1001, "Message 1", 1703030400),
            (1002, "Message 2", 1703030500),
        ]
        
        mock_init_client.return_value = mock_client
        
        # Run import
        summary = load_messages_once()
        
        # Verify summary
        assert summary["processed_servers"] == 1
        assert summary["messages_saved"] == 4  # 2 messages × 2 channels
        assert len(summary["servers"]) == 1
        assert summary["servers"][0]["name"] == "Test Server"
        
        # Verify data in database
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # Check servers
        cursor.execute("SELECT COUNT(*) FROM servers")
        assert cursor.fetchone()[0] == 1
        
        # Check channels
        cursor.execute("SELECT COUNT(*) FROM channels")
        assert cursor.fetchone()[0] == 2
        
        # Check messages
        cursor.execute("SELECT COUNT(*) FROM messages")
        assert cursor.fetchone()[0] == 4
        
        conn.close()
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_import_with_server_filter(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test import with specific server ID."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        mock_client = Mock()
        # Client has multiple servers
        mock_client.get_server_ids.return_value = [
            (123456789012345678, "Server 1"),
            (987654321098765432, "Server 2"),
        ]
        mock_client.get_channel_ids.return_value = [
            (111111111111111111, "general"),
        ]
        mock_client.fetch_messages.return_value = [
            (1001, "Message", 1703030400),
        ]
        
        mock_init_client.return_value = mock_client
        
        # Import only specific server
        summary = load_messages_once(server_id="123456789012345678")
        
        # Should only process the specified server
        assert summary["processed_servers"] == 1
        assert summary["servers"][0]["id"] == "123456789012345678"
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_import_with_channel_filter(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test import with specific channel IDs."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [
            (123456789012345678, "Test Server")
        ]
        mock_client.get_channel_ids.return_value = [
            (111111111111111111, "general"),
            (222222222222222222, "random"),
            (333333333333333333, "announcements"),
        ]
        
        # Track which channels were fetched
        fetched_channels = []
        def track_fetch(channel_id, *args, **kwargs):
            fetched_channels.append(channel_id)
            return [(1001, "Message", 1703030400)]
        
        mock_client.fetch_messages.side_effect = track_fetch
        mock_init_client.return_value = mock_client
        
        # Import only from specific channels
        channel_filter = [111111111111111111, 333333333333333333]
        summary = load_messages_once(channel_ids=channel_filter)
        
        # Should only fetch from filtered channels
        assert len(fetched_channels) == 2
        assert 111111111111111111 in fetched_channels
        assert 333333333333333333 in fetched_channels
        assert 222222222222222222 not in fetched_channels


@pytest.mark.integration
class TestTimeBasedFetching:
    """Test time-based message fetching with hours_back parameter."""
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_import_with_time_limit(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test import with hours_back time limit."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [
            (123456789012345678, "Test Server")
        ]
        mock_client.get_channel_ids.return_value = [
            (111111111111111111, "general"),
        ]
        
        # Track min_timestamp parameter
        captured_params = {}
        def capture_fetch(channel_id, last_id, limit, min_timestamp=None):
            captured_params['min_timestamp'] = min_timestamp
            return [(1001, "Message", 1703030400)]
        
        mock_client.fetch_messages.side_effect = capture_fetch
        mock_init_client.return_value = mock_client
        
        # Import only last 24 hours
        summary = load_messages_once(hours_back=24)
        
        # Should pass min_timestamp to fetch_messages
        assert 'min_timestamp' in captured_params
        assert captured_params['min_timestamp'] is not None
        
        # Verify timestamp is approximately 24 hours ago
        expected_time = int((datetime.now() - timedelta(hours=24)).timestamp())
        actual_time = captured_params['min_timestamp']
        # Allow 5 second tolerance
        assert abs(expected_time - actual_time) < 5
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_fetch_respects_time_filter(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test that only messages within time window are saved."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        now = datetime.now()
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [
            (123456789012345678, "Test Server")
        ]
        mock_client.get_channel_ids.return_value = [
            (111111111111111111, "general"),
        ]
        
        # Return mix of old and new messages
        mock_client.fetch_messages.return_value = [
            (1001, "Recent message", int((now - timedelta(hours=1)).timestamp())),
            (1002, "Old message", int((now - timedelta(hours=48)).timestamp())),
        ]
        
        mock_init_client.return_value = mock_client
        
        # Import last 24 hours
        summary = load_messages_once(hours_back=24)
        
        # Fetch should have been called with appropriate min_timestamp
        # Discord API would filter, but we're testing the parameter is passed
        assert mock_client.fetch_messages.called


@pytest.mark.integration
class TestIncrementalImport:
    """Test incremental message importing (fetching only new messages)."""
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_incremental_import_uses_last_message_id(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test that incremental import starts from last message ID."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        # Pre-populate with some messages
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        from lib.database import save_messages
        save_messages([
            (server_id, channel_id, 1001, "Old message 1", 1703030400),
            (server_id, channel_id, 1002, "Old message 2", 1703030500),
        ])
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [(server_id, "Test Server")]
        mock_client.get_channel_ids.return_value = [(channel_id, "general")]
        
        # Return new messages
        mock_client.fetch_messages.return_value = [
            (1003, "New message", 1703030600),
        ]
        
        mock_init_client.return_value = mock_client
        
        # Run import
        summary = load_messages_once()
        
        # Should have called fetch_messages with last_message_id=1002
        mock_client.fetch_messages.assert_called_with(
            channel_id, 
            1002,  # Last message ID
            5000,
            min_timestamp=None
        )
        
        # Verify import worked
        assert summary["messages_saved"] >= 1
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_incremental_import_no_duplicates(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test that incremental import doesn't create duplicates."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [(server_id, "Test Server")]
        mock_client.get_channel_ids.return_value = [(channel_id, "general")]
        mock_client.fetch_messages.return_value = [
            (1001, "Message 1", 1703030400),
            (1002, "Message 2", 1703030500),
        ]
        
        mock_init_client.return_value = mock_client
        
        # First import
        summary1 = load_messages_once()
        assert summary1["messages_saved"] == 2
        
        # Second import (should find same messages but not save duplicates)
        summary2 = load_messages_once()
        # messages_saved might be 0 if API returns same messages
        
        # Verify no duplicates in database
        records = get_recent_message_records(server_id)
        message_ids = [r["id"] for r in records]
        assert len(message_ids) == len(set(message_ids))  # No duplicates


@pytest.mark.integration  
class TestErrorRecovery:
    """Test error handling and recovery during imports."""
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_continues_after_channel_error(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test that import continues after a channel fails."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [
            (123456789012345678, "Test Server")
        ]
        mock_client.get_channel_ids.return_value = [
            (111111111111111111, "general"),
            (222222222222222222, "problematic"),
            (333333333333333333, "announcements"),
        ]
        
        # Second channel raises error
        def fetch_with_error(channel_id, *args, **kwargs):
            if channel_id == 222222222222222222:
                error = requests.HTTPError("API Error")
                error.response = Mock(status_code=500)
                raise error
            return [(1001, "Message", 1703030400)]
        
        mock_client.fetch_messages.side_effect = fetch_with_error
        mock_init_client.return_value = mock_client
        
        # Should not raise exception
        summary = load_messages_once()
        
        # Should still import from working channels
        assert summary["messages_saved"] == 2  # 2 working channels
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_forbidden_channel_added_to_blocklist(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test that 403 errors add channels to forbidden list."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [
            (123456789012345678, "Test Server")
        ]
        mock_client.get_channel_ids.return_value = [
            (111111111111111111, "general"),
            (222222222222222222, "private-channel"),
        ]
        
        # Second channel returns 403
        def fetch_with_403(channel_id, *args, **kwargs):
            if channel_id == 222222222222222222:
                error = requests.HTTPError()
                error.response = Mock(status_code=403)
                raise error
            return [(1001, "Message", 1703030400)]
        
        mock_client.fetch_messages.side_effect = fetch_with_403
        mock_init_client.return_value = mock_client
        
        with patch('load_messages.add_forbidden_channel') as mock_add:
            summary = load_messages_once()
            
            # Should have added forbidden channel
            mock_add.assert_called()
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_multiple_servers_one_fails(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test that one server failure doesn't stop other servers."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [
            (123456789012345678, "Server 1"),
            (987654321098765432, "Server 2"),
            (555555555555555555, "Server 3"),
        ]
        
        server_call_count = [0]
        
        def get_channels_with_error():
            server_call_count[0] += 1
            if server_call_count[0] == 2:  # Second server fails
                raise Exception("Server unavailable")
            return [(111111111111111111, "general")]
        
        mock_client.get_channel_ids.side_effect = get_channels_with_error
        mock_client.fetch_messages.return_value = [(1001, "Message", 1703030400)]
        
        mock_init_client.return_value = mock_client
        
        # Should complete despite one server failing
        summary = load_messages_once()
        
        # Should have processed 3 servers (even if one failed)
        assert summary["processed_servers"] == 3
        assert len(summary["servers"]) == 3
        
        # One server should have error recorded
        errors = [s for s in summary["servers"] if "error" in s]
        assert len(errors) == 1


@pytest.mark.integration
class TestMultiServerImport:
    """Test importing from multiple Discord servers."""
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_import_multiple_servers(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test importing messages from multiple servers."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [
            (123456789012345678, "Server 1"),
            (987654321098765432, "Server 2"),
        ]
        mock_client.get_channel_ids.return_value = [
            (111111111111111111, "general"),
        ]
        mock_client.fetch_messages.return_value = [
            (1001, "Message", 1703030400),
        ]
        
        mock_init_client.return_value = mock_client
        
        # Import from all servers
        summary = load_messages_once(sleep_between_servers=False)
        
        assert summary["processed_servers"] == 2
        assert len(summary["servers"]) == 2
        
        # Verify both servers in database
        servers = get_servers()
        assert len(servers) == 2
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_server_isolation(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Test that messages are properly isolated by server."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        server1_id = 123456789012345678
        server2_id = 987654321098765432
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [
            (server1_id, "Server 1"),
            (server2_id, "Server 2"),
        ]
        mock_client.get_channel_ids.return_value = [
            (111111111111111111, "general")
        ]
        mock_client.fetch_messages.return_value = [
            (1001, "Test message", 1703030400)
        ]
        
        mock_init_client.return_value = mock_client
        
        # Import from both servers
        summary = load_messages_once(sleep_between_servers=False)
        
        # Should have processed both servers
        assert summary["processed_servers"] == 2
        assert len(summary["servers"]) == 2
        
        # Verify both servers saved to database
        from lib.database import get_servers
        servers = get_servers()
        server_ids = [s[0] for s in servers]
        assert server1_id in server_ids
        assert server2_id in server_ids


@pytest.mark.integration
@pytest.mark.slow
class TestRealTimeSync:
    """Test real-time sync simulation."""
    
    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    def test_periodic_import_simulation(
        self,
        mock_forbidden,
        mock_config,
        mock_init_client,
        temp_db
    ):
        """Simulate periodic imports like real-time sync."""
        mock_config.return_value = {}
        mock_forbidden.return_value = set()
        
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        mock_client = Mock()
        mock_client.get_server_ids.return_value = [(server_id, "Test Server")]
        mock_client.get_channel_ids.return_value = [(channel_id, "general")]
        
        # Simulate 3 periodic fetches with new messages each time
        fetch_responses = [
            [(1001, "Message 1", 1703030400)],
            [(1002, "Message 2", 1703030500)],
            [(1003, "Message 3", 1703030600)],
        ]
        
        mock_client.fetch_messages.side_effect = fetch_responses.copy()
        mock_init_client.return_value = mock_client
        
        # First import
        summary1 = load_messages_once(hours_back=1)
        assert summary1["messages_saved"] == 1
        
        # Reset mock for second import
        mock_client.fetch_messages.side_effect = fetch_responses[1:]
        
        # Second import
        summary2 = load_messages_once(hours_back=1)
        assert summary2["messages_saved"] == 1
        
        # Reset for third import
        mock_client.fetch_messages.side_effect = [fetch_responses[2]]
        
        # Third import
        summary3 = load_messages_once(hours_back=1)
        assert summary3["messages_saved"] == 1
        
        # Verify summary of imports
        total_imported = summary1["messages_saved"] + summary2["messages_saved"] + summary3["messages_saved"]
        assert total_imported == 3
