"""
Unit tests for data fetching and import functionality.
Focus on testing core functions in isolation with mocked dependencies.
"""
import pytest
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import requests

from lib.database import (
    save_messages, _normalize_timestamp, _deduplicate_messages,
    get_last_message_id, save_server, save_channel,
    get_recent_message_records
)
from load_messages import (
    fetch_and_store_messages, handle_http_error, 
    initialize_discord_client
)


@pytest.mark.unit
class TestTimestampNormalization:
    """Test timestamp normalization with various input formats."""
    
    def test_normalize_timestamp_from_int(self):
        """Test normalization from integer timestamp."""
        timestamp = 1703030400
        result = _normalize_timestamp(timestamp)
        assert result == 1703030400
        assert isinstance(result, int)
    
    def test_normalize_timestamp_from_float(self):
        """Test normalization from float timestamp."""
        timestamp = 1703030400.5
        result = _normalize_timestamp(timestamp)
        assert result == 1703030400
        assert isinstance(result, int)
    
    def test_normalize_timestamp_from_datetime(self):
        """Test normalization from datetime object."""
        dt = datetime(2024, 1, 15, 10, 0, 0)
        result = _normalize_timestamp(dt)
        assert isinstance(result, int)
        assert result == int(dt.timestamp())
    
    def test_normalize_timestamp_from_iso_string(self):
        """Test normalization from ISO format string."""
        iso_string = "2024-01-15T10:00:00+00:00"
        result = _normalize_timestamp(iso_string)
        assert isinstance(result, int)
        expected = int(datetime.fromisoformat(iso_string).timestamp())
        assert result == expected
    
    def test_normalize_timestamp_from_iso_string_with_z(self):
        """Test normalization from ISO format string with Z suffix."""
        iso_string = "2024-01-15T10:00:00Z"
        result = _normalize_timestamp(iso_string)
        assert isinstance(result, int)
        expected = int(datetime.fromisoformat(iso_string.replace('Z', '+00:00')).timestamp())
        assert result == expected
    
    def test_normalize_timestamp_invalid_string(self):
        """Test that invalid string raises error."""
        with pytest.raises(ValueError):
            _normalize_timestamp("invalid-date-string")
    
    def test_normalize_timestamp_invalid_type(self):
        """Test that invalid type raises error."""
        with pytest.raises(TypeError):
            _normalize_timestamp(None)
        
        with pytest.raises(TypeError):
            _normalize_timestamp([1703030400])


@pytest.mark.unit
class TestMessageDeduplication:
    """Test message deduplication logic."""
    
    def test_deduplicate_same_channel(self):
        """Test deduplication within same channel."""
        rows = [
            (111111111111111111, 1001, "Message 1"),
            (111111111111111111, 1001, "Message 1 duplicate"),
            (111111111111111111, 1002, "Message 2"),
        ]
        result = _deduplicate_messages(rows)
        assert len(result) == 2
        assert result[0] == "Message 1"
        assert result[1] == "Message 2"
    
    def test_deduplicate_across_channels(self):
        """Test deduplication across different channels."""
        rows = [
            (111111111111111111, 1001, "Message 1"),
            (222222222222222222, 1001, "Message 1 in another channel"),
            (111111111111111111, 1002, "Message 2"),
        ]
        result = _deduplicate_messages(rows)
        # Should keep only first occurrence of message ID 1001
        assert len(result) == 2
        assert result[0] == "Message 1"
        assert result[1] == "Message 2"
    
    def test_deduplicate_preserves_order(self):
        """Test that deduplication preserves message order."""
        rows = [
            (111111111111111111, 1001, "First"),
            (111111111111111111, 1002, "Second"),
            (222222222222222222, 1001, "First duplicate"),
            (111111111111111111, 1003, "Third"),
        ]
        result = _deduplicate_messages(rows)
        assert result == ["First", "Second", "Third"]
    
    def test_deduplicate_empty_list(self):
        """Test deduplication with empty list."""
        result = _deduplicate_messages([])
        assert result == []


@pytest.mark.unit
class TestSaveMessages:
    """Test message saving with various scenarios."""
    
    def test_save_messages_with_valid_data(self, temp_db):
        """Test saving messages with valid server and channel references."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
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
    
    def test_save_messages_duplicate_handling(self, temp_db):
        """Test that duplicate messages are ignored."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        
        messages = [
            (server_id, channel_id, 1001, "Hello world!", 1703030400),
        ]
        
        # Save once
        save_messages(messages)
        
        # Try to save again
        save_messages(messages)
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages WHERE id = ?", (1001,))
        count = cursor.fetchone()[0]
        conn.close()
        
        # Should only have one copy
        assert count == 1
    
    def test_save_messages_missing_server_reference(self, temp_db):
        """Test handling of messages with missing server reference."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        # Don't save server, only channel
        save_channel(channel_id, server_id, "general")
        
        messages = [
            (server_id, channel_id, 1001, "Hello world!", 1703030400),
        ]
        
        # Should handle gracefully without crashing
        save_messages(messages)
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages WHERE id = ?", (1001,))
        count = cursor.fetchone()[0]
        conn.close()
        
        # Message should not be saved due to missing FK
        assert count == 0
    
    def test_save_messages_missing_channel_reference(self, temp_db):
        """Test handling of messages with missing channel reference."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        # Save server but not channel
        save_server(server_id, "Test Server")
        
        messages = [
            (server_id, channel_id, 1001, "Hello world!", 1703030400),
        ]
        
        # Should handle gracefully
        save_messages(messages)
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages WHERE id = ?", (1001,))
        count = cursor.fetchone()[0]
        conn.close()
        
        # Message should not be saved due to missing FK
        assert count == 0
    
    def test_save_messages_with_datetime_timestamp(self, temp_db):
        """Test saving messages with datetime objects as timestamps."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        
        dt = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            (server_id, channel_id, 1001, "Hello world!", dt),
        ]
        
        save_messages(messages)
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT sent_at FROM messages WHERE id = ?", (1001,))
        result = cursor.fetchone()
        conn.close()
        
        assert result is not None
        assert result[0] == int(dt.timestamp())
    
    def test_save_messages_empty_list(self, temp_db):
        """Test saving empty message list."""
        # Should not crash
        save_messages([])
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 0


@pytest.mark.unit
class TestHttpErrorHandling:
    """Test HTTP error handling in fetch operations."""
    
    def test_handle_403_forbidden_error(self):
        """Test handling of 403 Forbidden errors."""
        http_err = requests.HTTPError()
        http_err.response = Mock()
        http_err.response.status_code = 403
        
        config = {"forbidden_channels": []}
        channel_id = "111111111111111111"
        channel_name = "private-channel"
        
        with patch('load_messages.add_forbidden_channel') as mock_add_forbidden:
            handle_http_error(http_err, channel_id, channel_name, config)
            mock_add_forbidden.assert_called_once_with(config, channel_id)
    
    def test_handle_500_server_error(self):
        """Test handling of 500 server errors."""
        http_err = requests.HTTPError()
        http_err.response = Mock()
        http_err.response.status_code = 500
        
        config = {"forbidden_channels": []}
        channel_id = "111111111111111111"
        channel_name = "general"
        
        # Should log error but not add to forbidden channels
        with patch('load_messages.add_forbidden_channel') as mock_add_forbidden:
            handle_http_error(http_err, channel_id, channel_name, config)
            mock_add_forbidden.assert_not_called()
    
    def test_handle_429_rate_limit_error(self):
        """Test handling of 429 rate limit errors."""
        http_err = requests.HTTPError()
        http_err.response = Mock()
        http_err.response.status_code = 429
        
        config = {"forbidden_channels": []}
        channel_id = "111111111111111111"
        channel_name = "general"
        
        # Should not add to forbidden channels
        with patch('load_messages.add_forbidden_channel') as mock_add_forbidden:
            handle_http_error(http_err, channel_id, channel_name, config)
            mock_add_forbidden.assert_not_called()


@pytest.mark.unit
class TestChannelFiltering:
    """Test channel filtering logic in fetch operations."""
    
    @patch('load_messages.save_server')
    @patch('load_messages.save_channel')
    @patch('load_messages.get_last_message_id')
    def test_fetch_with_channel_filter(self, mock_get_last, mock_save_channel, mock_save_server):
        """Test fetching messages with channel filter."""
        mock_get_last.return_value = None
        
        # Mock Discord client
        mock_client = Mock()
        mock_client.get_channel_ids.return_value = [
            (111111111111111111, "general"),
            (222222222222222222, "random"),
            (333333333333333333, "announcements"),
        ]
        mock_client.fetch_messages.return_value = [
            (1001, "Message 1", 1703030400),
        ]
        
        # Filter to only fetch from channel 111111111111111111
        channel_filter = [111111111111111111]
        
        with patch('load_messages.save_messages') as mock_save_messages:
            result = fetch_and_store_messages(
                client=mock_client,
                forbidden_channels=set(),
                config={},
                server_id="123456789012345678",
                server_name="Test Server",
                sleep_between_channels=False,
                channel_filter=channel_filter,
                hours_back=None
            )
            
            # Should only fetch from one channel
            assert mock_client.fetch_messages.call_count == 1
            assert result == 1
    
    @patch('load_messages.save_server')
    @patch('load_messages.save_channel')
    @patch('load_messages.get_last_message_id')
    def test_fetch_skips_forbidden_channels(self, mock_get_last, mock_save_channel, mock_save_server):
        """Test that forbidden channels are skipped."""
        mock_get_last.return_value = None
        
        mock_client = Mock()
        mock_client.get_channel_ids.return_value = [
            (111111111111111111, "general"),
            (222222222222222222, "forbidden-channel"),
            (333333333333333333, "announcements"),
        ]
        mock_client.fetch_messages.return_value = []
        
        # Mark channel 222222222222222222 as forbidden
        forbidden_channels = {222222222222222222}
        
        with patch('load_messages.save_messages'):
            fetch_and_store_messages(
                client=mock_client,
                forbidden_channels=forbidden_channels,
                config={},
                server_id="123456789012345678",
                server_name="Test Server",
                sleep_between_channels=False,
                channel_filter=None,
                hours_back=None
            )
            
            # Should only call fetch_messages twice (not 3 times)
            assert mock_client.fetch_messages.call_count == 2


@pytest.mark.unit
class TestTimeBasedFiltering:
    """Test time-based message filtering."""
    
    def test_get_recent_messages_within_timeframe(self, temp_db):
        """Test retrieving messages within specified hours."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        
        now = datetime.now()
        
        messages = [
            (server_id, channel_id, 1001, "Recent message", int((now - timedelta(hours=1)).timestamp())),
            (server_id, channel_id, 1002, "Old message", int((now - timedelta(hours=25)).timestamp())),
        ]
        
        save_messages(messages)
        
        # Get messages from last 24 hours
        records = get_recent_message_records(server_id, hours=24)
        
        # Should only get the recent message
        assert len(records) == 1
        assert records[0]["content"] == "Recent message"
    
    def test_get_recent_messages_all_time(self, temp_db):
        """Test retrieving all messages without time filter."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        
        now = datetime.now()
        
        messages = [
            (server_id, channel_id, 1001, "Recent", int((now - timedelta(hours=1)).timestamp())),
            (server_id, channel_id, 1002, "Old", int((now - timedelta(days=30)).timestamp())),
        ]
        
        save_messages(messages)
        
        # Get all messages
        records = get_recent_message_records(server_id, hours=None)
        
        # Should get both messages
        assert len(records) == 2


@pytest.mark.unit
class TestLastMessageIdTracking:
    """Test last message ID tracking for incremental fetching."""
    
    def test_get_last_message_id_with_messages(self, temp_db):
        """Test retrieving last message ID when messages exist."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        
        messages = [
            (server_id, channel_id, 1001, "Message 1", 1703030400),
            (server_id, channel_id, 1002, "Message 2", 1703030500),
            (server_id, channel_id, 1003, "Message 3", 1703030600),
        ]
        
        save_messages(messages)
        
        last_id = get_last_message_id(server_id, channel_id)
        
        # Should return the most recent message ID
        assert last_id == 1003
    
    def test_get_last_message_id_no_messages(self, temp_db):
        """Test retrieving last message ID when no messages exist."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        
        last_id = get_last_message_id(server_id, channel_id)
        
        # Should return None
        assert last_id is None
    
    def test_get_last_message_id_different_channels(self, temp_db):
        """Test that last message ID is tracked per channel."""
        server_id = 123456789012345678
        channel1_id = 111111111111111111
        channel2_id = 222222222222222222
        
        save_server(server_id, "Test Server")
        save_channel(channel1_id, server_id, "general")
        save_channel(channel2_id, server_id, "random")
        
        messages = [
            (server_id, channel1_id, 1001, "Channel 1 msg", 1703030400),
            (server_id, channel2_id, 2001, "Channel 2 msg", 1703030500),
        ]
        
        save_messages(messages)
        
        last_id_channel1 = get_last_message_id(server_id, channel1_id)
        last_id_channel2 = get_last_message_id(server_id, channel2_id)
        
        assert last_id_channel1 == 1001
        assert last_id_channel2 == 2001


@pytest.mark.unit
class TestDiscordClientInitialization:
    """Test Discord client initialization."""
    
    def test_initialize_client_with_valid_token(self, mock_environment):
        """Test client initialization with valid token in .env."""
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', create=True) as mock_open:
            
            mock_file = MagicMock()
            mock_file.__enter__.return_value = mock_file
            mock_file.__iter__.return_value = iter([
                'DISCORD_TOKEN=mock_token_12345\n',
                'OTHER_VAR=value\n'
            ])
            mock_open.return_value = mock_file
            
            with patch('load_messages.Discord') as mock_discord:
                client = initialize_discord_client(None)
                mock_discord.assert_called_once()
    
    def test_initialize_client_missing_env_file(self):
        """Test that missing .env file raises error."""
        with patch('os.path.exists', return_value=False):
            with pytest.raises(RuntimeError, match=".env file not found"):
                initialize_discord_client(None)
    
    def test_initialize_client_missing_token(self):
        """Test that missing token raises error."""
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', create=True) as mock_open:
            
            mock_file = MagicMock()
            mock_file.__enter__.return_value = mock_file
            mock_file.__iter__.return_value = iter(['OTHER_VAR=value\n'])
            mock_open.return_value = mock_file
            
            with pytest.raises(RuntimeError, match="DISCORD_TOKEN not found"):
                initialize_discord_client(None)
