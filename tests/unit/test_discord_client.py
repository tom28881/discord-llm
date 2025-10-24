"""
Unit tests for Discord client functionality.
"""
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from lib.discord_client import Discord


@pytest.mark.unit
@pytest.mark.discord
class TestDiscordClient:
    """Test Discord client operations."""

    def test_discord_client_initialization(self, mock_environment):
        """Test Discord client initialization."""
        token = "mock_token_12345"
        server_id = "123456789012345678"
        
        client = Discord(token=token, server_id=server_id)
        
        assert client.server_id == server_id
        assert "Authorization" in client.headers
        assert client.headers["Authorization"] == token

    def test_discord_client_invalid_token(self):
        """Test Discord client with invalid token."""
        with pytest.raises(ValueError, match="Valid Discord token must be provided"):
            Discord(token="", server_id="123456789012345678")
        
        with pytest.raises(ValueError, match="Valid Discord token must be provided"):
            Discord(token="short", server_id="123456789012345678")

    @patch('requests.get')
    def test_get_channel_ids_success(self, mock_get):
        """Test successful channel ID retrieval."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = [
            {"id": "111111111111111111", "name": "general", "type": 0},
            {"id": "222222222222222222", "name": "random", "type": 0},
            {"id": "333333333333333333", "name": "voice", "type": 2},  # Voice channel (filtered out)
            {"id": "444444444444444444", "name": "announcements", "type": 5},
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = Discord(token="mock_token", server_id="123456789012345678")
        channels = client.get_channel_ids()
        
        # Should only return text channels (type 0 and 5)
        assert len(channels) == 3
        assert (111111111111111111, "general") in channels
        assert (222222222222222222, "random") in channels
        assert (444444444444444444, "announcements") in channels

    @patch('requests.get')
    def test_get_channel_ids_http_error(self, mock_get):
        """Test channel ID retrieval with HTTP error."""
        mock_get.side_effect = requests.HTTPError("401 Unauthorized")
        
        client = Discord(token="invalid_token", server_id="123456789012345678")
        
        with pytest.raises(requests.HTTPError):
            client.get_channel_ids()

    @patch('requests.get')
    def test_get_server_ids_success(self, mock_get):
        """Test successful server ID retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"id": "123456789012345678", "name": "Test Server 1"},
            {"id": "987654321098765432", "name": "Test Server 2"},
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = Discord(token="mock_token")
        servers = client.get_server_ids()
        
        assert len(servers) == 2
        assert (123456789012345678, "Test Server 1") in servers
        assert (987654321098765432, "Test Server 2") in servers

    @patch('requests.get')
    def test_fetch_messages_success(self, mock_get):
        """Test successful message fetching."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "id": "1001",
                "content": "Hello world!",
                "timestamp": "2024-01-15T10:00:00.000Z"
            },
            {
                "id": "1002", 
                "content": "How are you?",
                "timestamp": "2024-01-15T10:05:00.000Z"
            }
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = Discord(token="mock_token", server_id="123456789012345678")
        messages = client.fetch_messages(channel_id=111111111111111111, limit=100)
        
        assert len(messages) == 2
        assert messages[0][0] == "1001"  # Message ID
        assert messages[0][1] == "Hello world!"  # Content
        assert isinstance(messages[0][2], int)  # Timestamp

    @patch('requests.get')
    def test_fetch_messages_with_since_id(self, mock_get):
        """Test fetching messages with since_message_id parameter."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "id": "1003",
                "content": "New message",
                "timestamp": "2024-01-15T11:00:00.000Z"
            }
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = Discord(token="mock_token", server_id="123456789012345678")
        messages = client.fetch_messages(
            channel_id=111111111111111111,
            since_message_id="1002",
            limit=100
        )
        
        # Verify the request was made with correct parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert "after" in kwargs["params"]
        assert kwargs["params"]["after"] == "1002"

    @patch('requests.get')
    def test_fetch_messages_empty_response(self, mock_get):
        """Test fetching messages with empty response."""
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = Discord(token="mock_token", server_id="123456789012345678")
        messages = client.fetch_messages(channel_id=111111111111111111, limit=100)
        
        assert len(messages) == 0

    @patch('requests.get')
    def test_fetch_messages_pagination(self, mock_get):
        """Test message fetching with pagination."""
        # First batch - full 100 messages
        first_batch = [
            {
                "id": f"{i:04d}",
                "content": f"Message {i}",
                "timestamp": "2024-01-15T10:00:00.000Z"
            }
            for i in range(1001, 1101)  # 100 messages
        ]
        
        # Second batch - partial (less than 100)
        second_batch = [
            {
                "id": "1101",
                "content": "Last message",
                "timestamp": "2024-01-15T10:00:00.000Z"
            }
        ]
        
        mock_response_1 = Mock()
        mock_response_1.json.return_value = first_batch
        mock_response_1.raise_for_status.return_value = None
        
        mock_response_2 = Mock()
        mock_response_2.json.return_value = second_batch
        mock_response_2.raise_for_status.return_value = None
        
        mock_get.side_effect = [mock_response_1, mock_response_2]
        
        client = Discord(token="mock_token", server_id="123456789012345678")
        messages = client.fetch_messages(channel_id=111111111111111111, limit=150)
        
        # Should fetch both batches
        assert len(messages) == 101  # 100 + 1
        assert mock_get.call_count == 2

    @patch('requests.get')
    def test_fetch_messages_rate_limiting(self, mock_get):
        """Test rate limiting handling."""
        # Mock rate limit response
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
        
        mock_get.return_value = rate_limit_response
        
        client = Discord(token="mock_token", server_id="123456789012345678")
        
        with pytest.raises(requests.HTTPError, match="429 Too Many Requests"):
            client.fetch_messages(channel_id=111111111111111111, limit=100)

    @patch('requests.get')
    def test_fetch_messages_forbidden(self, mock_get):
        """Test handling forbidden channel access."""
        forbidden_response = Mock()
        forbidden_response.status_code = 403
        forbidden_response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        
        mock_get.return_value = forbidden_response
        
        client = Discord(token="mock_token", server_id="123456789012345678")
        
        with pytest.raises(requests.HTTPError, match="403 Forbidden"):
            client.fetch_messages(channel_id=111111111111111111, limit=100)

    def test_channel_type_filtering(self):
        """Test that only allowed channel types are processed."""
        client = Discord(token="mock_token", server_id="123456789012345678")
        
        # Test allowed channel types
        assert 0 in client.ALLOWED_CHANNEL_TYPES  # Text channel
        assert 5 in client.ALLOWED_CHANNEL_TYPES  # News channel
        assert 13 in client.ALLOWED_CHANNEL_TYPES  # Stage channel
        
        # Test disallowed channel types
        assert 2 not in client.ALLOWED_CHANNEL_TYPES  # Voice channel
        assert 4 not in client.ALLOWED_CHANNEL_TYPES  # Category channel

    @patch('requests.get')
    def test_request_headers(self, mock_get):
        """Test that proper headers are sent with requests."""
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        token = "mock_token_12345"
        client = Discord(token=token, server_id="123456789012345678")
        client.get_channel_ids()
        
        # Verify headers were set correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        headers = kwargs["headers"]
        
        assert headers["Authorization"] == token
        assert "User-Agent" in headers
        assert "Accept" in headers

    def test_timestamp_parsing(self):
        """Test Discord timestamp parsing."""
        client = Discord(token="mock_token", server_id="123456789012345678")
        
        # Test ISO format timestamp
        iso_timestamp = "2024-01-15T10:00:00.000Z"
        expected_unix = int(datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00")).timestamp())
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = [
                {
                    "id": "1001",
                    "content": "Test message",
                    "timestamp": iso_timestamp
                }
            ]
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            messages = client.fetch_messages(channel_id=111111111111111111, limit=1)
            
            assert len(messages) == 1
            assert messages[0][2] == expected_unix


@pytest.mark.unit
@pytest.mark.discord
@pytest.mark.performance
class TestDiscordClientPerformance:
    """Test Discord client performance characteristics."""

    @patch('requests.get')
    def test_message_fetching_performance(self, mock_get, performance_baseline):
        """Test message fetching performance."""
        import time
        
        # Mock large batch of messages
        messages = [
            {
                "id": f"{i:04d}",
                "content": f"Message {i}",
                "timestamp": "2024-01-15T10:00:00.000Z"
            }
            for i in range(1000)
        ]
        
        mock_response = Mock()
        mock_response.json.return_value = messages
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        client = Discord(token="mock_token", server_id="123456789012345678")
        
        start_time = time.time()
        result = client.fetch_messages(channel_id=111111111111111111, limit=1000)
        end_time = time.time()
        
        processing_time = end_time - start_time
        messages_per_second = len(result) / processing_time
        
        # Should meet performance baseline
        assert messages_per_second >= performance_baseline["message_import_rate"]