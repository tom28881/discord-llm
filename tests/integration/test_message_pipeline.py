"""
Integration tests for the complete message processing pipeline.
"""
import pytest
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from lib.database import init_db, save_server, save_channel, save_messages, get_recent_messages
from load_messages import load_messages_once
from lib.discord_client import Discord
import lib.llm as llm
from lib.importance_detector import MessageImportanceDetector


@pytest.mark.integration
@pytest.mark.slow
class TestMessageProcessingPipeline:
    """Test the complete message processing pipeline."""
    
    @patch('requests.get')
    def test_end_to_end_message_flow(self, mock_get, temp_db, mock_environment, sample_patterns):
        """Test complete end-to-end message processing flow."""
        # Setup Discord API mocks
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        
        # Mock server list response
        servers_response = Mock()
        servers_response.json.return_value = [
            {"id": "123456789012345678", "name": "Test Server"}
        ]
        
        # Mock channels response
        channels_response = Mock()
        channels_response.json.return_value = [
            {"id": "111111111111111111", "name": "general", "type": 0},
            {"id": "222222222222222222", "name": "announcements", "type": 0}
        ]
        
        # Mock messages response
        messages_response = Mock()
        messages_response.json.return_value = [
            {
                "id": "1001",
                "content": "URGENT: Server maintenance needed immediately!",
                "timestamp": "2024-01-15T10:00:00.000Z"
            },
            {
                "id": "1002",
                "content": "New group buy for keyboards starting tomorrow",
                "timestamp": "2024-01-15T10:05:00.000Z"
            },
            {
                "id": "1003",
                "content": "Just saying hi to everyone",
                "timestamp": "2024-01-15T10:10:00.000Z"
            }
        ]
        
        # Configure mock responses based on URL
        def mock_get_side_effect(url, **kwargs):
            if "/users/@me/guilds" in url:
                return servers_response
            elif "/guilds/" in url and "/channels" in url:
                return channels_response
            elif "/channels/" in url and "/messages" in url:
                return messages_response
            else:
                return mock_response
        
        mock_get.side_effect = mock_get_side_effect
        
        # Step 1: Initialize Discord client
        client = Discord(token="mock_token")
        
        # Step 2: Get servers and channels
        servers = client.get_server_ids()
        assert len(servers) == 1
        assert servers[0] == (123456789012345678, "Test Server")
        
        server_id, server_name = servers[0]
        client.server_id = str(server_id)
        
        channels = client.get_channel_ids()
        assert len(channels) == 2
        
        # Step 3: Save server and channels to database
        save_server(server_id, server_name)
        for channel_id, channel_name in channels:
            save_channel(channel_id, server_id, channel_name)
        
        # Step 4: Fetch and save messages
        for channel_id, channel_name in channels:
            messages = client.fetch_messages(channel_id, limit=100)
            messages_to_save = [
                (server_id, channel_id, int(msg_id), content, timestamp)
                for msg_id, content, timestamp in messages
            ]
            save_messages(messages_to_save)
        
        # Step 5: Retrieve messages and analyze importance
        recent_messages = get_recent_messages(server_id, hours=24)
        assert len(recent_messages) == 3
        
        # Step 6: Test importance detection
        detector = MessageImportanceDetector(sample_patterns)
        
        important_messages = []
        for message in recent_messages:
            result = detector.detect_importance(message)
            if result.score >= 0.7:
                important_messages.append({
                    "content": message,
                    "score": result.score,
                    "patterns": result.detected_patterns
                })
        
        # Should identify the urgent message and group buy as important
        assert len(important_messages) >= 2
        assert any("urgent" in msg["patterns"] for msg in important_messages)
        assert any("group_buy" in msg["patterns"] for msg in important_messages)

    @patch('load_messages.initialize_discord_client')
    @patch('load_messages.load_config')
    @patch('load_messages.load_forbidden_channels')
    @patch('lib.llm.get_completion')
    def test_llm_integration_pipeline(self, mock_get_completion, mock_forbidden, mock_config, mock_init_client, temp_db, sample_patterns):
        """Test integration with LLM for message analysis."""

        server_id = 123456789012345678
        channel_id = 111111111111111111
        base_timestamp = int(datetime.now().timestamp())

        mock_client = MagicMock(spec=Discord)
        mock_client.get_server_ids.return_value = [(server_id, "Test Server")]
        mock_client.get_channel_ids.return_value = [(channel_id, "general")]
        mock_client.fetch_messages.return_value = [
            ("1001", "Can someone help me with the new deployment process?", base_timestamp),
        ]
        mock_init_client.return_value = mock_client
        mock_config.return_value = {"forbidden_channels": []}
        mock_forbidden.return_value = set()

        mock_get_completion.return_value = (
            "This message is asking for help with deployment process. It's moderately important as it relates to development workflow."
        )

        summary = load_messages_once(
            sleep_between_servers=False,
            sleep_between_channels=False
        )

        # Post-run assertions
        assert summary["messages_saved"] == 1
        mock_client.fetch_messages.assert_called_once_with(channel_id, None, 5000)
        recent_messages = get_recent_messages(server_id, hours=24)
        assert recent_messages == ["Can someone help me with the new deployment process?"]

        for message in recent_messages:
            analysis = llm.get_completion(f"Analyze this Discord message for importance: {message}")
            assert "deployment process" in analysis
            assert "moderately important" in analysis

        mock_get_completion.assert_called_once_with(
            "Analyze this Discord message for importance: Can someone help me with the new deployment process?"
        )

    @patch('requests.get')
    def test_error_handling_integration(self, mock_get, temp_db):
        """Test error handling throughout the pipeline."""
        # Test HTTP errors
        mock_get.side_effect = Exception("Network error")
        
        client = Discord(token="mock_token", server_id="123456789012345678")
        
        with pytest.raises(Exception):
            client.get_server_ids()
        
        # Test rate limiting
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.raise_for_status.side_effect = Exception("Rate limited")
        mock_get.side_effect = None
        mock_get.return_value = rate_limit_response
        
        with pytest.raises(Exception):
            client.fetch_messages(111111111111111111)

    def test_concurrent_processing(self, temp_db, mock_environment, sample_patterns):
        """Test concurrent message processing."""
        import threading
        import concurrent.futures
        
        # Setup test data
        server_id = 123456789012345678
        save_server(server_id, "Test Server")
        
        channels = [
            (111111111111111111, "general"),
            (222222222222222222, "random"),
            (333333333333333333, "announcements")
        ]
        
        for channel_id, channel_name in channels:
            save_channel(channel_id, server_id, channel_name)
        
        # Create importance detector
        detector = MessageImportanceDetector(sample_patterns)
        
        def process_channel_messages(channel_info, channel_idx):
            """Process messages for a single channel."""
            channel_id, channel_name = channel_info
            base_message_id = channel_idx * 1000
            
            # Generate test messages
            messages = [
                (server_id, channel_id, base_message_id + i, f"Test message {i} in {channel_name}", 
                 int(datetime.now().timestamp()) + i)
                for i in range(10)
            ]
            
            # Save messages
            save_messages(messages)
            
            # Analyze importance
            results = []
            for _, _, _, content, _ in messages:
                result = detector.detect_importance(content, channel_name)
                results.append(result)
            
            return len(results)
        
        # Process channels concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_channel_messages, channel, idx) 
                      for idx, channel in enumerate(channels)]
            results = [future.result() for future in futures]
        
        # Verify all channels were processed
        assert all(result == 10 for result in results)
        
        # Verify all messages were saved
        total_messages = []
        for channel_id, _ in channels:
            total_messages.extend(get_recent_messages(server_id, hours=24, channel_id=channel_id))
        assert len(total_messages) == 30  # 10 messages per channel × 3 channels


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseIntegration:
    """Test database integration scenarios."""
    
    def test_large_message_batch_processing(self, temp_db):
        """Test processing large batches of messages."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        # Setup
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        
        # Create large batch of messages
        batch_size = 1000
        messages = [
            (server_id, channel_id, i, f"Message {i}", 
             int(datetime.now().timestamp()) + i)
            for i in range(batch_size)
        ]
        
        # Process in batches
        batch_size_limit = 100
        for i in range(0, len(messages), batch_size_limit):
            batch = messages[i:i + batch_size_limit]
            save_messages(batch)
        
        # Verify all messages were saved
        recent_messages = get_recent_messages(server_id, hours=24)
        assert len(recent_messages) == batch_size

    def test_database_consistency_under_load(self, temp_db):
        """Test database consistency under concurrent load."""
        import threading
        import time
        
        server_id = 123456789012345678
        save_server(server_id, "Test Server")
        
        # Create multiple channels
        channels = []
        for i in range(5):
            channel_id = 111111111111111111 + i
            save_channel(channel_id, server_id, f"channel-{i}")
            channels.append(channel_id)
        
        def write_messages(channel_id, start_id, count):
            """Write messages to a specific channel."""
            messages = [
                (server_id, channel_id, start_id + i, f"Message {start_id + i}", 
                 int(time.time()) + i)
                for i in range(count)
            ]
            save_messages(messages)
        
        # Start multiple threads writing to different channels
        threads = []
        for i, channel_id in enumerate(channels):
            thread = threading.Thread(
                target=write_messages, 
                args=(channel_id, i * 100, 50)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify data consistency
        total_messages = get_recent_messages(server_id, hours=24)
        assert len(total_messages) == 250  # 5 channels × 50 messages each


@pytest.mark.integration
@pytest.mark.api
class TestExternalAPIIntegration:
    """Test integration with external APIs."""
    
    @patch('requests.get')
    def test_discord_api_rate_limiting(self, mock_get):
        """Test handling of Discord API rate limits."""
        # First request succeeds
        success_response = Mock()
        success_response.raise_for_status.return_value = None
        success_response.json.return_value = [{"id": "123", "name": "test"}]
        
        # Second request is rate limited
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {"Retry-After": "1"}
        rate_limit_response.raise_for_status.side_effect = Exception("Rate limited")
        
        mock_get.side_effect = [success_response, rate_limit_response]
        
        client = Discord(token="mock_token", server_id="123456789012345678")
        
        # First request should succeed
        result1 = client.get_channel_ids()
        assert len(result1) == 1
        
        # Second request should raise exception
        with pytest.raises(Exception, match="Rate limited"):
            client.get_channel_ids()

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_llm_api_error_recovery(self, mock_model_class, mock_configure, mock_environment):
        """Test LLM API error handling and recovery."""
        mock_model = Mock()
        
        # First call fails
        mock_model.generate_content.side_effect = [
            Exception("API temporarily unavailable"),
            Mock(text="Recovery successful")
        ]
        mock_model_class.return_value = mock_model
        
        # First attempt should return empty string
        result1 = llm.get_completion("Test prompt")
        assert result1 == ""
        
        # Second attempt should succeed (in real scenario, might implement retry logic)
        result2 = llm.get_completion("Test prompt")
        assert result2 == "Recovery successful"

    def test_environment_configuration_integration(self, mock_environment):
        """Test integration with environment configuration."""
        # Test Discord client initialization
        client = Discord(token=mock_environment["DISCORD_TOKEN"])
        assert client.headers["Authorization"] == mock_environment["DISCORD_TOKEN"]
        
        # Test LLM configuration
        with patch('google.generativeai.configure') as mock_configure:
            llm.get_completion("Test")
            mock_configure.assert_called_with(api_key=mock_environment["GOOGLE_API_KEY"])


@pytest.mark.integration
@pytest.mark.performance
class TestIntegrationPerformance:
    """Performance tests for integrated systems."""
    
    @patch('requests.get')
    def test_message_processing_throughput(self, mock_get, temp_db, sample_patterns, performance_baseline):
        """Test message processing throughput."""
        import time
        
        # Setup mocks
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {
                "id": f"{i:04d}",
                "content": f"Test message {i} urgent group buy event",
                "timestamp": "2024-01-15T10:00:00.000Z"
            }
            for i in range(100)
        ]
        mock_get.return_value = mock_response
        
        # Setup database
        server_id = 123456789012345678
        channel_id = 111111111111111111
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        
        # Setup components
        client = Discord(token="mock_token", server_id=str(server_id))
        detector = MessageImportanceDetector(sample_patterns)
        
        # Measure processing time
        start_time = time.time()
        
        # Fetch messages
        messages = client.fetch_messages(channel_id, limit=100)
        
        # Save to database
        messages_to_save = [
            (server_id, channel_id, int(msg_id), content, timestamp)
            for msg_id, content, timestamp in messages
        ]
        save_messages(messages_to_save)
        
        # Analyze importance
        for msg_id, content, timestamp in messages:
            result = detector.detect_importance(content)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        messages_per_second = len(messages) / total_time
        
        # Should meet performance baseline
        assert messages_per_second >= performance_baseline["message_import_rate"]

    def test_memory_usage_under_load(self, temp_db, sample_patterns):
        """Test memory usage under sustained load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Setup
        server_id = 123456789012345678
        save_server(server_id, "Test Server")
        detector = MessageImportanceDetector(sample_patterns)
        
        # Process many messages
        for batch in range(10):
            channel_id = 111111111111111111 + batch
            save_channel(channel_id, server_id, f"channel-{batch}")
            
            messages = [
                (server_id, channel_id, batch * 1000 + i, 
                 f"Message {i} urgent announcement group buy deadline",
                 int(time.time()) + i)
                for i in range(100)
            ]
            
            save_messages(messages)
            
            # Analyze importance for each message
            for _, _, _, content, _ in messages:
                detector.detect_importance(content)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB increase