"""
Performance benchmarks and load testing for Discord monitoring system.
"""
import pytest
import time
import threading
import concurrent.futures
import psutil
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sqlite3

from lib.database import save_messages, get_recent_messages, save_server, save_channel
from lib.discord_client import Discord
from lib.importance_detector import MessageImportanceDetector
from lib.llm import get_completion


@pytest.mark.performance
@pytest.mark.slow
class TestDatabasePerformance:
    """Performance tests for database operations."""
    
    @pytest.mark.benchmark
    def test_bulk_message_insert_performance(self, temp_db, benchmark):
        """Benchmark bulk message insertion performance."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        
        # Create test messages
        messages = [
            (server_id, channel_id, i, f"Performance test message {i}", 
             int(time.time()) + i)
            for i in range(1000)
        ]
        
        # Benchmark the bulk insert
        result = benchmark(save_messages, messages)
        
        # Verify messages were saved
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 1000

    def test_concurrent_database_writes(self, temp_db, performance_baseline):
        """Test concurrent database write performance."""
        server_id = 123456789012345678
        save_server(server_id, "Test Server")
        
        # Create multiple channels
        channels = []
        for i in range(10):
            channel_id = 111111111111111111 + i
            save_channel(channel_id, server_id, f"channel-{i}")
            channels.append(channel_id)
        
        def write_to_channel(channel_id, start_id):
            """Write messages to a specific channel."""
            messages = [
                (server_id, channel_id, start_id + i, f"Concurrent message {start_id + i}",
                 int(time.time()) + i)
                for i in range(100)
            ]
            start_time = time.time()
            save_messages(messages)
            end_time = time.time()
            return end_time - start_time
        
        # Run concurrent writes
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(write_to_channel, channel_id, i * 1000)
                for i, channel_id in enumerate(channels)
            ]
            write_times = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        total_messages = 10 * 100  # 10 channels × 100 messages
        messages_per_second = total_messages / total_time
        avg_write_time = sum(write_times) / len(write_times)
        
        assert messages_per_second >= performance_baseline["message_import_rate"]
        assert avg_write_time <= performance_baseline["database_write_time"] * 100  # 100 messages

    def test_query_performance_with_large_dataset(self, temp_db, performance_baseline):
        """Test query performance with large dataset."""
        server_id = 123456789012345678
        channel_id = 111111111111111111
        
        save_server(server_id, "Test Server")
        save_channel(channel_id, server_id, "general")
        
        # Insert large dataset
        batch_size = 1000
        for batch in range(10):  # 10,000 total messages
            messages = [
                (server_id, channel_id, batch * batch_size + i,
                 f"Large dataset message {batch * batch_size + i}",
                 int(time.time()) - (batch * batch_size + i))  # Spread over time
                for i in range(batch_size)
            ]
            save_messages(messages)
        
        # Test query performance
        start_time = time.time()
        recent_messages = get_recent_messages(server_id, hours=24)
        query_time = time.time() - start_time
        
        assert query_time <= performance_baseline["query_response_time"]
        assert len(recent_messages) > 0

    def test_database_memory_usage(self, temp_db):
        """Test database memory usage under load."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        server_id = 123456789012345678
        save_server(server_id, "Test Server")
        
        # Create many channels and messages
        for channel_batch in range(20):
            channel_id = 111111111111111111 + channel_batch
            save_channel(channel_id, server_id, f"channel-{channel_batch}")
            
            # Insert messages in batches
            for msg_batch in range(10):
                messages = [
                    (server_id, channel_id, 
                     channel_batch * 10000 + msg_batch * 100 + i,
                     f"Memory test message {i}" * 10,  # Longer content
                     int(time.time()) + i)
                    for i in range(100)
                ]
                save_messages(messages)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable for 20,000 messages
        assert memory_increase < 200  # Less than 200MB


@pytest.mark.performance
@pytest.mark.slow
class TestImportanceDetectionPerformance:
    """Performance tests for importance detection."""
    
    @pytest.mark.benchmark
    def test_importance_detection_speed(self, sample_patterns, benchmark):
        """Benchmark importance detection speed."""
        detector = MessageImportanceDetector(sample_patterns)
        
        test_message = "URGENT: Group buy deadline tomorrow, server maintenance needed immediately!"
        
        # Benchmark the detection
        result = benchmark(detector.detect_importance, test_message)
        
        assert result.score > 0.8
        assert len(result.detected_patterns) > 0

    def test_batch_importance_analysis(self, sample_patterns, performance_baseline):
        """Test batch importance analysis performance."""
        detector = MessageImportanceDetector(sample_patterns)
        
        # Create diverse test messages
        test_messages = [
            "URGENT: System failure detected, immediate assistance required!",
            "Group buy for premium keyboards starting tomorrow, limited slots",
            "Meeting scheduled for next Tuesday at 3 PM, please confirm attendance",
            "New announcement: Platform update will be released next week",
            "Can someone help me with the deployment configuration?",
            "Good morning everyone, hope you have a great day!",
            "Thanks for the quick response, really appreciate it",
            "What do you think about the new design proposals?",
            "Breaking: Major security update available for download",
            "Reminder: Project deadline is tomorrow, please submit on time"
        ] * 100  # 1000 messages total
        
        start_time = time.time()
        
        results = []
        for message in test_messages:
            result = detector.detect_importance(message)
            results.append(result)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        messages_per_second = len(test_messages) / total_time
        avg_time_per_message = total_time / len(test_messages)
        
        assert avg_time_per_message <= performance_baseline["importance_scoring_time"]
        assert messages_per_second >= 50  # Should process at least 50 messages/second
        assert len(results) == len(test_messages)

    def test_concurrent_importance_analysis(self, sample_patterns):
        """Test concurrent importance analysis performance."""
        detector = MessageImportanceDetector(sample_patterns)
        
        test_messages = [
            f"Test message {i} with urgent group buy announcement event"
            for i in range(500)
        ]
        
        def analyze_batch(messages_batch):
            """Analyze a batch of messages."""
            results = []
            for message in messages_batch:
                result = detector.detect_importance(message)
                results.append(result)
            return results
        
        # Split messages into batches for concurrent processing
        batch_size = 50
        message_batches = [
            test_messages[i:i + batch_size] 
            for i in range(0, len(test_messages), batch_size)
        ]
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(analyze_batch, batch) for batch in message_batches]
            all_results = []
            for future in futures:
                all_results.extend(future.result())
        
        end_time = time.time()
        
        total_time = end_time - start_time
        messages_per_second = len(test_messages) / total_time
        
        assert messages_per_second >= 100  # Concurrent processing should be faster
        assert len(all_results) == len(test_messages)

    def test_importance_detector_memory_efficiency(self, sample_patterns):
        """Test memory efficiency of importance detector."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple detectors (simulating multiple channels/servers)
        detectors = []
        for i in range(10):
            detector = MessageImportanceDetector(sample_patterns)
            detectors.append(detector)
        
        memory_after_creation = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many messages with each detector
        for i, detector in enumerate(detectors):
            for j in range(100):
                message = f"Detector {i} message {j} urgent group buy event announcement"
                detector.detect_importance(message)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        creation_memory_increase = memory_after_creation - initial_memory
        processing_memory_increase = final_memory - memory_after_creation
        
        # Memory usage should be reasonable
        assert creation_memory_increase < 50  # Creating detectors shouldn't use much memory
        assert processing_memory_increase < 30  # Processing shouldn't accumulate memory


@pytest.mark.performance
@pytest.mark.api
class TestAPIPerformance:
    """Performance tests for external API interactions."""
    
    @patch('requests.get')
    def test_discord_api_throughput(self, mock_get, performance_baseline):
        """Test Discord API throughput simulation."""
        # Mock API responses with realistic delays
        def mock_response_with_delay(*args, **kwargs):
            time.sleep(0.1)  # Simulate network latency
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = [
                {
                    "id": f"{i:04d}",
                    "content": f"Message {i}",
                    "timestamp": "2024-01-15T10:00:00.000Z"
                }
                for i in range(100)
            ]
            return response
        
        mock_get.side_effect = mock_response_with_delay
        
        client = Discord(token="mock_token", server_id="123456789012345678")
        
        # Measure API call performance
        start_time = time.time()
        
        total_messages = 0
        for _ in range(10):  # 10 API calls
            messages = client.fetch_messages(111111111111111111, limit=100)
            total_messages += len(messages)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        messages_per_second = total_messages / total_time
        
        # Account for simulated network delay
        assert messages_per_second >= 80  # Should handle at least 80 messages/second

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_llm_api_performance(self, mock_model_class, mock_configure, 
                                mock_environment, performance_baseline):
        """Test LLM API performance."""
        mock_model = Mock()
        
        def mock_generate_with_delay(prompt):
            time.sleep(0.5)  # Simulate LLM processing time
            response = Mock()
            response.text = f"Analysis of: {prompt[:50]}..."
            return response
        
        mock_model.generate_content.side_effect = mock_generate_with_delay
        mock_model_class.return_value = mock_model
        
        # Test multiple LLM calls
        prompts = [
            f"Analyze this Discord message {i}: Test message content"
            for i in range(10)
        ]
        
        start_time = time.time()
        
        results = []
        for prompt in prompts:
            result = get_completion(prompt)
            results.append(result)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_call = total_time / len(prompts)
        
        assert avg_time_per_call <= performance_baseline["llm_response_time"]
        assert len(results) == len(prompts)
        assert all(result for result in results)


@pytest.mark.performance
@pytest.mark.stress
class TestStressTests:
    """Stress tests for system limits."""
    
    def test_high_message_volume_stress(self, temp_db):
        """Stress test with high message volume."""
        server_id = 123456789012345678
        save_server(server_id, "Stress Test Server")
        
        # Create many channels
        channels = []
        for i in range(50):
            channel_id = 111111111111111111 + i
            save_channel(channel_id, server_id, f"stress-channel-{i}")
            channels.append(channel_id)
        
        # Generate high volume of messages
        total_messages = 0
        start_time = time.time()
        
        for channel_id in channels:
            messages = [
                (server_id, channel_id, total_messages + i,
                 f"Stress test message {total_messages + i} with various keywords urgent group buy",
                 int(time.time()) + i)
                for i in range(100)
            ]
            save_messages(messages)
            total_messages += len(messages)
        
        end_time = time.time()
        
        processing_time = end_time - start_time
        messages_per_second = total_messages / processing_time
        
        # Should handle high volume efficiently
        assert messages_per_second >= 500
        assert total_messages == 5000  # 50 channels × 100 messages

    def test_memory_stress_under_load(self, temp_db, sample_patterns):
        """Memory stress test under sustained load."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        server_id = 123456789012345678
        save_server(server_id, "Memory Stress Server")
        
        detector = MessageImportanceDetector(sample_patterns)
        
        # Sustained processing load
        for batch in range(100):
            channel_id = 111111111111111111 + batch
            save_channel(channel_id, server_id, f"memory-test-{batch}")
            
            # Process messages with importance detection
            messages = []
            for i in range(50):
                message_content = f"Batch {batch} message {i} urgent group buy event announcement"
                
                # Analyze importance (memory-intensive operation)
                result = detector.detect_importance(message_content)
                
                messages.append((
                    server_id, channel_id, batch * 50 + i,
                    message_content, int(time.time()) + i
                ))
            
            save_messages(messages)
            
            # Check memory usage periodically
            if batch % 20 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Memory shouldn't grow excessively
                assert memory_increase < 300, f"Memory usage too high: {memory_increase}MB"
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        # Final memory check
        assert total_memory_increase < 400  # Total increase should be reasonable

    def test_concurrent_user_simulation(self, temp_db, sample_patterns):
        """Simulate multiple concurrent users/processes."""
        server_id = 123456789012345678
        save_server(server_id, "Concurrent Test Server")
        
        detector = MessageImportanceDetector(sample_patterns)
        
        def simulate_user_activity(user_id):
            """Simulate a single user's activity."""
            channel_id = 111111111111111111 + user_id
            save_channel(channel_id, server_id, f"user-{user_id}-channel")
            
            # User generates messages
            messages = []
            analysis_results = []
            
            for i in range(20):
                content = f"User {user_id} message {i} urgent announcement group buy"
                
                # Analyze importance
                result = detector.detect_importance(content)
                analysis_results.append(result)
                
                messages.append((
                    server_id, channel_id, user_id * 1000 + i,
                    content, int(time.time()) + i
                ))
            
            # Save messages
            save_messages(messages)
            
            # Query recent messages
            recent = get_recent_messages(server_id, hours=1, channel_id=channel_id)
            
            return len(recent), len(analysis_results)
        
        # Simulate 20 concurrent users
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(simulate_user_activity, user_id)
                for user_id in range(20)
            ]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Verify all users completed successfully
        assert len(results) == 20
        assert all(saved == analyzed == 20 for saved, analyzed in results)
        
        # Performance should be reasonable even with concurrent load
        assert total_time < 30  # Should complete within 30 seconds


@pytest.mark.performance
@pytest.mark.benchmark
class TestSystemBenchmarks:
    """System-wide performance benchmarks."""
    
    def test_end_to_end_pipeline_benchmark(self, temp_db, sample_patterns, 
                                         performance_baseline, benchmark):
        """Benchmark complete end-to-end pipeline."""
        def pipeline_operation():
            server_id = 123456789012345678
            channel_id = 111111111111111111
            
            # Setup
            save_server(server_id, "Benchmark Server")
            save_channel(channel_id, server_id, "benchmark")
            
            # Create messages
            messages = [
                (server_id, channel_id, i,
                 f"Benchmark message {i} urgent group buy announcement",
                 int(time.time()) + i)
                for i in range(100)
            ]
            
            # Save messages
            save_messages(messages)
            
            # Analyze importance
            detector = MessageImportanceDetector(sample_patterns)
            important_count = 0
            
            for _, _, _, content, _ in messages:
                result = detector.detect_importance(content)
                if result.score >= 0.7:
                    important_count += 1
            
            # Query recent messages
            recent = get_recent_messages(server_id, hours=24)
            
            return len(recent), important_count
        
        # Benchmark the complete pipeline
        result = benchmark(pipeline_operation)
        
        recent_count, important_count = result
        assert recent_count == 100
        assert important_count > 0

    def test_sustained_throughput_benchmark(self, temp_db, sample_patterns):
        """Benchmark sustained throughput over time."""
        server_id = 123456789012345678
        save_server(server_id, "Throughput Server")
        
        detector = MessageImportanceDetector(sample_patterns)
        
        throughput_measurements = []
        
        # Measure throughput in intervals
        for interval in range(10):
            channel_id = 111111111111111111 + interval
            save_channel(channel_id, server_id, f"throughput-{interval}")
            
            start_time = time.time()
            
            # Process batch of messages
            messages = [
                (server_id, channel_id, interval * 100 + i,
                 f"Interval {interval} message {i} urgent group buy",
                 int(time.time()) + i)
                for i in range(100)
            ]
            
            save_messages(messages)
            
            # Analyze each message
            for _, _, _, content, _ in messages:
                detector.detect_importance(content)
            
            end_time = time.time()
            interval_time = end_time - start_time
            interval_throughput = len(messages) / interval_time
            
            throughput_measurements.append(interval_throughput)
        
        # Verify consistent throughput
        avg_throughput = sum(throughput_measurements) / len(throughput_measurements)
        min_throughput = min(throughput_measurements)
        max_throughput = max(throughput_measurements)
        
        # Throughput should be consistent (not degrade over time)
        throughput_variation = (max_throughput - min_throughput) / avg_throughput
        
        assert avg_throughput >= 50  # Average throughput
        assert throughput_variation < 0.5  # Less than 50% variation
        assert min_throughput >= 30  # Minimum acceptable throughput