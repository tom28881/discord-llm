"""
ML model tests for message importance detection.
"""
import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
import tempfile
import os

from lib.importance_detector import MessageImportanceDetector, ImportanceResult, create_importance_detector


@pytest.mark.ml
@pytest.mark.unit
class TestMessageImportanceDetector:
    """Test message importance detection ML functionality."""
    
    def test_detector_initialization(self, sample_patterns):
        """Test detector initialization with custom patterns."""
        detector = MessageImportanceDetector(sample_patterns)
        assert len(detector.patterns) == len(sample_patterns)
        assert "group_buy" in detector.patterns
        assert detector.keyword_weights["urgent"] > 0.8

    def test_detector_default_patterns(self):
        """Test detector with default patterns."""
        detector = MessageImportanceDetector()
        assert len(detector.patterns) > 0
        assert "urgent" in detector.patterns
        assert "group_buy" in detector.patterns
        assert "event" in detector.patterns

    def test_urgent_message_detection(self, sample_patterns):
        """Test detection of urgent messages."""
        detector = MessageImportanceDetector(sample_patterns)
        
        urgent_messages = [
            "URGENT: Server is down, need immediate help!",
            "Emergency maintenance required ASAP",
            "Critical issue with payment system, help needed",
            "Server offline, please fix immediately"
        ]
        
        for message in urgent_messages:
            result = detector.detect_importance(message)
            assert result.score >= 0.8, f"Failed for message: {message}"
            assert "urgent" in result.detected_patterns
            assert result.confidence > 0.6

    def test_group_buy_detection(self, sample_patterns):
        """Test detection of group buy messages."""
        detector = MessageImportanceDetector(sample_patterns)
        
        group_buy_messages = [
            "New group buy for mechanical keyboards starting tomorrow!",
            "IC for custom keycaps, deadline next week",
            "GB for switches is live, $50 shipping",
            "Interest check: Premium keyboard case"
        ]
        
        for message in group_buy_messages:
            result = detector.detect_importance(message)
            assert result.score >= 0.7, f"Failed for message: {message}"
            assert "group_buy" in result.detected_patterns

    def test_event_detection(self, sample_patterns):
        """Test detection of event messages."""
        detector = MessageImportanceDetector(sample_patterns)
        
        event_messages = [
            "Meeting tomorrow at 3 PM, don't forget!",
            "Workshop scheduled for next week",
            "Deadline reminder: project due today",
            "Conference call in 30 minutes"
        ]
        
        for message in event_messages:
            result = detector.detect_importance(message)
            assert result.score >= 0.6, f"Failed for message: {message}"
            assert "event" in result.detected_patterns

    def test_low_importance_messages(self, sample_patterns):
        """Test detection of low importance messages."""
        detector = MessageImportanceDetector(sample_patterns)
        
        low_importance_messages = [
            "Just saying hi to everyone",
            "How's the weather today?",
            "lol that's funny",
            "Good morning!",
            "Thanks for sharing"
        ]
        
        for message in low_importance_messages:
            result = detector.detect_importance(message)
            assert result.score <= 0.5, f"Failed for message: {message}"

    def test_empty_message_handling(self, sample_patterns):
        """Test handling of empty or invalid messages."""
        detector = MessageImportanceDetector(sample_patterns)
        
        invalid_messages = ["", "   ", None]
        
        for message in invalid_messages:
            if message is None:
                continue
            result = detector.detect_importance(message)
            assert result.score == 0.0
            assert len(result.detected_patterns) == 0

    def test_channel_context_modifiers(self, sample_patterns):
        """Test channel-based importance modifiers."""
        detector = MessageImportanceDetector(sample_patterns)
        
        message = "New announcement for everyone"
        
        # High importance channel
        result_important = detector.detect_importance(message, channel_name="announcements")
        
        # Low importance channel
        result_random = detector.detect_importance(message, channel_name="random")
        
        # Medium importance channel
        result_general = detector.detect_importance(message, channel_name="general")
        
        assert result_important.score > result_general.score
        assert result_general.score > result_random.score

    def test_time_context_modifiers(self, sample_patterns):
        """Test time-based importance modifiers."""
        detector = MessageImportanceDetector(sample_patterns)
        
        message = "Important update for everyone"
        now = datetime.now()
        
        # Recent message (1 hour ago)
        recent_time = now - timedelta(hours=1)
        result_recent = detector.detect_importance(message, timestamp=recent_time)
        
        # Old message (1 week ago)
        old_time = now - timedelta(weeks=1)
        result_old = detector.detect_importance(message, timestamp=old_time)
        
        # Very recent message (30 minutes ago)
        very_recent_time = now - timedelta(minutes=30)
        result_very_recent = detector.detect_importance(message, timestamp=very_recent_time)
        
        assert result_very_recent.score >= result_recent.score
        assert result_recent.score > result_old.score

    def test_context_boost_features(self, sample_patterns):
        """Test context boost features."""
        detector = MessageImportanceDetector(sample_patterns)
        
        # Test all caps boost
        caps_message = "URGENT ANNOUNCEMENT FOR EVERYONE"
        normal_message = "urgent announcement for everyone"
        
        caps_result = detector.detect_importance(caps_message)
        normal_result = detector.detect_importance(normal_message)
        
        assert caps_result.score > normal_result.score
        
        # Test @everyone boost
        mention_message = "Update for @everyone please read"
        no_mention_message = "Update please read"
        
        mention_result = detector.detect_importance(mention_message)
        no_mention_result = detector.detect_importance(no_mention_message)
        
        assert mention_result.score > no_mention_result.score
        
        # Test URL boost
        url_message = "Check this important link: https://example.com"
        no_url_message = "Check this important link"
        
        url_result = detector.detect_importance(url_message)
        no_url_result = detector.detect_importance(no_url_message)
        
        assert url_result.score > no_url_result.score

    def test_multiple_pattern_detection(self, sample_patterns):
        """Test detection of multiple patterns in one message."""
        detector = MessageImportanceDetector(sample_patterns)
        
        complex_message = "URGENT: Group buy deadline tomorrow! Meeting at 3 PM to discuss."
        
        result = detector.detect_importance(complex_message)
        
        # Should detect multiple patterns
        assert len(result.detected_patterns) >= 2
        assert "urgent" in result.detected_patterns
        assert "group_buy" in result.detected_patterns or "event" in result.detected_patterns
        
        # Should have high confidence due to multiple matches
        assert result.confidence > 0.7
        assert result.score > 0.8

    def test_confidence_calculation(self, sample_patterns):
        """Test confidence score calculation."""
        detector = MessageImportanceDetector(sample_patterns)
        
        # High confidence: long message with clear patterns
        high_conf_message = "URGENT: Emergency server maintenance scheduled for tomorrow at 3 PM. All users please save your work immediately!"
        high_conf_result = detector.detect_importance(high_conf_message)
        
        # Low confidence: short, unclear message
        low_conf_message = "hmm"
        low_conf_result = detector.detect_importance(low_conf_message)
        
        assert high_conf_result.confidence > low_conf_result.confidence
        assert high_conf_result.confidence > 0.7
        assert low_conf_result.confidence < 0.5

    def test_pattern_stats(self, sample_patterns):
        """Test pattern statistics functionality."""
        detector = MessageImportanceDetector(sample_patterns)
        stats = detector.get_pattern_stats()
        
        assert "total_patterns" in stats
        assert "total_keywords" in stats
        assert "patterns" in stats
        assert stats["total_patterns"] == len(sample_patterns)
        
        for pattern_name in sample_patterns.keys():
            assert pattern_name in stats["patterns"]
            assert "weight" in stats["patterns"][pattern_name]

    def test_pattern_updates(self, sample_patterns):
        """Test dynamic pattern updates."""
        detector = MessageImportanceDetector(sample_patterns)
        initial_count = len(detector.patterns)
        
        new_patterns = {
            "security": {
                "keywords": ["security", "breach", "hack", "vulnerability"],
                "weight": 0.95,
                "context_keywords": ["data", "password", "account"],
                "regex_patterns": [r"\b(security|breach|hack)\b"]
            }
        }
        
        detector.update_patterns(new_patterns)
        
        assert len(detector.patterns) == initial_count + 1
        assert "security" in detector.patterns
        
        # Test new pattern works
        security_message = "Security breach detected in user accounts"
        result = detector.detect_importance(security_message)
        assert result.score >= 0.9
        assert "security" in result.detected_patterns


@pytest.mark.ml
@pytest.mark.integration
class TestImportanceDetectorIntegration:
    """Integration tests for importance detector."""
    
    def test_factory_function_default(self):
        """Test factory function with default config."""
        detector = create_importance_detector()
        assert isinstance(detector, MessageImportanceDetector)
        assert len(detector.patterns) > 0

    def test_factory_function_with_config(self, sample_patterns):
        """Test factory function with config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_patterns, f)
            config_path = f.name
        
        try:
            detector = create_importance_detector(config_path)
            assert len(detector.patterns) == len(sample_patterns)
        finally:
            os.unlink(config_path)

    def test_factory_function_invalid_config(self):
        """Test factory function with invalid config file."""
        detector = create_importance_detector("/nonexistent/path.json")
        # Should fall back to default patterns
        assert len(detector.patterns) > 0


@pytest.mark.ml
@pytest.mark.performance
class TestImportanceDetectorPerformance:
    """Performance tests for importance detector."""
    
    def test_detection_performance(self, sample_patterns, performance_baseline):
        """Test importance detection performance."""
        import time
        
        detector = MessageImportanceDetector(sample_patterns)
        
        test_messages = [
            "URGENT: Server maintenance required immediately!",
            "Group buy for keyboards starting tomorrow",
            "Meeting scheduled for next week",
            "Just a regular chat message",
            "New announcement for everyone @here"
        ] * 100  # 500 messages total
        
        start_time = time.time()
        
        for message in test_messages:
            result = detector.detect_importance(message)
            assert isinstance(result, ImportanceResult)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        messages_per_second = len(test_messages) / total_time
        avg_time_per_message = total_time / len(test_messages)
        
        # Should meet performance baseline
        assert avg_time_per_message <= performance_baseline["importance_scoring_time"]
        assert messages_per_second >= 20  # Should process at least 20 messages/second

    def test_memory_usage(self, sample_patterns):
        """Test memory usage of importance detector."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple detectors
        detectors = []
        for _ in range(10):
            detector = MessageImportanceDetector(sample_patterns)
            detectors.append(detector)
        
        # Process many messages
        for detector in detectors:
            for i in range(100):
                message = f"Test message {i} with various keywords urgent group buy event"
                detector.detect_importance(message)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50MB for this test)
        assert memory_increase < 50


@pytest.mark.ml
@pytest.mark.regression
class TestImportanceDetectorRegression:
    """Regression tests for importance detector."""
    
    def test_known_good_cases(self, sample_patterns):
        """Test known good cases to prevent regressions."""
        detector = MessageImportanceDetector(sample_patterns)
        
        # Known test cases with expected results
        test_cases = [
            {
                "message": "URGENT: Server is down, immediate assistance needed!",
                "expected_score_min": 0.85,
                "expected_patterns": ["urgent"],
                "expected_confidence_min": 0.7
            },
            {
                "message": "New group buy for mechanical keyboards, $100 price point",
                "expected_score_min": 0.75,
                "expected_patterns": ["group_buy"],
                "expected_confidence_min": 0.6
            },
            {
                "message": "Meeting tomorrow at 3 PM, please attend",
                "expected_score_min": 0.65,
                "expected_patterns": ["event"],
                "expected_confidence_min": 0.6
            },
            {
                "message": "hello everyone",
                "expected_score_max": 0.4,
                "expected_patterns": [],
                "expected_confidence_max": 0.7
            }
        ]
        
        for case in test_cases:
            result = detector.detect_importance(case["message"])
            
            if "expected_score_min" in case:
                assert result.score >= case["expected_score_min"], \
                    f"Score too low for: {case['message']}"
            
            if "expected_score_max" in case:
                assert result.score <= case["expected_score_max"], \
                    f"Score too high for: {case['message']}"
            
            if "expected_patterns" in case:
                for pattern in case["expected_patterns"]:
                    assert pattern in result.detected_patterns, \
                        f"Missing pattern {pattern} for: {case['message']}"
            
            if "expected_confidence_min" in case:
                assert result.confidence >= case["expected_confidence_min"], \
                    f"Confidence too low for: {case['message']}"
            
            if "expected_confidence_max" in case:
                assert result.confidence <= case["expected_confidence_max"], \
                    f"Confidence too high for: {case['message']}"


@pytest.mark.ml
@pytest.mark.slow
class TestImportanceDetectorAccuracy:
    """Accuracy tests for importance detector."""
    
    def test_classification_accuracy(self, sample_patterns, quality_metrics):
        """Test overall classification accuracy."""
        detector = MessageImportanceDetector(sample_patterns)
        
        # Test dataset with labeled importance scores
        test_dataset = [
            # High importance (score >= 0.7)
            ("URGENT: System failure, fix needed ASAP!", 1.0),
            ("Group buy deadline tomorrow, last chance!", 0.9), 
            ("Emergency meeting in 1 hour", 0.8),
            ("Server maintenance tonight @everyone", 0.8),
            ("CRITICAL: Security breach detected", 0.95),
            
            # Medium importance (0.4 <= score < 0.7)
            ("Meeting next week, please confirm attendance", 0.6),
            ("New software update available", 0.5),
            ("Question about keyboard recommendations", 0.4),
            ("Announcement: New member joined", 0.5),
            
            # Low importance (score < 0.4)
            ("Good morning everyone!", 0.2),
            ("Thanks for the help", 0.1),
            ("lol that's funny", 0.1),
            ("How's everyone doing?", 0.3),
            ("Nice weather today", 0.1),
        ]
        
        correct_predictions = 0
        total_predictions = len(test_dataset)
        
        high_threshold = 0.7
        medium_threshold = 0.4
        
        for message, expected_score in test_dataset:
            result = detector.detect_importance(message)
            predicted_score = result.score
            
            # Classify into categories
            if expected_score >= high_threshold:
                expected_category = "high"
            elif expected_score >= medium_threshold:
                expected_category = "medium"
            else:
                expected_category = "low"
            
            if predicted_score >= high_threshold:
                predicted_category = "high"
            elif predicted_score >= medium_threshold:
                predicted_category = "medium"
            else:
                predicted_category = "low"
            
            if expected_category == predicted_category:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        assert accuracy >= quality_metrics["importance_accuracy"], \
            f"Accuracy {accuracy:.2f} below threshold {quality_metrics['importance_accuracy']}"

    def test_false_positive_rate(self, sample_patterns, quality_metrics):
        """Test false positive rate for low importance messages."""
        detector = MessageImportanceDetector(sample_patterns)
        
        # Low importance messages that should NOT trigger high scores
        low_importance_messages = [
            "Good morning!",
            "Thanks for sharing",
            "How are you?", 
            "Nice work!",
            "Have a great day",
            "See you later",
            "lol",
            "What's for lunch?",
            "Beautiful weather",
            "Happy birthday!"
        ]
        
        false_positives = 0
        
        for message in low_importance_messages:
            result = detector.detect_importance(message)
            if result.score >= 0.7:  # Incorrectly classified as high importance
                false_positives += 1
        
        false_positive_rate = false_positives / len(low_importance_messages)
        assert false_positive_rate <= quality_metrics["false_positive_rate"], \
            f"False positive rate {false_positive_rate:.2f} above threshold"

    def test_false_negative_rate(self, sample_patterns, quality_metrics):
        """Test false negative rate for high importance messages.""" 
        detector = MessageImportanceDetector(sample_patterns)
        
        # High importance messages that SHOULD trigger high scores
        high_importance_messages = [
            "URGENT: Server is completely down!",
            "Emergency: Data loss detected",
            "Critical bug in production system",
            "Group buy closing in 1 hour!",
            "ASAP: Need help with system restore",
            "Breaking: Major announcement coming",
            "Immediate action required @everyone",
            "System compromised, all hands on deck",
            "Deadline today, submission required",
            "Emergency maintenance in 30 minutes"
        ]
        
        false_negatives = 0
        
        for message in high_importance_messages:
            result = detector.detect_importance(message)
            if result.score < 0.7:  # Incorrectly classified as low importance
                false_negatives += 1
        
        false_negative_rate = false_negatives / len(high_importance_messages)
        assert false_negative_rate <= quality_metrics["false_negative_rate"], \
            f"False negative rate {false_negative_rate:.2f} above threshold"