"""
Tests for quality metrics and monitoring system.
"""
import pytest
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from tests.utils.quality_metrics import (
    QualityMetricsCollector, QualityMonitor, QualityMetric, 
    TestResult, QualityReport, get_quality_collector
)


@pytest.mark.quality
@pytest.mark.unit
class TestQualityMetricsCollector:
    """Test quality metrics collection functionality."""
    
    def test_collector_initialization(self):
        """Test quality metrics collector initialization."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            
            assert collector.metrics_db_path == tmp_file.name
            assert len(collector.thresholds) > 0
            assert 'importance_accuracy' in collector.thresholds
    
    def test_record_metric_pass(self):
        """Test recording a passing metric."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            
            collector.record_metric('importance_accuracy', 0.90, {'model': 'test'})
            
            assert 'importance_accuracy' in collector.current_metrics
            metric = collector.current_metrics['importance_accuracy']
            assert metric.value == 0.90
            assert metric.status == 'pass'
            assert metric.details['model'] == 'test'
    
    def test_record_metric_fail(self):
        """Test recording a failing metric."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            
            collector.record_metric('importance_accuracy', 0.70)  # Below threshold
            
            metric = collector.current_metrics['importance_accuracy']
            assert metric.value == 0.70
            assert metric.status == 'fail'
    
    def test_record_test_result(self):
        """Test recording test results."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            
            collector.record_test_result(
                'test_importance_detection',
                'pass',
                1.5,
                metrics={'accuracy': 0.95}
            )
            
            # Verify it was saved to database
            summary = collector.get_test_results_summary(hours=1)
            assert summary['total'] == 1
            assert 'pass' in summary
            assert summary['pass']['count'] == 1
    
    def test_get_metrics_history(self):
        """Test retrieving metrics history."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            
            # Record multiple metrics
            collector.record_metric('api_response_time', 1.0)
            collector.record_metric('api_response_time', 1.5)
            collector.record_metric('api_response_time', 0.8)
            
            history = collector.get_metrics_history('api_response_time', hours=1)
            assert len(history) == 3
            assert all(m.name == 'api_response_time' for m in history)
    
    def test_test_results_summary(self):
        """Test test results summary generation."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            
            # Record various test results
            collector.record_test_result('test1', 'pass', 1.0)
            collector.record_test_result('test2', 'pass', 1.2)
            collector.record_test_result('test3', 'fail', 2.0, 'Assertion error')
            collector.record_test_result('test4', 'skip', 0.0)
            
            summary = collector.get_test_results_summary(hours=1)
            
            assert summary['total'] == 4
            assert summary['pass']['count'] == 2
            assert summary['fail']['count'] == 1
            assert summary['skip']['count'] == 1
            assert summary['pass']['percentage'] == 50.0
    
    def test_generate_quality_report(self):
        """Test quality report generation."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            
            # Record some metrics and test results
            collector.record_metric('importance_accuracy', 0.90)
            collector.record_metric('false_positive_rate', 0.10)
            collector.record_test_result('test1', 'pass', 1.0)
            collector.record_test_result('test2', 'pass', 1.5)
            
            report = collector.generate_quality_report()
            
            assert isinstance(report, QualityReport)
            assert report.overall_score > 0.0
            assert len(report.metrics) > 0
            assert len(report.recommendations) > 0
            assert 'cpu_percent' in report.system_info
    
    def test_overall_score_calculation(self):
        """Test overall quality score calculation."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            
            # All metrics passing
            collector.record_metric('importance_accuracy', 0.95)
            collector.record_metric('false_positive_rate', 0.05)
            collector.record_metric('api_response_time', 0.5)
            collector.record_test_result('test1', 'pass', 1.0)
            collector.record_test_result('test2', 'pass', 1.2)
            
            report = collector.generate_quality_report()
            assert report.overall_score > 0.8  # Should be high with all passing
            
            # Add some failures
            collector.record_metric('importance_accuracy', 0.70)  # Fail
            collector.record_test_result('test3', 'fail', 2.0, 'Error')
            
            report2 = collector.generate_quality_report()
            assert report2.overall_score < report.overall_score  # Should be lower
    
    def test_recommendations_generation(self):
        """Test recommendation generation."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            
            # Record failing metrics
            collector.record_metric('importance_accuracy', 0.70)  # Below threshold
            collector.record_metric('api_response_time', 3.0)    # Above threshold
            collector.record_test_result('test1', 'fail', 2.0, 'Error')
            
            report = collector.generate_quality_report()
            
            recommendations = report.recommendations
            assert len(recommendations) > 0
            
            # Check for specific recommendations
            accuracy_rec = any('accuracy' in rec.lower() for rec in recommendations)
            api_rec = any('api response time' in rec.lower() for rec in recommendations)
            test_rec = any('test failure' in rec.lower() for rec in recommendations)
            
            assert accuracy_rec or api_rec or test_rec
    
    def test_export_metrics_json(self):
        """Test exporting metrics to JSON."""
        with tempfile.NamedTemporaryFile(suffix='.db') as db_file, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as json_file:
            
            collector = QualityMetricsCollector(db_file.name)
            
            # Record some data
            collector.record_metric('test_metric', 0.95)
            collector.record_test_result('test1', 'pass', 1.0)
            
            # Export to JSON
            collector.export_metrics(json_file.name, format='json')
            
            # Verify export
            with open(json_file.name, 'r') as f:
                exported_data = json.load(f)
            
            assert 'metrics' in exported_data
            assert 'test_results' in exported_data
            assert len(exported_data['metrics']) > 0
            assert len(exported_data['test_results']) > 0


@pytest.mark.quality
@pytest.mark.unit
class TestQualityMonitor:
    """Test quality monitoring functionality."""
    
    def test_monitor_initialization(self):
        """Test quality monitor initialization."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            monitor = QualityMonitor(collector)
            
            assert monitor.collector == collector
            assert len(monitor.alerts) == 0
            assert monitor.monitoring == False
    
    def test_alert_generation(self):
        """Test alert generation for failing metrics."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            monitor = QualityMonitor(collector)
            
            # Record failing metric
            collector.record_metric('importance_accuracy', 0.70)  # Below threshold
            
            # Trigger threshold check
            monitor._check_quality_thresholds()
            
            alerts = monitor.get_recent_alerts()
            assert len(alerts) > 0
            
            alert = alerts[0]
            assert alert['metric_name'] == 'importance_accuracy'
            assert alert['value'] == 0.70
            assert 'threshold exceeded' in alert['message'].lower()
    
    def test_duplicate_alert_prevention(self):
        """Test prevention of duplicate alerts."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            monitor = QualityMonitor(collector)
            
            # Record same failing metric multiple times
            collector.record_metric('api_response_time', 5.0)
            monitor._check_quality_thresholds()
            
            collector.record_metric('api_response_time', 5.1)
            monitor._check_quality_thresholds()
            
            alerts = monitor.get_recent_alerts()
            # Should only have one alert despite multiple checks
            api_alerts = [a for a in alerts if a['metric_name'] == 'api_response_time']
            assert len(api_alerts) == 1
    
    @patch('time.sleep')
    def test_monitoring_lifecycle(self, mock_sleep):
        """Test starting and stopping monitoring."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            monitor = QualityMonitor(collector)
            
            # Start monitoring
            monitor.start_monitoring(check_interval=1)
            assert monitor.monitoring == True
            
            # Stop monitoring
            monitor.stop_monitoring()
            assert monitor.monitoring == False


@pytest.mark.quality
@pytest.mark.integration
class TestQualityMetricsIntegration:
    """Integration tests for quality metrics system."""
    
    def test_end_to_end_quality_tracking(self, sample_patterns):
        """Test end-to-end quality tracking workflow."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            monitor = QualityMonitor(collector)
            
            # Simulate running tests and collecting metrics
            from lib.importance_detector import MessageImportanceDetector
            
            detector = MessageImportanceDetector(sample_patterns)
            
            # Test importance detection and record metrics
            test_cases = [
                ("URGENT: Server down!", True),
                ("Group buy starting tomorrow", True),
                ("Hello everyone", False),
                ("Thanks for the help", False)
            ]
            
            correct_predictions = 0
            total_predictions = len(test_cases)
            
            for message, expected_important in test_cases:
                result = detector.detect_importance(message)
                predicted_important = result.score >= 0.7
                
                if predicted_important == expected_important:
                    correct_predictions += 1
            
            # Record accuracy metric
            accuracy = correct_predictions / total_predictions
            collector.record_metric('importance_accuracy', accuracy)
            
            # Record test results
            if accuracy >= 0.75:
                collector.record_test_result('test_importance_detection', 'pass', 1.0,
                                           metrics={'accuracy': accuracy})
            else:
                collector.record_test_result('test_importance_detection', 'fail', 1.0,
                                           error_message='Low accuracy',
                                           metrics={'accuracy': accuracy})
            
            # Generate quality report
            report = collector.generate_quality_report()
            
            assert report.overall_score > 0.0
            assert len(report.metrics) > 0
            assert 'importance_accuracy' in [m.name for m in report.metrics]
            
            # Check monitoring
            monitor._check_quality_thresholds()
            if accuracy < 0.85:  # Below threshold
                alerts = monitor.get_recent_alerts()
                assert len(alerts) > 0
    
    def test_global_collector_access(self):
        """Test global quality collector access."""
        collector1 = get_quality_collector()
        collector2 = get_quality_collector()
        
        # Should return same instance
        assert collector1 is collector2
        
        # Should be functional
        collector1.record_metric('test_metric', 0.95)
        assert 'test_metric' in collector1.current_metrics


@pytest.mark.quality
@pytest.mark.performance
class TestQualityMetricsPerformance:
    """Performance tests for quality metrics system."""
    
    def test_metrics_recording_performance(self, performance_baseline):
        """Test performance of metrics recording."""
        import time
        
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            
            # Record many metrics
            start_time = time.time()
            
            for i in range(1000):
                collector.record_metric(f'test_metric_{i % 10}', 0.95, {'iteration': i})
            
            end_time = time.time()
            
            total_time = end_time - start_time
            metrics_per_second = 1000 / total_time
            
            # Should be able to record metrics quickly
            assert metrics_per_second >= 100  # At least 100 metrics/second
            assert total_time < 10  # Should complete within 10 seconds
    
    def test_quality_report_generation_performance(self):
        """Test performance of quality report generation."""
        import time
        
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            
            # Record many metrics and test results
            for i in range(100):
                collector.record_metric('importance_accuracy', 0.90 + (i % 10) * 0.01)
                collector.record_test_result(f'test_{i}', 'pass', 1.0)
            
            # Generate report and measure time
            start_time = time.time()
            report = collector.generate_quality_report()
            end_time = time.time()
            
            generation_time = end_time - start_time
            
            assert generation_time < 5.0  # Should generate within 5 seconds
            assert isinstance(report, QualityReport)
            assert report.overall_score > 0.0


@pytest.mark.quality
@pytest.mark.regression
class TestQualityMetricsRegression:
    """Regression tests for quality metrics system."""
    
    def test_metric_threshold_consistency(self):
        """Test that metric thresholds remain consistent."""
        collector = QualityMetricsCollector()
        
        # Verify expected thresholds
        expected_thresholds = {
            'importance_accuracy': 0.85,
            'false_positive_rate': 0.15,
            'false_negative_rate': 0.10,
            'notification_relevance': 0.80,
            'test_coverage': 0.80
        }
        
        for metric_name, expected_threshold in expected_thresholds.items():
            assert metric_name in collector.thresholds
            assert collector.thresholds[metric_name] == expected_threshold
    
    def test_database_schema_consistency(self):
        """Test that database schema remains consistent."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            
            import sqlite3
            conn = sqlite3.connect(tmp_file.name)
            cursor = conn.cursor()
            
            # Check metrics table
            cursor.execute("PRAGMA table_info(metrics)")
            metrics_columns = [row[1] for row in cursor.fetchall()]
            expected_metrics_columns = ['id', 'name', 'value', 'threshold', 'status', 'timestamp', 'details']
            assert all(col in metrics_columns for col in expected_metrics_columns)
            
            # Check test_results table
            cursor.execute("PRAGMA table_info(test_results)")
            test_columns = [row[1] for row in cursor.fetchall()]
            expected_test_columns = ['id', 'test_name', 'status', 'duration', 'error_message', 'metrics', 'timestamp']
            assert all(col in test_columns for col in expected_test_columns)
            
            conn.close()
    
    def test_report_format_consistency(self):
        """Test that quality report format remains consistent."""
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
            collector = QualityMetricsCollector(tmp_file.name)
            
            collector.record_metric('test_metric', 0.95)
            collector.record_test_result('test1', 'pass', 1.0)
            
            report = collector.generate_quality_report()
            
            # Verify report structure
            assert hasattr(report, 'timestamp')
            assert hasattr(report, 'overall_score')
            assert hasattr(report, 'metrics')
            assert hasattr(report, 'test_results')
            assert hasattr(report, 'recommendations')
            assert hasattr(report, 'system_info')
            
            assert isinstance(report.overall_score, float)
            assert 0.0 <= report.overall_score <= 1.0
            assert isinstance(report.recommendations, list)
            assert isinstance(report.system_info, dict)