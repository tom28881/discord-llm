"""
Quality metrics tracking and monitoring system for Discord monitoring assistant.
"""
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import threading

logger = logging.getLogger('quality_metrics')


@dataclass
class QualityMetric:
    """Individual quality metric measurement."""
    name: str
    value: float
    threshold: float
    timestamp: datetime
    status: str  # 'pass', 'fail', 'warning'
    details: Dict[str, Any]


@dataclass
class TestResult:
    """Test execution result."""
    test_name: str
    status: str  # 'pass', 'fail', 'skip'
    duration: float
    error_message: Optional[str]
    metrics: Dict[str, float]
    timestamp: datetime


@dataclass
class QualityReport:
    """Comprehensive quality report."""
    timestamp: datetime
    overall_score: float
    metrics: List[QualityMetric]
    test_results: List[TestResult]
    recommendations: List[str]
    system_info: Dict[str, Any]


class QualityMetricsCollector:
    """Collects and tracks quality metrics over time."""
    
    def __init__(self, metrics_db_path: str = "data/quality_metrics.db"):
        self.metrics_db_path = metrics_db_path
        self.current_metrics = {}
        self.thresholds = self._load_default_thresholds()
        self._init_metrics_db()
        self._lock = threading.Lock()
    
    def _load_default_thresholds(self) -> Dict[str, float]:
        """Load default quality thresholds."""
        return {
            "importance_accuracy": 0.85,
            "false_positive_rate": 0.15,
            "false_negative_rate": 0.10,
            "notification_relevance": 0.80,
            "system_uptime": 0.99,
            "test_coverage": 0.80,
            "api_response_time": 2.0,
            "database_query_time": 0.5,
            "memory_usage_mb": 300,
            "cpu_usage_percent": 80,
            "message_processing_rate": 100,
            "error_rate": 0.05,
            "availability": 0.995
        }
    
    def _init_metrics_db(self):
        """Initialize metrics database."""
        Path(self.metrics_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.metrics_db_path)
        cursor = conn.cursor()
        
        # Metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            threshold REAL NOT NULL,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            details TEXT
        )
        ''')
        
        # Test results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_name TEXT NOT NULL,
            status TEXT NOT NULL,
            duration REAL NOT NULL,
            error_message TEXT,
            metrics TEXT,
            timestamp TEXT NOT NULL
        )
        ''')
        
        # Quality reports table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS quality_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            overall_score REAL NOT NULL,
            report_data TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_metric(self, name: str, value: float, details: Dict[str, Any] = None):
        """Record a quality metric."""
        with self._lock:
            threshold = self.thresholds.get(name, 1.0)
            
            # Determine status based on metric type
            if name.endswith('_rate') and name != 'message_processing_rate':
                # Lower is better for rates like error_rate, false_positive_rate
                status = 'pass' if value <= threshold else 'fail'
            elif name in ['accuracy', 'relevance', 'uptime', 'coverage', 'availability']:
                # Higher is better for these metrics
                status = 'pass' if value >= threshold else 'fail'
            elif name.endswith('_time'):
                # Lower is better for time metrics
                status = 'pass' if value <= threshold else 'fail'
            else:
                # Default: lower is better
                status = 'pass' if value <= threshold else 'fail'
            
            metric = QualityMetric(
                name=name,
                value=value,
                threshold=threshold,
                timestamp=datetime.now(),
                status=status,
                details=details or {}
            )
            
            self.current_metrics[name] = metric
            self._save_metric(metric)
    
    def _save_metric(self, metric: QualityMetric):
        """Save metric to database."""
        conn = sqlite3.connect(self.metrics_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO metrics (name, value, threshold, status, timestamp, details)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            metric.name,
            metric.value,
            metric.threshold,
            metric.status,
            metric.timestamp.isoformat(),
            json.dumps(metric.details)
        ))
        
        conn.commit()
        conn.close()
    
    def record_test_result(self, test_name: str, status: str, duration: float,
                          error_message: str = None, metrics: Dict[str, float] = None):
        """Record test execution result."""
        with self._lock:
            result = TestResult(
                test_name=test_name,
                status=status,
                duration=duration,
                error_message=error_message,
                metrics=metrics or {},
                timestamp=datetime.now()
            )
            
            self._save_test_result(result)
    
    def _save_test_result(self, result: TestResult):
        """Save test result to database."""
        conn = sqlite3.connect(self.metrics_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO test_results (test_name, status, duration, error_message, metrics, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            result.test_name,
            result.status,
            result.duration,
            result.error_message,
            json.dumps(result.metrics),
            result.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_metrics_history(self, metric_name: str, hours: int = 24) -> List[QualityMetric]:
        """Get history of a specific metric."""
        since = datetime.now() - timedelta(hours=hours)
        
        conn = sqlite3.connect(self.metrics_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT name, value, threshold, status, timestamp, details
        FROM metrics
        WHERE name = ? AND timestamp >= ?
        ORDER BY timestamp DESC
        ''', (metric_name, since.isoformat()))
        
        metrics = []
        for row in cursor.fetchall():
            metric = QualityMetric(
                name=row[0],
                value=row[1],
                threshold=row[2],
                status=row[3],
                timestamp=datetime.fromisoformat(row[4]),
                details=json.loads(row[5]) if row[5] else {}
            )
            metrics.append(metric)
        
        conn.close()
        return metrics
    
    def get_test_results_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of test results."""
        since = datetime.now() - timedelta(hours=hours)
        
        conn = sqlite3.connect(self.metrics_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT status, COUNT(*) as count, AVG(duration) as avg_duration
        FROM test_results
        WHERE timestamp >= ?
        GROUP BY status
        ''', (since.isoformat(),))
        
        summary = {}
        total_tests = 0
        
        for row in cursor.fetchall():
            status, count, avg_duration = row
            summary[status] = {
                'count': count,
                'avg_duration': avg_duration
            }
            total_tests += count
        
        # Calculate percentages
        for status in summary:
            summary[status]['percentage'] = (summary[status]['count'] / total_tests) * 100
        
        summary['total'] = total_tests
        conn.close()
        
        return summary
    
    def generate_quality_report(self) -> QualityReport:
        """Generate comprehensive quality report."""
        timestamp = datetime.now()
        
        # Get recent metrics
        metrics = []
        for name in self.thresholds.keys():
            recent = self.get_metrics_history(name, hours=1)
            if recent:
                metrics.append(recent[0])  # Most recent value
        
        # Get test results
        test_summary = self.get_test_results_summary(hours=24)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(metrics, test_summary)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, test_summary)
        
        # System info
        system_info = self._get_system_info()
        
        report = QualityReport(
            timestamp=timestamp,
            overall_score=overall_score,
            metrics=metrics,
            test_results=[],  # Detailed results would be too large
            recommendations=recommendations,
            system_info=system_info
        )
        
        self._save_quality_report(report)
        return report
    
    def _calculate_overall_score(self, metrics: List[QualityMetric], 
                                test_summary: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        if not metrics:
            return 0.0
        
        # Metric scores (weighted by importance)
        metric_weights = {
            'importance_accuracy': 0.2,
            'false_positive_rate': 0.15,
            'false_negative_rate': 0.15,
            'notification_relevance': 0.1,
            'test_coverage': 0.1,
            'api_response_time': 0.05,
            'database_query_time': 0.05,
            'error_rate': 0.1,
            'availability': 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            weight = metric_weights.get(metric.name, 0.01)
            if metric.status == 'pass':
                score = 1.0
            elif metric.status == 'warning':
                score = 0.7
            else:
                score = 0.0
            
            weighted_score += score * weight
            total_weight += weight
        
        # Test success rate
        test_pass_rate = 0.0
        if test_summary.get('total', 0) > 0:
            pass_count = test_summary.get('pass', {}).get('count', 0)
            test_pass_rate = pass_count / test_summary['total']
        
        # Combine metric score and test pass rate
        if total_weight > 0:
            metric_score = weighted_score / total_weight
            overall_score = (metric_score * 0.7) + (test_pass_rate * 0.3)
        else:
            overall_score = test_pass_rate
        
        return min(max(overall_score, 0.0), 1.0)
    
    def _generate_recommendations(self, metrics: List[QualityMetric], 
                                 test_summary: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Check failed metrics
        failed_metrics = [m for m in metrics if m.status == 'fail']
        
        for metric in failed_metrics:
            if metric.name == 'importance_accuracy':
                recommendations.append(
                    f"Importance detection accuracy is {metric.value:.2f}, below threshold {metric.threshold:.2f}. "
                    "Consider retraining the model or adjusting detection patterns."
                )
            elif metric.name == 'false_positive_rate':
                recommendations.append(
                    f"False positive rate is {metric.value:.2f}, above threshold {metric.threshold:.2f}. "
                    "Review importance detection criteria to reduce false positives."
                )
            elif metric.name == 'api_response_time':
                recommendations.append(
                    f"API response time is {metric.value:.2f}s, above threshold {metric.threshold:.2f}s. "
                    "Consider optimizing API calls or implementing caching."
                )
            elif metric.name == 'database_query_time':
                recommendations.append(
                    f"Database query time is {metric.value:.2f}s, above threshold {metric.threshold:.2f}s. "
                    "Consider adding database indexes or optimizing queries."
                )
            elif metric.name == 'memory_usage_mb':
                recommendations.append(
                    f"Memory usage is {metric.value:.1f}MB, above threshold {metric.threshold:.1f}MB. "
                    "Investigate memory leaks or optimize data structures."
                )
        
        # Check test failures
        fail_rate = test_summary.get('fail', {}).get('percentage', 0)
        if fail_rate > 10:
            recommendations.append(
                f"Test failure rate is {fail_rate:.1f}%. "
                "Review and fix failing tests to maintain code quality."
            )
        
        # Check test coverage
        coverage_metrics = [m for m in metrics if m.name == 'test_coverage']
        if coverage_metrics and coverage_metrics[0].status == 'fail':
            recommendations.append(
                f"Test coverage is {coverage_metrics[0].value:.2f}, below threshold. "
                "Add more unit tests to improve coverage."
            )
        
        if not recommendations:
            recommendations.append("All quality metrics are within acceptable thresholds. Great job!")
        
        return recommendations
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        import psutil
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_quality_report(self, report: QualityReport):
        """Save quality report to database."""
        conn = sqlite3.connect(self.metrics_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO quality_reports (overall_score, report_data, timestamp)
        VALUES (?, ?, ?)
        ''', (
            report.overall_score,
            json.dumps(asdict(report), default=str),
            report.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def export_metrics(self, output_file: str, format: str = 'json'):
        """Export metrics to file."""
        if format == 'json':
            self._export_json(output_file)
        elif format == 'csv':
            self._export_csv(output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self, output_file: str):
        """Export metrics as JSON."""
        conn = sqlite3.connect(self.metrics_db_path)
        cursor = conn.cursor()
        
        # Get all metrics
        cursor.execute('SELECT * FROM metrics ORDER BY timestamp DESC LIMIT 1000')
        metrics_data = cursor.fetchall()
        
        # Get all test results
        cursor.execute('SELECT * FROM test_results ORDER BY timestamp DESC LIMIT 1000')
        test_data = cursor.fetchall()
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'metrics': [
                {
                    'name': row[1],
                    'value': row[2],
                    'threshold': row[3],
                    'status': row[4],
                    'timestamp': row[5],
                    'details': json.loads(row[6]) if row[6] else {}
                }
                for row in metrics_data
            ],
            'test_results': [
                {
                    'test_name': row[1],
                    'status': row[2],
                    'duration': row[3],
                    'error_message': row[4],
                    'metrics': json.loads(row[5]) if row[5] else {},
                    'timestamp': row[6]
                }
                for row in test_data
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        conn.close()
    
    def _export_csv(self, output_file: str):
        """Export metrics as CSV."""
        import csv
        
        conn = sqlite3.connect(self.metrics_db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT name, value, threshold, status, timestamp FROM metrics ORDER BY timestamp DESC LIMIT 1000')
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_name', 'value', 'threshold', 'status', 'timestamp'])
            writer.writerows(cursor.fetchall())
        
        conn.close()


class QualityMonitor:
    """Real-time quality monitoring."""
    
    def __init__(self, collector: QualityMetricsCollector):
        self.collector = collector
        self.alerts = []
        self.monitoring = False
    
    def start_monitoring(self, check_interval: int = 60):
        """Start continuous quality monitoring."""
        import threading
        import time
        
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                try:
                    self._check_quality_thresholds()
                    time.sleep(check_interval)
                except Exception as e:
                    logger.error(f"Error in quality monitoring: {e}")
                    time.sleep(check_interval)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        logger.info("Quality monitoring started")
    
    def stop_monitoring(self):
        """Stop quality monitoring."""
        self.monitoring = False
        logger.info("Quality monitoring stopped")
    
    def _check_quality_thresholds(self):
        """Check if quality metrics exceed thresholds."""
        current_time = datetime.now()
        
        for metric_name, metric in self.collector.current_metrics.items():
            if metric.status == 'fail':
                alert = {
                    'timestamp': current_time,
                    'metric_name': metric_name,
                    'value': metric.value,
                    'threshold': metric.threshold,
                    'message': f"Quality threshold exceeded: {metric_name} = {metric.value} (threshold: {metric.threshold})"
                }
                
                # Avoid duplicate alerts
                if not self._is_duplicate_alert(alert):
                    self.alerts.append(alert)
                    logger.warning(alert['message'])
    
    def _is_duplicate_alert(self, new_alert: Dict[str, Any]) -> bool:
        """Check if alert is duplicate of recent alert."""
        cutoff_time = datetime.now() - timedelta(minutes=5)
        
        for alert in self.alerts:
            if (alert['timestamp'] > cutoff_time and
                alert['metric_name'] == new_alert['metric_name']):
                return True
        
        return False
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent quality alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]


# Global quality metrics collector instance
_quality_collector = None

def get_quality_collector() -> QualityMetricsCollector:
    """Get global quality metrics collector."""
    global _quality_collector
    if _quality_collector is None:
        _quality_collector = QualityMetricsCollector()
    return _quality_collector


def record_metric(name: str, value: float, details: Dict[str, Any] = None):
    """Convenience function to record a metric."""
    collector = get_quality_collector()
    collector.record_metric(name, value, details)


def record_test_result(test_name: str, status: str, duration: float,
                      error_message: str = None, metrics: Dict[str, float] = None):
    """Convenience function to record a test result.""" 
    collector = get_quality_collector()
    collector.record_test_result(test_name, status, duration, error_message, metrics)