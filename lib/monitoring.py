"""
Comprehensive logging, monitoring, and alerting system for Discord monitoring assistant.

Provides structured logging, performance metrics, health monitoring, alerting,
and debugging capabilities for production reliability.
"""

import logging
import logging.handlers
import json
import time
import threading
import queue
import os
import smtplib
import requests
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from pathlib import Path
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import traceback
import psutil
import gc

from .exceptions import DiscordMonitorException

# Custom JSON formatter for structured logging
class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': record.thread,
            'thread_name': record.threadName,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in log_data and not key.startswith('_'):
                try:
                    json.dumps(value)  # Test if JSON serializable
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)
        
        return json.dumps(log_data, default=str)


@dataclass
class MetricData:
    """Container for metric data points."""
    name: str
    value: Union[int, float]
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class AlertConfig:
    """Configuration for alert rules."""
    name: str
    condition: Callable[[Any], bool]
    message_template: str
    cooldown_minutes: int = 5
    severity: str = "warning"  # info, warning, error, critical
    channels: List[str] = field(default_factory=list)  # email, slack, webhook


class MetricsCollector:
    """Collects and stores application metrics."""
    
    def __init__(self, max_datapoints: int = 10000):
        self.max_datapoints = max_datapoints
        self.metrics = defaultdict(lambda: deque(maxlen=max_datapoints))
        self.lock = threading.Lock()
    
    def record(self, name: str, value: Union[int, float], tags: Dict[str, str] = None, unit: str = ""):
        """Record a metric data point."""
        with self.lock:
            metric = MetricData(name, value, tags=tags or {}, unit=unit)
            self.metrics[name].append(metric)
    
    def increment(self, name: str, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        self.record(name, 1, tags, "count")
    
    def gauge(self, name: str, value: Union[int, float], tags: Dict[str, str] = None, unit: str = ""):
        """Set a gauge metric."""
        self.record(name, value, tags, unit)
    
    def timing(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timing metric."""
        self.record(name, duration, tags, "seconds")
    
    def get_metrics(self, name: str = None, since: datetime = None) -> Dict[str, List[MetricData]]:
        """Get metrics, optionally filtered by name and time."""
        with self.lock:
            if name:
                metrics = {name: list(self.metrics.get(name, []))}
            else:
                metrics = {k: list(v) for k, v in self.metrics.items()}
        
        if since:
            filtered_metrics = {}
            for metric_name, datapoints in metrics.items():
                filtered_datapoints = [dp for dp in datapoints if dp.timestamp >= since]
                if filtered_datapoints:
                    filtered_metrics[metric_name] = filtered_datapoints
            metrics = filtered_metrics
        
        return metrics
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        with self.lock:
            if name in self.metrics and self.metrics[name]:
                return self.metrics[name][-1].value
        return None
    
    def get_aggregates(self, name: str, minutes: int = 5) -> Dict[str, float]:
        """Get aggregated values for a metric over the specified time period."""
        since = datetime.now() - timedelta(minutes=minutes)
        metrics = self.get_metrics(name, since)
        
        if name not in metrics or not metrics[name]:
            return {}
        
        values = [dp.value for dp in metrics[name]]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'sum': sum(values)
        }


class SystemMonitor:
    """Monitors system resources and health."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitoring = False
        self.monitor_thread = None
        self.interval = 30  # seconds
    
    def start_monitoring(self):
        """Start system monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.getLogger(__name__).info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """System monitoring loop."""
        logger = logging.getLogger(__name__)
        
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics.gauge("system.cpu.usage_percent", cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics.gauge("system.memory.usage_percent", memory.percent)
                self.metrics.gauge("system.memory.available_bytes", memory.available)
                self.metrics.gauge("system.memory.used_bytes", memory.used)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.metrics.gauge("system.disk.usage_percent", disk.percent)
                self.metrics.gauge("system.disk.free_bytes", disk.free)
                
                # Network I/O
                net_io = psutil.net_io_counters()
                self.metrics.gauge("system.network.bytes_sent", net_io.bytes_sent)
                self.metrics.gauge("system.network.bytes_recv", net_io.bytes_recv)
                
                # Process info
                process = psutil.Process()
                self.metrics.gauge("process.memory.rss_bytes", process.memory_info().rss)
                self.metrics.gauge("process.cpu.percent", process.cpu_percent())
                self.metrics.gauge("process.threads", process.num_threads())
                
                # Python GC stats
                gc_stats = gc.get_stats()
                for i, stats in enumerate(gc_stats):
                    self.metrics.gauge(f"python.gc.gen{i}.collections", stats['collections'])
                    self.metrics.gauge(f"python.gc.gen{i}.collected", stats['collected'])
                    self.metrics.gauge(f"python.gc.gen{i}.uncollectable", stats['uncollectable'])
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(5)  # Wait before retrying


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alert_configs = {}
        self.alert_state = {}  # Track alert cooldowns
        self.notification_channels = {}
        self.logger = logging.getLogger(__name__ + ".alerts")
    
    def add_alert(self, alert_config: AlertConfig):
        """Add an alert configuration."""
        self.alert_configs[alert_config.name] = alert_config
        self.alert_state[alert_config.name] = {
            'last_triggered': None,
            'active': False
        }
    
    def add_notification_channel(self, name: str, handler: Callable[[str, str, str], bool]):
        """Add a notification channel handler."""
        self.notification_channels[name] = handler
    
    def check_alerts(self):
        """Check all alert conditions and trigger notifications if needed."""
        for name, config in self.alert_configs.items():
            try:
                if self._should_check_alert(name, config):
                    if config.condition(self.metrics):
                        self._trigger_alert(name, config)
            except Exception as e:
                self.logger.error(f"Error checking alert {name}: {e}")
    
    def _should_check_alert(self, name: str, config: AlertConfig) -> bool:
        """Check if alert should be evaluated (considering cooldown)."""
        state = self.alert_state[name]
        
        if not state['last_triggered']:
            return True
        
        cooldown_end = state['last_triggered'] + timedelta(minutes=config.cooldown_minutes)
        return datetime.now() >= cooldown_end
    
    def _trigger_alert(self, name: str, config: AlertConfig):
        """Trigger an alert and send notifications."""
        now = datetime.now()
        self.alert_state[name]['last_triggered'] = now
        self.alert_state[name]['active'] = True
        
        # Format message
        message = config.message_template
        
        # Add context information
        context = {
            'timestamp': now.isoformat(),
            'alert_name': name,
            'severity': config.severity
        }
        
        # Try to format with context
        try:
            message = message.format(**context)
        except (KeyError, ValueError):
            pass  # Use original message if formatting fails
        
        self.logger.warning(f"Alert triggered: {name} - {message}")
        
        # Send notifications
        for channel_name in config.channels:
            if channel_name in self.notification_channels:
                try:
                    success = self.notification_channels[channel_name](name, message, config.severity)
                    if success:
                        self.metrics.increment("alerts.notifications.sent", {"channel": channel_name})
                    else:
                        self.metrics.increment("alerts.notifications.failed", {"channel": channel_name})
                except Exception as e:
                    self.logger.error(f"Notification failed for channel {channel_name}: {e}")
                    self.metrics.increment("alerts.notifications.errors", {"channel": channel_name})
        
        self.metrics.increment("alerts.triggered", {"alert": name, "severity": config.severity})


class NotificationChannels:
    """Built-in notification channel implementations."""
    
    @staticmethod
    def email_handler(smtp_host: str, smtp_port: int, username: str, password: str,
                     from_addr: str, to_addrs: List[str]) -> Callable:
        """Create email notification handler."""
        def send_email(alert_name: str, message: str, severity: str) -> bool:
            try:
                msg = MimeMultipart()
                msg['From'] = from_addr
                msg['To'] = ', '.join(to_addrs)
                msg['Subject'] = f"Discord Monitor Alert: {alert_name} [{severity.upper()}]"
                
                body = f"""
Discord Monitoring Alert

Alert: {alert_name}
Severity: {severity.upper()}
Time: {datetime.now().isoformat()}

Details:
{message}

This is an automated alert from the Discord monitoring system.
"""
                msg.attach(MimeText(body, 'plain'))
                
                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    server.starttls()
                    server.login(username, password)
                    server.sendmail(from_addr, to_addrs, msg.as_string())
                
                return True
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Email notification failed: {e}")
                return False
        
        return send_email
    
    @staticmethod
    def slack_webhook_handler(webhook_url: str) -> Callable:
        """Create Slack webhook notification handler."""
        def send_slack(alert_name: str, message: str, severity: str) -> bool:
            try:
                color_map = {
                    'info': 'good',
                    'warning': 'warning',
                    'error': 'danger',
                    'critical': 'danger'
                }
                
                payload = {
                    'text': f'Discord Monitor Alert: {alert_name}',
                    'attachments': [{
                        'color': color_map.get(severity, 'warning'),
                        'fields': [
                            {'title': 'Alert', 'value': alert_name, 'short': True},
                            {'title': 'Severity', 'value': severity.upper(), 'short': True},
                            {'title': 'Time', 'value': datetime.now().isoformat(), 'short': False},
                            {'title': 'Details', 'value': message, 'short': False}
                        ]
                    }]
                }
                
                response = requests.post(webhook_url, json=payload, timeout=10)
                return response.status_code == 200
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Slack notification failed: {e}")
                return False
        
        return send_slack
    
    @staticmethod
    def webhook_handler(webhook_url: str, headers: Dict[str, str] = None) -> Callable:
        """Create generic webhook notification handler."""
        def send_webhook(alert_name: str, message: str, severity: str) -> bool:
            try:
                payload = {
                    'alert_name': alert_name,
                    'message': message,
                    'severity': severity,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'discord-monitor'
                }
                
                response = requests.post(
                    webhook_url, 
                    json=payload, 
                    headers=headers or {}, 
                    timeout=10
                )
                return 200 <= response.status_code < 300
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Webhook notification failed: {e}")
                return False
        
        return send_webhook


class PerformanceProfiler:
    """Performance profiling and debugging tools."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.active_traces = {}
        self.lock = threading.Lock()
    
    def start_trace(self, name: str, tags: Dict[str, str] = None) -> str:
        """Start a performance trace."""
        trace_id = f"{name}_{int(time.time() * 1000000)}"
        
        with self.lock:
            self.active_traces[trace_id] = {
                'name': name,
                'start_time': time.time(),
                'tags': tags or {}
            }
        
        return trace_id
    
    def end_trace(self, trace_id: str):
        """End a performance trace and record metrics."""
        with self.lock:
            if trace_id in self.active_traces:
                trace_data = self.active_traces.pop(trace_id)
                duration = time.time() - trace_data['start_time']
                
                self.metrics.timing(
                    f"performance.{trace_data['name']}.duration",
                    duration,
                    trace_data['tags']
                )
    
    def trace(self, name: str, tags: Dict[str, str] = None):
        """Context manager for performance tracing."""
        class TraceContext:
            def __init__(self, profiler, trace_name, trace_tags):
                self.profiler = profiler
                self.name = trace_name
                self.tags = trace_tags
                self.trace_id = None
            
            def __enter__(self):
                self.trace_id = self.profiler.start_trace(self.name, self.tags)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.trace_id:
                    self.profiler.end_trace(self.trace_id)
        
        return TraceContext(self, name, tags)


class MonitoringSystem:
    """Central monitoring system coordinator."""
    
    def __init__(self, log_dir: str = "logs", config_file: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Core components
        self.metrics = MetricsCollector()
        self.system_monitor = SystemMonitor(self.metrics)
        self.alert_manager = AlertManager(self.metrics)
        self.profiler = PerformanceProfiler(self.metrics)
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        if config_file and os.path.exists(config_file):
            self._load_config(config_file)
        
        # Setup default alerts
        self._setup_default_alerts()
        
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.monitor_thread = None
    
    def _setup_logging(self):
        """Setup comprehensive logging configuration."""
        # Create custom logger for the application
        app_logger = logging.getLogger('discord_bot')
        app_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        app_logger.handlers.clear()
        
        # Console handler with human-readable format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        app_logger.addHandler(console_handler)
        
        # File handler with JSON format for structured logs
        log_file = self.log_dir / "discord_monitor.jsonl"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=100*1024*1024, backupCount=5  # 100MB per file, 5 backups
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        app_logger.addHandler(file_handler)
        
        # Error-only file handler
        error_file = self.log_dir / "errors.jsonl"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file, maxBytes=50*1024*1024, backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        app_logger.addHandler(error_handler)
    
    def _load_config(self, config_file: str):
        """Load monitoring configuration from file."""
        try:
            with open(config_file) as f:
                config = json.load(f)
            
            # Load alert configurations
            for alert_config in config.get('alerts', []):
                # This would need to be implemented based on your specific needs
                pass
                
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load monitoring config: {e}")
    
    def _setup_default_alerts(self):
        """Setup default system alerts."""
        # High memory usage
        self.alert_manager.add_alert(AlertConfig(
            name="high_memory_usage",
            condition=lambda metrics: (
                metrics.get_latest_value("system.memory.usage_percent") or 0
            ) > 85,
            message_template="High memory usage: {system.memory.usage_percent}%",
            cooldown_minutes=10,
            severity="warning",
            channels=["log"]
        ))
        
        # High CPU usage
        self.alert_manager.add_alert(AlertConfig(
            name="high_cpu_usage",
            condition=lambda metrics: (
                metrics.get_latest_value("system.cpu.usage_percent") or 0
            ) > 90,
            message_template="High CPU usage: {system.cpu.usage_percent}%",
            cooldown_minutes=5,
            severity="warning",
            channels=["log"]
        ))
        
        # Discord API errors
        self.alert_manager.add_alert(AlertConfig(
            name="discord_api_errors",
            condition=lambda metrics: (
                metrics.get_aggregates("discord.api.errors", 5).get('count', 0)
            ) > 10,
            message_template="High Discord API error rate: {count} errors in 5 minutes",
            cooldown_minutes=15,
            severity="error",
            channels=["log"]
        ))
        
        # Database errors
        self.alert_manager.add_alert(AlertConfig(
            name="database_errors",
            condition=lambda metrics: (
                metrics.get_aggregates("database.errors", 5).get('count', 0)
            ) > 5,
            message_template="Database error spike: {count} errors in 5 minutes",
            cooldown_minutes=10,
            severity="error",
            channels=["log"]
        ))
    
    def start(self):
        """Start the monitoring system."""
        if self.running:
            return
        
        self.running = True
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        # Start alert checking
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Monitoring system started")
    
    def stop(self):
        """Stop the monitoring system."""
        self.running = False
        
        self.system_monitor.stop_monitoring()
        
        if self.monitor_thread:
            self.monitor_thread.join()
        
        self.logger.info("Monitoring system stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop for alerts and health checks."""
        while self.running:
            try:
                # Check alerts
                self.alert_manager.check_alerts()
                
                # Record monitoring heartbeat
                self.metrics.increment("monitoring.heartbeat")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(30)  # Wait before retrying
    
    def log_exception(self, exc: Exception, context: Dict[str, Any] = None):
        """Log exception with additional context."""
        logger = logging.getLogger('discord_bot.exceptions')
        
        extra_data = {
            'exception_type': type(exc).__name__,
            'exception_message': str(exc)
        }
        
        if isinstance(exc, DiscordMonitorException):
            extra_data.update(exc.to_dict())
        
        if context:
            extra_data['context'] = context
        
        logger.error(f"Exception occurred: {exc}", exc_info=True, extra=extra_data)
        self.metrics.increment("exceptions.total", {"type": type(exc).__name__})
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'total_datapoints': sum(len(metrics) for metrics in self.metrics.metrics.values()),
                'active_metrics': len(self.metrics.metrics)
            },
            'alerts': {
                'total_configs': len(self.alert_manager.alert_configs),
                'active_alerts': sum(1 for state in self.alert_manager.alert_state.values() if state['active'])
            },
            'system': {
                'monitoring_active': self.running,
                'system_monitor_active': self.system_monitor.monitoring
            },
            'recent_metrics': self.metrics.get_metrics(since=datetime.now() - timedelta(minutes=5))
        }


# Global monitoring instance
monitoring_system: Optional[MonitoringSystem] = None

def get_monitoring_system(log_dir: str = "logs") -> MonitoringSystem:
    """Get or create the global monitoring system."""
    global monitoring_system
    if monitoring_system is None:
        monitoring_system = MonitoringSystem(log_dir)
    return monitoring_system