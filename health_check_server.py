"""
Health check HTTP server for Discord monitoring service.

Provides health check endpoints for monitoring service status,
metrics, and troubleshooting in production environments.
"""

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, Any

from enhanced_main import DiscordMonitoringService


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoints."""
    
    def __init__(self, service: DiscordMonitoringService, *args, **kwargs):
        self.service = service
        self.logger = logging.getLogger(__name__ + '.handler')
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        try:
            parsed_path = urlparse(self.path)
            path = parsed_path.path
            query_params = parse_qs(parsed_path.query)
            
            # Route requests
            if path == '/health':
                self._handle_health_check()
            elif path == '/health/detailed':
                self._handle_detailed_health_check()
            elif path == '/metrics':
                self._handle_metrics()
            elif path == '/status':
                self._handle_status()
            elif path == '/config':
                self._handle_config()
            else:
                self._send_error(404, "Not Found")
                
        except Exception as e:
            self.logger.error(f"Error handling request: {e}")
            self._send_error(500, "Internal Server Error")
    
    def _handle_health_check(self):
        """Handle basic health check."""
        try:
            if not self.service.running:
                self._send_json_response({'status': 'unhealthy', 'reason': 'service not running'}, 503)
                return
            
            # Check core components
            issues = []
            
            # Check Discord client
            if not self.service.discord_client or not self.service.discord_client.connection_healthy:
                issues.append('discord_connection_unhealthy')
            
            # Check database
            if not self.service.database:
                issues.append('database_not_initialized')
            else:
                db_health = self.service.database.health_monitor.check_integrity()
                if not db_health.get('integrity_ok', False):
                    issues.append('database_integrity_issues')
            
            # Check processing manager
            if not self.service.processing_manager:
                issues.append('processing_manager_not_initialized')
            else:
                processing_health = self.service.processing_manager.get_health_status()
                if not processing_health['overall_healthy']:
                    issues.append('processing_pipelines_unhealthy')
            
            if issues:
                response = {
                    'status': 'unhealthy',
                    'issues': issues,
                    'timestamp': time.time()
                }
                self._send_json_response(response, 503)
            else:
                response = {
                    'status': 'healthy',
                    'timestamp': time.time()
                }
                self._send_json_response(response, 200)
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._send_json_response({'status': 'error', 'message': str(e)}, 500)
    
    def _handle_detailed_health_check(self):
        """Handle detailed health check with component status."""
        try:
            status = self.service.get_status()
            
            # Add health analysis
            health_analysis = {
                'overall_healthy': status['running'],
                'components': {}
            }
            
            # Discord client health
            discord_health = status.get('discord_client_health', {})
            health_analysis['components']['discord'] = {
                'healthy': discord_health.get('healthy', False),
                'consecutive_failures': discord_health.get('consecutive_failures', 0),
                'time_since_last_success': discord_health.get('time_since_last_success', 0)
            }
            
            # Database health
            db_stats = status.get('database_stats', {})
            db_health = db_stats.get('health', {})
            health_analysis['components']['database'] = {
                'healthy': db_health.get('integrity_ok', False),
                'integrity_result': db_health.get('integrity_result', ''),
                'foreign_key_violations': db_health.get('foreign_key_violations', 0)
            }
            
            # Processing health
            processing_status = status.get('processing_status', {})
            health_analysis['components']['processing'] = {
                'healthy': processing_status.get('totals', {}).get('active_workers', 0) > 0,
                'total_pipelines': processing_status.get('totals', {}).get('pipelines_count', 0),
                'active_workers': processing_status.get('totals', {}).get('active_workers', 0),
                'queue_size': processing_status.get('totals', {}).get('queue_size', 0)
            }
            
            # LLM health
            llm_usage = status.get('llm_usage', {})
            health_analysis['components']['llm'] = {
                'healthy': not llm_usage.get('cost_limit_exceeded', True),
                'cost_limit_remaining': llm_usage.get('cost_limit_remaining', 0),
                'requests_today': llm_usage.get('total_requests_today', 0)
            }
            
            # Overall health determination
            component_health = [comp['healthy'] for comp in health_analysis['components'].values()]
            health_analysis['overall_healthy'] = all(component_health) and status['running']
            
            response = {
                'health': health_analysis,
                'detailed_status': status,
                'timestamp': time.time()
            }
            
            status_code = 200 if health_analysis['overall_healthy'] else 503
            self._send_json_response(response, status_code)
            
        except Exception as e:
            self.logger.error(f"Detailed health check failed: {e}")
            self._send_json_response({'status': 'error', 'message': str(e)}, 500)
    
    def _handle_metrics(self):
        """Handle metrics endpoint."""
        try:
            if not self.service.monitoring:
                self._send_json_response({'error': 'monitoring not available'}, 503)
                return
            
            # Get recent metrics (last 5 minutes)
            from datetime import datetime, timedelta
            since = datetime.now() - timedelta(minutes=5)
            metrics = self.service.monitoring.metrics.get_metrics(since=since)
            
            # Format for JSON response
            formatted_metrics = {}
            for metric_name, datapoints in metrics.items():
                formatted_metrics[metric_name] = [
                    {
                        'value': dp.value,
                        'timestamp': dp.timestamp.isoformat(),
                        'tags': dp.tags,
                        'unit': dp.unit
                    }
                    for dp in datapoints
                ]
            
            self._send_json_response({
                'metrics': formatted_metrics,
                'timestamp': time.time(),
                'time_range': '5_minutes'
            })
            
        except Exception as e:
            self.logger.error(f"Metrics endpoint failed: {e}")
            self._send_json_response({'error': str(e)}, 500)
    
    def _handle_status(self):
        """Handle status endpoint."""
        try:
            status = self.service.get_status()
            self._send_json_response(status)
            
        except Exception as e:
            self.logger.error(f"Status endpoint failed: {e}")
            self._send_json_response({'error': str(e)}, 500)
    
    def _handle_config(self):
        """Handle configuration endpoint (sanitized)."""
        try:
            # Return sanitized configuration (remove sensitive data)
            config_dict = self.service.config.to_dict()
            
            # Remove sensitive fields
            sensitive_fields = [
                'discord.token',
                'llm.google_api_key',
                'llm.openai_api_key', 
                'llm.openrouter_api_key',
                'alerts.email_password',
                'alerts.slack_webhook_url',
                'alerts.webhook_url'
            ]
            
            for field_path in sensitive_fields:
                parts = field_path.split('.')
                current = config_dict
                for part in parts[:-1]:
                    if part in current and isinstance(current[part], dict):
                        current = current[part]
                    else:
                        break
                else:
                    if parts[-1] in current:
                        current[parts[-1]] = '***REDACTED***' if current[parts[-1]] else None
            
            self._send_json_response({
                'config': config_dict,
                'timestamp': time.time()
            })
            
        except Exception as e:
            self.logger.error(f"Config endpoint failed: {e}")
            self._send_json_response({'error': str(e)}, 500)
    
    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response_data = json.dumps(data, indent=2, default=str)
        self.wfile.write(response_data.encode('utf-8'))
    
    def _send_error(self, status_code: int, message: str):
        """Send error response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        error_data = {
            'error': message,
            'status_code': status_code,
            'timestamp': time.time()
        }
        
        self.wfile.write(json.dumps(error_data).encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to use proper logging."""
        self.logger.info(format % args)


class HealthCheckServer:
    """Health check HTTP server."""
    
    def __init__(self, service: DiscordMonitoringService, port: int = 8080):
        self.service = service
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the health check server."""
        if self.running:
            return
        
        try:
            # Create handler class with service reference
            def handler_factory(*args, **kwargs):
                return HealthCheckHandler(self.service, *args, **kwargs)
            
            # Create server
            self.server = HTTPServer(('0.0.0.0', self.port), handler_factory)
            
            # Start in separate thread
            self.server_thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True,
                name=f"HealthCheckServer-{self.port}"
            )
            
            self.running = True
            self.server_thread.start()
            
            self.logger.info(f"Health check server started on port {self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start health check server: {e}")
            raise
    
    def stop(self):
        """Stop the health check server."""
        if not self.running:
            return
        
        try:
            self.running = False
            
            if self.server:
                self.server.shutdown()
                self.server.server_close()
            
            if self.server_thread:
                self.server_thread.join(timeout=5.0)
            
            self.logger.info("Health check server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping health check server: {e}")


if __name__ == "__main__":
    # For testing the health check server independently
    import sys
    import time
    from unittest.mock import Mock
    
    # Create mock service for testing
    mock_service = Mock()
    mock_service.running = True
    mock_service.config = Mock()
    mock_service.config.to_dict.return_value = {'test': 'config'}
    mock_service.get_status.return_value = {'status': 'ok'}
    
    # Start server
    server = HealthCheckServer(mock_service, 8080)
    server.start()
    
    print("Health check server running on http://localhost:8080")
    print("Test endpoints:")
    print("  http://localhost:8080/health")
    print("  http://localhost:8080/health/detailed") 
    print("  http://localhost:8080/status")
    print("  http://localhost:8080/metrics")
    print("  http://localhost:8080/config")
    print("\nPress Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()
        print("Server stopped")