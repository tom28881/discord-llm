"""
Enhanced main application with comprehensive error handling and production features.

Integrates all resilience components for reliable 24/7 Discord monitoring.
"""

import os
import sys
import signal
import argparse
import asyncio
import threading
import time
import logging
from typing import Optional
from datetime import datetime

from production_config import ConfigurationManager, ProductionConfig
from lib.resilient_discord_client import EnhancedDiscordClient
from lib.resilient_llm import get_enhanced_llm_client
from lib.resilient_database import EnhancedDatabase, DatabaseConfig
from lib.monitoring import get_monitoring_system, NotificationChannels
from lib.processing_manager import ProcessingManager, ProcessingPipeline, ProcessingTask
from lib.exceptions import (
    DiscordMonitorException, DiscordAPIError, DiscordForbiddenError,
    DatabaseError, ConfigurationError
)
from lib.config_manager import load_config, load_forbidden_channels, add_forbidden_channel


class DiscordMonitoringService:
    """Main Discord monitoring service with comprehensive error handling."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Core components
        self.discord_client: Optional[EnhancedDiscordClient] = None
        self.database: Optional[EnhancedDatabase] = None
        self.llm_client = None
        self.monitoring = None
        self.processing_manager = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self._initialize_components()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup application logging."""
        # Create logger
        logger = logging.getLogger('discord_monitor_service')
        logger.setLevel(getattr(logging, self.config.monitoring.log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        if self.config.monitoring.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_components(self):
        """Initialize all service components."""
        try:
            self.logger.info("Initializing Discord monitoring service components...")
            
            # Initialize monitoring system
            self.monitoring = get_monitoring_system(self.config.monitoring.log_dir)
            self.monitoring.start()
            
            # Initialize database
            db_config = DatabaseConfig(
                db_path=self.config.database.path,
                connection_timeout=self.config.database.connection_timeout,
                max_retries=self.config.database.max_retries,
                backup_interval_hours=self.config.database.backup_interval_hours,
                max_backups=self.config.database.max_backups,
                wal_mode=self.config.database.wal_mode,
                synchronous=self.config.database.synchronous,
                journal_size_limit=self.config.database.journal_size_limit,
                auto_vacuum=self.config.database.auto_vacuum,
                page_size=self.config.database.page_size
            )
            
            self.database = EnhancedDatabase(db_config)
            self.database.start_maintenance()
            
            # Initialize Discord client
            self.discord_client = EnhancedDiscordClient(
                token=self.config.discord.token,
                max_retries=self.config.discord.max_retries
            )
            
            # Initialize LLM client
            self.llm_client = get_enhanced_llm_client(self.config.llm.daily_cost_limit)
            
            # Initialize processing manager
            self.processing_manager = ProcessingManager()
            
            # Setup processing pipelines
            self._setup_processing_pipelines()
            
            # Setup monitoring alerts
            self._setup_monitoring_alerts()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            self.monitoring.log_exception(e, {'component': 'initialization'})
            raise
    
    def _setup_processing_pipelines(self):
        """Setup processing pipelines for different tasks."""
        # Message processing pipeline
        message_pipeline = ProcessingPipeline(
            name="message_processing",
            processor_func=self._process_message_task,
            max_queue_size=self.config.processing.max_queue_size,
            worker_count=self.config.processing.worker_count
        )
        
        self.processing_manager.add_pipeline(message_pipeline)
        
        # Server/channel discovery pipeline
        discovery_pipeline = ProcessingPipeline(
            name="discovery",
            processor_func=self._process_discovery_task,
            max_queue_size=1000,
            worker_count=2
        )
        
        self.processing_manager.add_pipeline(discovery_pipeline)
        
        self.logger.info("Processing pipelines configured")
    
    def _setup_monitoring_alerts(self):
        """Setup monitoring alerts and notification channels."""
        # Add notification channels
        if self.config.alerts.email_enabled:
            email_handler = NotificationChannels.email_handler(
                smtp_host=self.config.alerts.email_smtp_host,
                smtp_port=self.config.alerts.email_smtp_port,
                username=self.config.alerts.email_username,
                password=self.config.alerts.email_password,
                from_addr=self.config.alerts.email_from,
                to_addrs=self.config.alerts.email_to
            )
            self.monitoring.alert_manager.add_notification_channel("email", email_handler)
        
        if self.config.alerts.slack_enabled:
            slack_handler = NotificationChannels.slack_webhook_handler(
                webhook_url=self.config.alerts.slack_webhook_url
            )
            self.monitoring.alert_manager.add_notification_channel("slack", slack_handler)
        
        if self.config.alerts.webhook_enabled:
            webhook_handler = NotificationChannels.webhook_handler(
                webhook_url=self.config.alerts.webhook_url,
                headers=self.config.alerts.webhook_headers
            )
            self.monitoring.alert_manager.add_notification_channel("webhook", webhook_handler)
        
        self.logger.info("Monitoring alerts configured")
    
    def _process_message_task(self, task: ProcessingTask) -> dict:
        """Process a message fetching task."""
        try:
            server_id = task.data['server_id']
            server_name = task.data['server_name']
            
            self.monitoring.profiler.start_trace("fetch_messages", {"server_id": str(server_id)})
            
            # Save server information
            self.database.save_server(server_id, server_name)
            
            # Get channel information
            self.discord_client.server_id = str(server_id)
            channel_info = self.discord_client.get_channel_ids()
            
            # Load forbidden channels
            config = load_config()
            forbidden_channels = load_forbidden_channels(config)
            
            messages_processed = 0
            channels_processed = 0
            
            for channel_id, channel_name in channel_info:
                if channel_id in forbidden_channels:
                    self.logger.info(f"Skipping forbidden channel: #{channel_name}")
                    continue
                
                try:
                    # Save channel information
                    self.database.save_channel(channel_id, server_id, channel_name)
                    
                    # Get last message ID for incremental fetching
                    last_message_id = self.database.get_last_message_id(server_id, channel_id)
                    
                    # Fetch messages
                    new_messages = self.discord_client.fetch_messages(
                        channel_id, 
                        last_message_id, 
                        self.config.discord.max_message_batch_size
                    )
                    
                    if new_messages:
                        # Convert to database format
                        messages_to_save = [
                            (server_id, channel_id, message_id, content, sent_at)
                            for message_id, content, sent_at in new_messages
                        ]
                        
                        # Save messages
                        saved_count = self.database.save_messages(messages_to_save)
                        messages_processed += saved_count
                        
                        self.logger.info(f"Saved {saved_count} messages from #{channel_name}")
                        
                        # Record metrics
                        self.monitoring.metrics.increment("messages.saved", {"server": str(server_id)})
                        self.monitoring.metrics.gauge("messages.batch_size", saved_count)
                    
                    channels_processed += 1
                    
                    # Delay between channels
                    time.sleep(self.config.discord.channel_delay_seconds)
                    
                except DiscordForbiddenError as e:
                    self.logger.warning(f"Access forbidden to channel #{channel_name}: {e}")
                    add_forbidden_channel(config, channel_id)
                    self.monitoring.metrics.increment("channels.forbidden")
                    
                except Exception as e:
                    self.logger.error(f"Error processing channel #{channel_name}: {e}")
                    self.monitoring.log_exception(e, {
                        'server_id': server_id,
                        'channel_id': channel_id,
                        'channel_name': channel_name
                    })
                    continue
            
            return {
                'server_id': server_id,
                'server_name': server_name,
                'channels_processed': channels_processed,
                'messages_processed': messages_processed,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Message processing task failed: {e}")
            self.monitoring.log_exception(e, task.data)
            raise
        
        finally:
            self.monitoring.profiler.end_trace(task.id)
    
    def _process_discovery_task(self, task: ProcessingTask) -> dict:
        """Process a server/channel discovery task."""
        try:
            # Discover servers
            server_info = self.discord_client.get_server_ids()
            
            servers_discovered = 0
            for server_id, server_name in server_info:
                try:
                    self.database.save_server(server_id, server_name)
                    servers_discovered += 1
                except Exception as e:
                    self.logger.warning(f"Failed to save server {server_name}: {e}")
            
            self.monitoring.metrics.gauge("servers.discovered", servers_discovered)
            
            return {
                'servers_discovered': servers_discovered,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Discovery task failed: {e}")
            self.monitoring.log_exception(e, task.data)
            raise
    
    def start(self):
        """Start the monitoring service."""
        if self.running:
            self.logger.warning("Service already running")
            return
        
        try:
            self.logger.info("Starting Discord monitoring service...")
            self.running = True
            
            # Start processing manager
            self.processing_manager.start_all()
            
            # Start main processing loop
            self._start_main_loop()
            
            self.logger.info("Discord monitoring service started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            self.monitoring.log_exception(e, {'component': 'service_start'})
            self.stop()
            raise
    
    def stop(self):
        """Stop the monitoring service gracefully."""
        if not self.running:
            return
        
        self.logger.info("Stopping Discord monitoring service...")
        self.running = False
        self.shutdown_event.set()
        
        try:
            # Stop processing manager
            if self.processing_manager:
                self.processing_manager.stop_all(self.config.shutdown_timeout_seconds)
            
            # Stop database maintenance
            if self.database:
                self.database.stop_maintenance()
                self.database.close()
            
            # Stop monitoring
            if self.monitoring:
                self.monitoring.stop()
            
            self.logger.info("Discord monitoring service stopped")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def _start_main_loop(self):
        """Start the main processing loop."""
        def main_loop():
            while self.running and not self.shutdown_event.is_set():
                try:
                    # Schedule discovery task
                    discovery_task = ProcessingTask(
                        id=f"discovery_{int(time.time())}",
                        data={}
                    )
                    self.processing_manager.submit_task("discovery", discovery_task)
                    
                    # Get servers and schedule message processing
                    servers = self.database.get_servers()
                    for server_id, server_name in servers:
                        message_task = ProcessingTask(
                            id=f"messages_{server_id}_{int(time.time())}",
                            data={
                                'server_id': server_id,
                                'server_name': server_name
                            }
                        )
                        self.processing_manager.submit_task("message_processing", message_task)
                    
                    # Wait between cycles
                    self.shutdown_event.wait(timeout=self.config.discord.server_delay_seconds)
                    
                except Exception as e:
                    self.logger.error(f"Main loop error: {e}")
                    self.monitoring.log_exception(e, {'component': 'main_loop'})
                    
                    # Wait before retrying
                    self.shutdown_event.wait(timeout=self.config.discord.error_retry_delay_seconds)
        
        # Start in separate thread
        main_thread = threading.Thread(target=main_loop, daemon=True, name="MainLoop")
        main_thread.start()
    
    def get_status(self) -> dict:
        """Get service status information."""
        return {
            'running': self.running,
            'uptime': (datetime.now() - self.monitoring.global_stats['start_time']).total_seconds(),
            'discord_client_health': self.discord_client.get_connection_health() if self.discord_client else {},
            'database_stats': self.database.get_statistics() if self.database else {},
            'processing_status': self.processing_manager.get_global_status() if self.processing_manager else {},
            'llm_usage': self.llm_client.get_usage_stats() if self.llm_client else {},
            'monitoring_health': self.monitoring.get_health_report() if self.monitoring else {}
        }


def setup_signal_handlers(service: DiscordMonitoringService):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger = logging.getLogger('discord_monitor_service')
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced Discord monitoring service")
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--env-file', type=str, help='Environment file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--create-sample-config', action='store_true', 
                       help='Create sample configuration file')
    parser.add_argument('--validate-config', action='store_true', 
                       help='Validate configuration and exit')
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.create_sample_config:
        from production_config import create_sample_config
        create_sample_config()
        return
    
    try:
        # Load configuration
        config_manager = ConfigurationManager(args.config, args.env_file)
        config = config_manager.load_config()
        
        # Override debug mode if specified
        if args.debug:
            config.debug_mode = True
            config.monitoring.log_level = "DEBUG"
        
        # Validate configuration
        if args.validate_config:
            print("Configuration validation passed!")
            return
        
        # Create and start service
        service = DiscordMonitoringService(config)
        setup_signal_handlers(service)
        
        # Start service
        service.start()
        
        # Keep running until shutdown
        try:
            while service.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    except Exception as e:
        logger = logging.getLogger('discord_monitor_service')
        logger.error(f"Service failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()