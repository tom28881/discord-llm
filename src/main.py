"""
Main application entry point for the Discord Real-time Personal Monitor.
This module orchestrates all services and provides the main application lifecycle.
"""
import asyncio
import logging
import sys
import signal
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
import traceback
from dataclasses import asdict
import json

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from core.services.config_service import ConfigurationService
from core.services.importance_analyzer import ImportanceAnalyzerService
from core.services.message_processor import MessageProcessingService
from core.plugins.base import PluginManager
from infrastructure.repositories.sqlite_repository import SqliteRepositoryManager
from infrastructure.services.discord_service import DiscordStreamingService
from infrastructure.services.notification_service import NotificationService
from infrastructure.services.llm_service import LLMService
from infrastructure.services.event_service import EventService
from infrastructure.logging.logger import setup_logging, get_logger
from infrastructure.health.health_monitor import HealthMonitor

logger = get_logger(__name__)


class DiscordMonitorApplication:
    """
    Main application class that orchestrates all services and manages lifecycle.
    
    This class follows the dependency injection pattern and provides:
    - Service initialization and shutdown
    - Error handling and recovery
    - Health monitoring
    - Graceful shutdown handling
    - Configuration management
    """
    
    def __init__(self, config_file_path: Optional[str] = None):
        self.config_file_path = config_file_path or "config/app_config.yaml"
        
        # Core services
        self.config_service: Optional[ConfigurationService] = None
        self.repository_manager: Optional[SqliteRepositoryManager] = None
        self.plugin_manager: Optional[PluginManager] = None
        
        # Business services
        self.importance_analyzer: Optional[ImportanceAnalyzerService] = None
        self.message_processor: Optional[MessageProcessingService] = None
        self.discord_service: Optional[DiscordStreamingService] = None
        self.notification_service: Optional[NotificationService] = None
        self.llm_service: Optional[LLMService] = None
        self.event_service: Optional[EventService] = None
        
        # Infrastructure services
        self.health_monitor: Optional[HealthMonitor] = None
        
        # Application state
        self._shutdown_event = asyncio.Event()
        self._initialization_complete = False
        self._running_tasks: Dict[str, asyncio.Task] = {}
        
    async def initialize(self) -> bool:
        """
        Initialize all application services in the correct order.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("🚀 Starting Discord Real-time Personal Monitor")
            
            # Step 1: Initialize repository layer
            if not await self._initialize_repositories():
                return False
            
            # Step 2: Initialize configuration service
            if not await self._initialize_configuration():
                return False
            
            # Step 3: Initialize core services
            if not await self._initialize_core_services():
                return False
            
            # Step 4: Initialize business services
            if not await self._initialize_business_services():
                return False
            
            # Step 5: Initialize plugin system
            if not await self._initialize_plugins():
                return False
            
            # Step 6: Initialize health monitoring
            if not await self._initialize_health_monitoring():
                return False
            
            # Step 7: Start background tasks
            await self._start_background_tasks()
            
            self._initialization_complete = True
            logger.info("✅ Discord Monitor initialization complete")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Discord Monitor: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            await self.shutdown()
            return False
    
    async def _initialize_repositories(self) -> bool:
        """Initialize data repositories."""
        try:
            logger.info("📊 Initializing database repositories...")
            
            # Get database path from environment or use default
            import os
            db_path = os.getenv("DATABASE_PATH", "data/discord_monitor.db")
            
            self.repository_manager = SqliteRepositoryManager(db_path)
            
            # Initialize all database tables
            success = await self.repository_manager.initialize_all_tables()
            if not success:
                raise Exception("Failed to initialize database tables")
            
            logger.info("✅ Database repositories initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize repositories: {e}")
            return False
    
    async def _initialize_configuration(self) -> bool:
        """Initialize configuration service."""
        try:
            logger.info("⚙️ Initializing configuration service...")
            
            self.config_service = ConfigurationService(
                config_repository=self.repository_manager.config_repository,
                config_file_path=self.config_file_path,
                enable_hot_reload=True
            )
            
            success = await self.config_service.initialize()
            if not success:
                raise Exception("Failed to initialize configuration service")
            
            # Update log level based on configuration
            log_level = await self.config_service.get_config("system.log_level", "INFO")
            setup_logging(level=log_level)
            
            logger.info("✅ Configuration service initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize configuration: {e}")
            return False
    
    async def _initialize_core_services(self) -> bool:
        """Initialize core services."""
        try:
            logger.info("🔧 Initializing core services...")
            
            # Initialize LLM service
            self.llm_service = LLMService(
                config_service=self.config_service
            )
            
            if not await self.llm_service.initialize():
                logger.warning("⚠️ LLM service initialization failed - some features will be disabled")
            
            # Initialize event service
            self.event_service = EventService(
                event_repository=self.repository_manager.event_repository,
                config_service=self.config_service
            )
            
            if not await self.event_service.initialize():
                raise Exception("Failed to initialize event service")
            
            logger.info("✅ Core services initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize core services: {e}")
            return False
    
    async def _initialize_business_services(self) -> bool:
        """Initialize business logic services."""
        try:
            logger.info("🧠 Initializing business services...")
            
            # Initialize importance analyzer
            self.importance_analyzer = ImportanceAnalyzerService(
                config_repository=self.repository_manager.config_repository
            )
            
            # Initialize notification service
            self.notification_service = NotificationService(
                config_service=self.config_service
            )
            
            if not await self.notification_service.initialize():
                logger.warning("⚠️ Notification service initialization failed")
            
            # Initialize message processor
            self.message_processor = MessageProcessingService(
                message_repository=self.repository_manager.message_repository,
                event_repository=self.repository_manager.event_repository,
                importance_analyzer=self.importance_analyzer,
                event_service=self.event_service,
                notification_service=self.notification_service,
                llm_service=self.llm_service
            )
            
            # Initialize Discord service
            self.discord_service = DiscordStreamingService(
                config_service=self.config_service,
                server_repository=self.repository_manager.server_repository,
                channel_repository=self.repository_manager.channel_repository,
                user_repository=self.repository_manager.user_repository,
                message_processor=self.message_processor
            )
            
            if not await self.discord_service.initialize():
                raise Exception("Failed to initialize Discord service")
            
            logger.info("✅ Business services initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize business services: {e}")
            return False
    
    async def _initialize_plugins(self) -> bool:
        """Initialize plugin system."""
        try:
            # Check if plugin system is enabled
            plugins_enabled = await self.config_service.get_feature_flag("plugin_system")
            if not plugins_enabled:
                logger.info("🔌 Plugin system disabled by feature flag")
                return True
            
            logger.info("🔌 Initializing plugin system...")
            
            self.plugin_manager = PluginManager()
            
            # Add plugin directories
            plugin_dirs = [
                "src/core/plugins",
                "plugins",  # External plugins directory
                "config/plugins"  # User plugins directory
            ]
            
            for plugin_dir in plugin_dirs:
                self.plugin_manager.add_plugin_directory(plugin_dir)
            
            # Discover and load plugins
            loaded_plugins = await self.plugin_manager.discover_and_load_plugins()
            logger.info(f"📦 Loaded {len(loaded_plugins)} plugins")
            
            # Initialize plugins with configuration
            plugin_config = await self.config_service.get_config("plugins", {})
            initialization_results = await self.plugin_manager.registry.initialize_all_plugins(plugin_config)
            
            successful_plugins = sum(1 for success in initialization_results.values() if success)
            logger.info(f"✅ Initialized {successful_plugins}/{len(initialization_results)} plugins")
            
            # Register notification channels with notification service
            if self.notification_service and self.plugin_manager:
                notification_channels = await self.plugin_manager.get_notification_channels()
                for channel in notification_channels:
                    await self.notification_service.register_plugin_channel(channel)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize plugins: {e}")
            return False
    
    async def _initialize_health_monitoring(self) -> bool:
        """Initialize health monitoring."""
        try:
            logger.info("🏥 Initializing health monitoring...")
            
            self.health_monitor = HealthMonitor(
                config_service=self.config_service,
                discord_service=self.discord_service,
                repository_manager=self.repository_manager,
                llm_service=self.llm_service,
                plugin_manager=self.plugin_manager
            )
            
            if not await self.health_monitor.initialize():
                logger.warning("⚠️ Health monitoring initialization failed")
                return True  # Non-critical, continue anyway
            
            logger.info("✅ Health monitoring initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize health monitoring: {e}")
            return True  # Non-critical
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks."""
        try:
            logger.info("🔄 Starting background tasks...")
            
            # Start real-time monitoring if enabled
            real_time_enabled = await self.config_service.get_feature_flag("real_time_monitoring")
            if real_time_enabled and self.discord_service:
                self._running_tasks["discord_monitor"] = asyncio.create_task(
                    self._discord_monitoring_task(),
                    name="discord_monitor"
                )
                logger.info("📡 Real-time Discord monitoring started")
            
            # Start health monitoring task
            if self.health_monitor:
                self._running_tasks["health_monitor"] = asyncio.create_task(
                    self._health_monitoring_task(),
                    name="health_monitor"
                )
                logger.info("🏥 Health monitoring task started")
            
            # Start message processing task for backlog
            self._running_tasks["message_processor"] = asyncio.create_task(
                self._message_processing_task(),
                name="message_processor"
            )
            logger.info("⚙️ Message processing task started")
            
            logger.info("✅ Background tasks started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start background tasks: {e}")
    
    async def _discord_monitoring_task(self) -> None:
        """Background task for real-time Discord monitoring."""
        try:
            async for message in self.discord_service.start_real_time_monitoring():
                try:
                    # Process message through the pipeline
                    result = await self.message_processor.process_message(message)
                    
                    if not result.success:
                        logger.warning(f"Failed to process message {message.id}: {result.error_message}")
                    else:
                        logger.debug(f"Processed message {message.id} (importance: {result.importance.level.value})")
                        
                except Exception as e:
                    logger.error(f"Error processing real-time message {message.id}: {e}")
                
                # Check for shutdown
                if self._shutdown_event.is_set():
                    break
                    
        except Exception as e:
            logger.error(f"Discord monitoring task failed: {e}")
            # Attempt to restart after delay
            if not self._shutdown_event.is_set():
                logger.info("Restarting Discord monitoring in 30 seconds...")
                await asyncio.sleep(30)
                if not self._shutdown_event.is_set():
                    self._running_tasks["discord_monitor"] = asyncio.create_task(
                        self._discord_monitoring_task(),
                        name="discord_monitor_restart"
                    )
    
    async def _health_monitoring_task(self) -> None:
        """Background task for health monitoring."""
        try:
            check_interval = await self.config_service.get_config("system.health_check_interval", 300)
            
            while not self._shutdown_event.is_set():
                try:
                    health_status = await self.health_monitor.get_overall_health()
                    
                    if not health_status.get("healthy", True):
                        logger.warning(f"Health check failed: {health_status}")
                        
                        # Send health alert if notification service is available
                        if self.notification_service:
                            await self.notification_service.send_health_alert(health_status)
                    
                    await asyncio.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(60)  # Shorter retry interval on error
                    
        except Exception as e:
            logger.error(f"Health monitoring task failed: {e}")
    
    async def _message_processing_task(self) -> None:
        """Background task for processing message backlog."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Get unprocessed messages
                    unprocessed = await self.repository_manager.message_repository.get_unprocessed_messages(50)
                    
                    if unprocessed:
                        logger.info(f"Processing {len(unprocessed)} unprocessed messages")
                        
                        # Process in batches
                        results = await self.message_processor.process_message_batch(unprocessed)
                        
                        successful = sum(1 for r in results if r.success)
                        logger.info(f"Processed {successful}/{len(results)} messages successfully")
                    
                    # Sleep before next batch
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Message processing task error: {e}")
                    await asyncio.sleep(30)  # Shorter retry interval on error
                    
        except Exception as e:
            logger.error(f"Message processing task failed: {e}")
    
    async def run(self) -> int:
        """
        Run the application until shutdown is requested.
        
        Returns:
            int: Exit code (0 for success, 1 for error)
        """
        try:
            if not self._initialization_complete:
                logger.error("Application not properly initialized")
                return 1
            
            logger.info("🎯 Discord Monitor is now running...")
            logger.info("Press Ctrl+C to shutdown gracefully")
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
            logger.info("🛑 Shutdown requested")
            return 0
            
        except KeyboardInterrupt:
            logger.info("🛑 Received keyboard interrupt")
            return 0
        except Exception as e:
            logger.error(f"❌ Application runtime error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 1
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown all services.
        """
        try:
            logger.info("🔄 Starting graceful shutdown...")
            
            # Signal shutdown to all tasks
            self._shutdown_event.set()
            
            # Cancel background tasks
            for task_name, task in self._running_tasks.items():
                if not task.done():
                    logger.info(f"Cancelling task: {task_name}")
                    task.cancel()
                    
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        logger.warning(f"Task {task_name} did not shutdown gracefully")
            
            # Shutdown services in reverse order
            services_to_shutdown = [
                ("Health Monitor", self.health_monitor),
                ("Plugin Manager", self.plugin_manager),
                ("Discord Service", self.discord_service),
                ("Notification Service", self.notification_service),
                ("Event Service", self.event_service),
                ("LLM Service", self.llm_service),
                ("Configuration Service", self.config_service)
            ]
            
            for service_name, service in services_to_shutdown:
                if service:
                    try:
                        logger.info(f"Shutting down {service_name}...")
                        if hasattr(service, 'shutdown'):
                            await service.shutdown()
                        logger.info(f"✅ {service_name} shutdown complete")
                    except Exception as e:
                        logger.error(f"❌ Error shutting down {service_name}: {e}")
            
            logger.info("✅ Graceful shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            if not self._shutdown_event.is_set():
                asyncio.create_task(self._handle_shutdown_signal())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _handle_shutdown_signal(self) -> None:
        """Handle shutdown signal."""
        if not self._shutdown_event.is_set():
            self._shutdown_event.set()


async def main() -> int:
    """Main application entry point."""
    try:
        # Setup basic logging first
        setup_logging()
        
        # Create and initialize application
        app = DiscordMonitorApplication()
        
        # Setup signal handlers
        app.setup_signal_handlers()
        
        # Initialize all services
        if not await app.initialize():
            logger.error("❌ Failed to initialize application")
            return 1
        
        # Run application
        exit_code = await app.run()
        
        # Shutdown gracefully
        await app.shutdown()
        
        return exit_code
        
    except Exception as e:
        logger.error(f"❌ Critical application error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Application interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)