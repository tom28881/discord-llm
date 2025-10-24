"""
Base plugin system for extensible notification channels and processors.
This provides the foundation for a pluggable architecture.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import importlib
import inspect

from ..domain.models import Message, ImportanceLevel

logger = logging.getLogger(__name__)


class PluginType(Enum):
    NOTIFICATION_CHANNEL = "notification_channel"
    MESSAGE_PROCESSOR = "message_processor"
    IMPORTANCE_ANALYZER = "importance_analyzer"
    EVENT_HANDLER = "event_handler"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class IPlugin(ABC):
    """Base interface for all plugins."""
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Plugin metadata."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Cleanup when plugin is being disabled."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin health status."""
        pass


class INotificationChannelPlugin(IPlugin):
    """Interface for notification channel plugins."""
    
    @abstractmethod
    async def send_notification(
        self, 
        message: str, 
        importance_level: ImportanceLevel,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a notification through this channel."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the notification channel is working."""
        pass
    
    @abstractmethod
    def get_supported_importance_levels(self) -> List[ImportanceLevel]:
        """Get list of importance levels this channel supports."""
        pass


class IMessageProcessorPlugin(IPlugin):
    """Interface for message processor plugins."""
    
    @abstractmethod
    async def process_message(self, message: Message) -> Dict[str, Any]:
        """Process a message and return processing results."""
        pass
    
    @abstractmethod
    def get_processing_priority(self) -> int:
        """Get processing priority (lower number = higher priority)."""
        pass


class IImportanceAnalyzerPlugin(IPlugin):
    """Interface for importance analyzer plugins."""
    
    @abstractmethod
    async def analyze_importance(self, message: Message) -> float:
        """Analyze message importance and return a score (0.0-1.0)."""
        pass
    
    @abstractmethod
    def get_analyzer_weight(self) -> float:
        """Get weight of this analyzer in final score calculation."""
        pass


class IEventHandlerPlugin(IPlugin):
    """Interface for event handler plugins."""
    
    @abstractmethod
    async def handle_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Handle a specific event type."""
        pass
    
    @abstractmethod
    def get_supported_events(self) -> List[str]:
        """Get list of event types this handler supports."""
        pass


@dataclass
class PluginRegistry:
    """Registry for managing plugins."""
    _plugins: Dict[str, IPlugin] = field(default_factory=dict)
    _plugin_types: Dict[PluginType, List[str]] = field(default_factory=dict)
    _enabled_plugins: Dict[str, bool] = field(default_factory=dict)
    
    def register_plugin(self, plugin: IPlugin) -> bool:
        """Register a plugin."""
        try:
            metadata = plugin.metadata
            
            if metadata.name in self._plugins:
                logger.warning(f"Plugin {metadata.name} is already registered")
                return False
            
            self._plugins[metadata.name] = plugin
            self._enabled_plugins[metadata.name] = metadata.enabled
            
            # Group by type
            if metadata.plugin_type not in self._plugin_types:
                self._plugin_types[metadata.plugin_type] = []
            
            self._plugin_types[metadata.plugin_type].append(metadata.name)
            
            logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register plugin: {e}")
            return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin."""
        try:
            if plugin_name not in self._plugins:
                return False
            
            plugin = self._plugins[plugin_name]
            plugin_type = plugin.metadata.plugin_type
            
            # Remove from type grouping
            if plugin_type in self._plugin_types:
                if plugin_name in self._plugin_types[plugin_type]:
                    self._plugin_types[plugin_type].remove(plugin_name)
            
            # Remove from main registry
            del self._plugins[plugin_name]
            del self._enabled_plugins[plugin_name]
            
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister plugin {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[IPlugin]:
        """Get a specific plugin by name."""
        return self._plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[IPlugin]:
        """Get all plugins of a specific type."""
        plugin_names = self._plugin_types.get(plugin_type, [])
        return [self._plugins[name] for name in plugin_names if self.is_plugin_enabled(name)]
    
    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled."""
        return self._enabled_plugins.get(plugin_name, False)
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        if plugin_name in self._plugins:
            self._enabled_plugins[plugin_name] = True
            return True
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        if plugin_name in self._plugins:
            self._enabled_plugins[plugin_name] = False
            return True
        return False
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins with their status."""
        plugins_info = []
        
        for name, plugin in self._plugins.items():
            metadata = plugin.metadata
            plugins_info.append({
                "name": name,
                "version": metadata.version,
                "description": metadata.description,
                "author": metadata.author,
                "type": metadata.plugin_type.value,
                "enabled": self.is_plugin_enabled(name),
                "dependencies": metadata.dependencies
            })
        
        return plugins_info
    
    async def initialize_all_plugins(self, config: Dict[str, Any]) -> Dict[str, bool]:
        """Initialize all enabled plugins."""
        results = {}
        
        for name, plugin in self._plugins.items():
            if not self.is_plugin_enabled(name):
                continue
            
            try:
                plugin_config = config.get(f"plugins.{name}", {})
                success = await plugin.initialize(plugin_config)
                results[name] = success
                
                if success:
                    logger.info(f"Initialized plugin: {name}")
                else:
                    logger.warning(f"Failed to initialize plugin: {name}")
                    
            except Exception as e:
                logger.error(f"Error initializing plugin {name}: {e}")
                results[name] = False
        
        return results
    
    async def shutdown_all_plugins(self) -> Dict[str, bool]:
        """Shutdown all plugins."""
        results = {}
        
        for name, plugin in self._plugins.items():
            try:
                success = await plugin.shutdown()
                results[name] = success
                
                if success:
                    logger.info(f"Shutdown plugin: {name}")
                else:
                    logger.warning(f"Failed to shutdown plugin: {name}")
                    
            except Exception as e:
                logger.error(f"Error shutting down plugin {name}: {e}")
                results[name] = False
        
        return results


class PluginManager:
    """Manager for plugin lifecycle and operations."""
    
    def __init__(self):
        self.registry = PluginRegistry()
        self._plugin_directories: List[str] = []
    
    def add_plugin_directory(self, directory: str) -> None:
        """Add a directory to search for plugins."""
        self._plugin_directories.append(directory)
    
    async def discover_and_load_plugins(self) -> Dict[str, bool]:
        """Discover and load plugins from configured directories."""
        results = {}
        
        for directory in self._plugin_directories:
            try:
                loaded = await self._load_plugins_from_directory(directory)
                results.update(loaded)
            except Exception as e:
                logger.error(f"Failed to load plugins from {directory}: {e}")
        
        return results
    
    async def _load_plugins_from_directory(self, directory: str) -> Dict[str, bool]:
        """Load plugins from a specific directory."""
        import os
        import sys
        
        results = {}
        
        if not os.path.exists(directory):
            logger.warning(f"Plugin directory does not exist: {directory}")
            return results
        
        # Add directory to Python path temporarily
        if directory not in sys.path:
            sys.path.insert(0, directory)
        
        try:
            for filename in os.listdir(directory):
                if filename.endswith('.py') and not filename.startswith('__'):
                    module_name = filename[:-3]  # Remove .py extension
                    
                    try:
                        # Import the module
                        module = importlib.import_module(module_name)
                        
                        # Look for plugin classes in the module
                        plugin_classes = self._find_plugin_classes(module)
                        
                        # Instantiate and register plugins
                        for plugin_class in plugin_classes:
                            plugin_instance = plugin_class()
                            success = self.registry.register_plugin(plugin_instance)
                            results[plugin_instance.metadata.name] = success
                            
                    except Exception as e:
                        logger.error(f"Failed to load plugin from {filename}: {e}")
                        results[module_name] = False
        
        finally:
            # Remove directory from Python path
            if directory in sys.path:
                sys.path.remove(directory)
        
        return results
    
    def _find_plugin_classes(self, module) -> List[Type[IPlugin]]:
        """Find plugin classes in a module."""
        plugin_classes = []
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, IPlugin) and 
                obj is not IPlugin and
                not inspect.isabstract(obj)):
                
                plugin_classes.append(obj)
        
        return plugin_classes
    
    async def get_notification_channels(self) -> List[INotificationChannelPlugin]:
        """Get all enabled notification channel plugins."""
        plugins = self.registry.get_plugins_by_type(PluginType.NOTIFICATION_CHANNEL)
        return [plugin for plugin in plugins if isinstance(plugin, INotificationChannelPlugin)]
    
    async def get_message_processors(self) -> List[IMessageProcessorPlugin]:
        """Get all enabled message processor plugins, sorted by priority."""
        plugins = self.registry.get_plugins_by_type(PluginType.MESSAGE_PROCESSOR)
        processors = [plugin for plugin in plugins if isinstance(plugin, IMessageProcessorPlugin)]
        return sorted(processors, key=lambda p: p.get_processing_priority())
    
    async def get_importance_analyzers(self) -> List[IImportanceAnalyzerPlugin]:
        """Get all enabled importance analyzer plugins."""
        plugins = self.registry.get_plugins_by_type(PluginType.IMPORTANCE_ANALYZER)
        return [plugin for plugin in plugins if isinstance(plugin, IImportanceAnalyzerPlugin)]
    
    async def get_event_handlers(self, event_type: str) -> List[IEventHandlerPlugin]:
        """Get all event handlers that support a specific event type."""
        plugins = self.registry.get_plugins_by_type(PluginType.EVENT_HANDLER)
        handlers = [plugin for plugin in plugins if isinstance(plugin, IEventHandlerPlugin)]
        return [handler for handler in handlers if event_type in handler.get_supported_events()]
    
    async def health_check_plugins(self) -> Dict[str, Dict[str, Any]]:
        """Run health checks on all enabled plugins."""
        results = {}
        
        for plugin_name, plugin in self.registry._plugins.items():
            if not self.registry.is_plugin_enabled(plugin_name):
                continue
            
            try:
                health_info = await plugin.health_check()
                results[plugin_name] = {
                    "healthy": True,
                    "details": health_info
                }
            except Exception as e:
                logger.error(f"Health check failed for plugin {plugin_name}: {e}")
                results[plugin_name] = {
                    "healthy": False,
                    "error": str(e)
                }
        
        return results