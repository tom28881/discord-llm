"""
Configuration service with feature flags, environment-based config, and dynamic updates.
This service provides a unified interface for all configuration management needs.
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import yaml

from ..interfaces.services import IConfigService
from ..interfaces.repositories import IConfigRepository

logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    ENVIRONMENT = "environment"
    FILE = "file"  
    DATABASE = "database"
    REMOTE = "remote"
    DEFAULT = "default"


class ConfigType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    JSON = "json"


@dataclass
class ConfigDefinition:
    """Definition of a configuration parameter."""
    key: str
    config_type: ConfigType
    default_value: Any
    description: str = ""
    required: bool = False
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    sources: List[ConfigSource] = field(default_factory=lambda: [ConfigSource.ENVIRONMENT, ConfigSource.FILE, ConfigSource.DATABASE])
    sensitive: bool = False  # If true, value will be masked in logs
    category: str = "general"
    reload_on_change: bool = True


@dataclass 
class FeatureFlag:
    """Feature flag definition."""
    name: str
    enabled: bool
    description: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)
    rollout_percentage: float = 100.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigurationService(IConfigService):
    """
    Comprehensive configuration management service.
    
    Features:
    - Environment variable support
    - File-based configuration (JSON, YAML)
    - Database-backed configuration
    - Feature flags with rollout control
    - Configuration validation and type conversion
    - Hot reloading capabilities
    - User preference management
    - Configuration change notifications
    """
    
    def __init__(
        self, 
        config_repository: IConfigRepository,
        config_file_path: Optional[str] = None,
        enable_hot_reload: bool = True
    ):
        self.config_repository = config_repository
        self.config_file_path = config_file_path or "config/app_config.yaml"
        self.enable_hot_reload = enable_hot_reload
        
        # In-memory caches
        self._config_cache: Dict[str, Any] = {}
        self._feature_flags_cache: Dict[str, FeatureFlag] = {}
        self._user_preferences_cache: Dict[str, Dict[str, Any]] = {}
        self._config_definitions: Dict[str, ConfigDefinition] = {}
        
        # File watching for hot reload
        self._last_config_file_mtime: Optional[float] = None
        self._reload_task: Optional[asyncio.Task] = None
        
        self._initialize_default_config_definitions()
    
    def _initialize_default_config_definitions(self) -> None:
        """Initialize default configuration definitions."""
        default_configs = [
            # Discord Configuration
            ConfigDefinition(
                key="discord.token",
                config_type=ConfigType.STRING,
                default_value="",
                description="Discord bot token",
                required=True,
                sensitive=True,
                category="discord"
            ),
            ConfigDefinition(
                key="discord.rate_limit_delay",
                config_type=ConfigType.INTEGER,
                default_value=1,
                description="Delay between Discord API calls (seconds)",
                validation_rules={"min": 1, "max": 60},
                category="discord"
            ),
            ConfigDefinition(
                key="discord.max_messages_per_request",
                config_type=ConfigType.INTEGER,
                default_value=100,
                description="Maximum messages to fetch per API request",
                validation_rules={"min": 1, "max": 1000},
                category="discord"
            ),
            
            # Database Configuration
            ConfigDefinition(
                key="database.url",
                config_type=ConfigType.STRING,
                default_value="sqlite:///data/discord_monitor.db",
                description="Database connection URL",
                category="database"
            ),
            ConfigDefinition(
                key="database.pool_size",
                config_type=ConfigType.INTEGER,
                default_value=20,
                description="Database connection pool size",
                validation_rules={"min": 1, "max": 100},
                category="database"
            ),
            
            # LLM Configuration
            ConfigDefinition(
                key="llm.provider",
                config_type=ConfigType.STRING,
                default_value="openai",
                description="LLM provider (openai, anthropic, google)",
                validation_rules={"choices": ["openai", "anthropic", "google"]},
                category="llm"
            ),
            ConfigDefinition(
                key="llm.model",
                config_type=ConfigType.STRING,
                default_value="gpt-4",
                description="LLM model to use",
                category="llm"
            ),
            ConfigDefinition(
                key="llm.api_key",
                config_type=ConfigType.STRING,
                default_value="",
                description="LLM API key",
                required=True,
                sensitive=True,
                category="llm"
            ),
            ConfigDefinition(
                key="llm.max_tokens",
                config_type=ConfigType.INTEGER,
                default_value=4000,
                description="Maximum tokens per LLM request",
                validation_rules={"min": 100, "max": 32000},
                category="llm"
            ),
            
            # Monitoring Configuration  
            ConfigDefinition(
                key="monitoring.real_time_enabled",
                config_type=ConfigType.BOOLEAN,
                default_value=True,
                description="Enable real-time message monitoring",
                category="monitoring"
            ),
            ConfigDefinition(
                key="monitoring.batch_size",
                config_type=ConfigType.INTEGER,
                default_value=50,
                description="Batch size for message processing",
                validation_rules={"min": 1, "max": 1000},
                category="monitoring"
            ),
            ConfigDefinition(
                key="monitoring.importance_threshold",
                config_type=ConfigType.STRING,
                default_value="medium",
                description="Minimum importance level for notifications",
                validation_rules={"choices": ["critical", "high", "medium", "low", "noise"]},
                category="monitoring"
            ),
            
            # Notification Configuration
            ConfigDefinition(
                key="notifications.enabled",
                config_type=ConfigType.BOOLEAN,
                default_value=True,
                description="Enable notifications",
                category="notifications"
            ),
            ConfigDefinition(
                key="notifications.rate_limit_per_hour",
                config_type=ConfigType.INTEGER,
                default_value=50,
                description="Maximum notifications per hour",
                validation_rules={"min": 1, "max": 1000},
                category="notifications"
            ),
            
            # Importance Analysis Configuration
            ConfigDefinition(
                key="importance.keywords",
                config_type=ConfigType.DICT,
                default_value={
                    "urgent": 0.8,
                    "important": 0.6,
                    "critical": 0.9,
                    "help": 0.4,
                    "error": 0.7,
                    "bug": 0.6,
                    "issue": 0.5,
                    "problem": 0.5,
                    "alert": 0.8,
                    "warning": 0.6
                },
                description="Keywords and their importance weights",
                category="importance"
            ),
            ConfigDefinition(
                key="importance.channel_weights",
                config_type=ConfigType.DICT,
                default_value={},
                description="Channel-specific importance weights",
                category="importance"
            ),
            
            # System Configuration
            ConfigDefinition(
                key="system.log_level",
                config_type=ConfigType.STRING,
                default_value="INFO",
                description="Logging level",
                validation_rules={"choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                category="system"
            ),
            ConfigDefinition(
                key="system.max_concurrent_processing",
                config_type=ConfigType.INTEGER,
                default_value=10,
                description="Maximum concurrent message processing",
                validation_rules={"min": 1, "max": 100},
                category="system"
            ),
            ConfigDefinition(
                key="system.health_check_interval",
                config_type=ConfigType.INTEGER,
                default_value=300,
                description="Health check interval in seconds",
                validation_rules={"min": 60, "max": 3600},
                category="system"
            )
        ]
        
        for config_def in default_configs:
            self._config_definitions[config_def.key] = config_def
    
    async def initialize(self) -> bool:
        """Initialize the configuration service."""
        try:
            # Load configuration from all sources
            await self._load_configuration()
            
            # Start hot reload task if enabled
            if self.enable_hot_reload:
                self._reload_task = asyncio.create_task(self._hot_reload_worker())
            
            logger.info("Configuration service initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize configuration service: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the configuration service."""
        if self._reload_task:
            self._reload_task.cancel()
            try:
                await self._reload_task
            except asyncio.CancelledError:
                pass
    
    async def _load_configuration(self) -> None:
        """Load configuration from all sources."""
        # 1. Load defaults
        for key, config_def in self._config_definitions.items():
            self._config_cache[key] = config_def.default_value
        
        # 2. Load from file
        await self._load_from_file()
        
        # 3. Load from environment variables
        self._load_from_environment()
        
        # 4. Load from database
        await self._load_from_database()
        
        # 5. Load feature flags
        await self._load_feature_flags()
    
    async def _load_from_file(self) -> None:
        """Load configuration from file."""
        try:
            config_path = Path(self.config_file_path)
            if not config_path.exists():
                logger.info(f"Configuration file not found: {self.config_file_path}")
                return
            
            # Track file modification time for hot reload
            stat = config_path.stat()
            self._last_config_file_mtime = stat.st_mtime
            
            # Load based on file extension
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)
            
            # Flatten nested configuration
            flattened = self._flatten_dict(file_config)
            
            # Update cache with file values
            for key, value in flattened.items():
                if key in self._config_definitions:
                    try:
                        validated_value = self._validate_and_convert(key, value)
                        self._config_cache[key] = validated_value
                    except Exception as e:
                        logger.warning(f"Invalid config value for {key}: {e}")
            
            logger.info(f"Loaded configuration from {self.config_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from file: {e}")
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        try:
            for key, config_def in self._config_definitions.items():
                if ConfigSource.ENVIRONMENT not in config_def.sources:
                    continue
                
                # Convert key to environment variable format
                env_key = key.replace('.', '_').upper()
                env_value = os.getenv(env_key)
                
                if env_value is not None:
                    try:
                        validated_value = self._validate_and_convert(key, env_value)
                        self._config_cache[key] = validated_value
                        
                        if not config_def.sensitive:
                            logger.debug(f"Loaded config from env {env_key}: {validated_value}")
                    except Exception as e:
                        logger.warning(f"Invalid env config value for {env_key}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from environment: {e}")
    
    async def _load_from_database(self) -> None:
        """Load configuration from database."""
        try:
            for key in self._config_definitions.keys():
                try:
                    db_value = await self.config_repository.get_config(key)
                    if db_value is not None:
                        validated_value = self._validate_and_convert(key, db_value)
                        self._config_cache[key] = validated_value
                except Exception as e:
                    logger.debug(f"No database config for {key}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from database: {e}")
    
    async def _load_feature_flags(self) -> None:
        """Load feature flags from database."""
        try:
            # This would load feature flags from database
            # For now, using some default flags
            default_flags = [
                FeatureFlag(
                    name="real_time_monitoring",
                    enabled=True,
                    description="Enable real-time message monitoring"
                ),
                FeatureFlag(
                    name="advanced_importance_analysis", 
                    enabled=True,
                    description="Enable advanced importance analysis with ML"
                ),
                FeatureFlag(
                    name="notification_rate_limiting",
                    enabled=True,
                    description="Enable notification rate limiting"
                ),
                FeatureFlag(
                    name="plugin_system",
                    enabled=True,
                    description="Enable plugin system"
                ),
                FeatureFlag(
                    name="web_dashboard",
                    enabled=False,
                    description="Enable web dashboard interface"
                ),
                FeatureFlag(
                    name="llm_powered_summaries",
                    enabled=True,
                    description="Enable LLM-powered message summaries"
                )
            ]
            
            for flag in default_flags:
                self._feature_flags_cache[flag.name] = flag
            
        except Exception as e:
            logger.error(f"Failed to load feature flags: {e}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for key, value in d.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, sep=sep).items())
            else:
                items.append((new_key, value))
        return dict(items)
    
    def _validate_and_convert(self, key: str, value: Any) -> Any:
        """Validate and convert configuration value."""
        config_def = self._config_definitions.get(key)
        if not config_def:
            return value
        
        # Type conversion
        try:
            if config_def.config_type == ConfigType.STRING:
                converted_value = str(value)
            elif config_def.config_type == ConfigType.INTEGER:
                converted_value = int(value)
            elif config_def.config_type == ConfigType.FLOAT:
                converted_value = float(value)
            elif config_def.config_type == ConfigType.BOOLEAN:
                if isinstance(value, str):
                    converted_value = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    converted_value = bool(value)
            elif config_def.config_type == ConfigType.LIST:
                if isinstance(value, str):
                    # Try to parse JSON array
                    converted_value = json.loads(value) if value.startswith('[') else [value]
                else:
                    converted_value = list(value) if not isinstance(value, list) else value
            elif config_def.config_type == ConfigType.DICT:
                if isinstance(value, str):
                    converted_value = json.loads(value)
                else:
                    converted_value = dict(value) if not isinstance(value, dict) else value
            elif config_def.config_type == ConfigType.JSON:
                if isinstance(value, str):
                    converted_value = json.loads(value)
                else:
                    converted_value = value
            else:
                converted_value = value
        except Exception as e:
            raise ValueError(f"Cannot convert {value} to {config_def.config_type.value}: {e}")
        
        # Validation rules
        if config_def.validation_rules:
            self._validate_value(converted_value, config_def.validation_rules)
        
        return converted_value
    
    def _validate_value(self, value: Any, rules: Dict[str, Any]) -> None:
        """Validate value against rules."""
        if "min" in rules and isinstance(value, (int, float)):
            if value < rules["min"]:
                raise ValueError(f"Value {value} is below minimum {rules['min']}")
        
        if "max" in rules and isinstance(value, (int, float)):
            if value > rules["max"]:
                raise ValueError(f"Value {value} is above maximum {rules['max']}")
        
        if "choices" in rules and value not in rules["choices"]:
            raise ValueError(f"Value {value} is not in allowed choices {rules['choices']}")
        
        if "regex" in rules and isinstance(value, str):
            import re
            if not re.match(rules["regex"], value):
                raise ValueError(f"Value {value} does not match pattern {rules['regex']}")
    
    async def _hot_reload_worker(self) -> None:
        """Worker task for hot reloading configuration."""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                if not self.config_file_path:
                    continue
                
                config_path = Path(self.config_file_path)
                if not config_path.exists():
                    continue
                
                stat = config_path.stat()
                if (self._last_config_file_mtime is None or 
                    stat.st_mtime > self._last_config_file_mtime):
                    
                    logger.info("Configuration file changed, reloading...")
                    await self._load_from_file()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Hot reload error: {e}")
    
    # IConfigService implementation
    
    async def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config_cache.get(key, default)
    
    async def set_config(self, key: str, value: Any) -> bool:
        """Set configuration value."""
        try:
            # Validate if we have a definition for this key
            if key in self._config_definitions:
                validated_value = self._validate_and_convert(key, value)
                self._config_cache[key] = validated_value
            else:
                self._config_cache[key] = value
            
            # Persist to database
            await self.config_repository.set_config(key, value)
            
            # Log change (mask sensitive values)
            config_def = self._config_definitions.get(key)
            if config_def and config_def.sensitive:
                logger.info(f"Configuration updated: {key} = [MASKED]")
            else:
                logger.info(f"Configuration updated: {key} = {value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set configuration {key}: {e}")
            return False
    
    async def get_feature_flag(self, flag_name: str) -> bool:
        """Get feature flag status."""
        flag = self._feature_flags_cache.get(flag_name)
        if not flag:
            return False
        
        if not flag.enabled:
            return False
        
        # Check rollout percentage
        if flag.rollout_percentage < 100.0:
            import hashlib
            # Use consistent hash to determine if this instance should have the flag
            hash_input = f"{flag_name}_{os.getenv('INSTANCE_ID', 'default')}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            rollout_bucket = (hash_value % 100) + 1
            
            if rollout_bucket > flag.rollout_percentage:
                return False
        
        # Check conditions if any
        if flag.conditions:
            # This could be extended to check various conditions
            # For now, just return the enabled status
            pass
        
        return True
    
    async def set_feature_flag(self, flag_name: str, enabled: bool) -> bool:
        """Set feature flag status."""
        try:
            if flag_name in self._feature_flags_cache:
                flag = self._feature_flags_cache[flag_name]
                updated_flag = FeatureFlag(
                    name=flag.name,
                    enabled=enabled,
                    description=flag.description,
                    conditions=flag.conditions,
                    rollout_percentage=flag.rollout_percentage,
                    created_at=flag.created_at,
                    updated_at=datetime.utcnow(),
                    metadata=flag.metadata
                )
            else:
                updated_flag = FeatureFlag(
                    name=flag_name,
                    enabled=enabled,
                    description=f"Feature flag: {flag_name}"
                )
            
            self._feature_flags_cache[flag_name] = updated_flag
            
            # Persist to database
            await self.config_repository.set_config(f"feature_flags.{flag_name}", asdict(updated_flag))
            
            logger.info(f"Feature flag updated: {flag_name} = {enabled}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set feature flag {flag_name}: {e}")
            return False
    
    async def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        monitoring_config = {}
        
        for key, config_def in self._config_definitions.items():
            if config_def.category == "monitoring":
                monitoring_config[key.replace("monitoring.", "")] = self._config_cache.get(key)
        
        return monitoring_config
    
    async def update_monitoring_config(self, config: Dict[str, Any]) -> bool:
        """Update monitoring configuration."""
        try:
            success = True
            
            for key, value in config.items():
                full_key = f"monitoring.{key}"
                result = await self.set_config(full_key, value)
                if not result:
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update monitoring config: {e}")
            return False
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences."""
        if user_id not in self._user_preferences_cache:
            try:
                prefs = await self.config_repository.get_user_preferences(user_id)
                self._user_preferences_cache[user_id] = prefs
            except Exception as e:
                logger.error(f"Failed to load user preferences for {user_id}: {e}")
                self._user_preferences_cache[user_id] = {}
        
        return self._user_preferences_cache[user_id]
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        try:
            # Update cache
            if user_id not in self._user_preferences_cache:
                self._user_preferences_cache[user_id] = {}
            
            self._user_preferences_cache[user_id].update(preferences)
            
            # Persist to database
            for key, value in preferences.items():
                await self.config_repository.set_user_preference(user_id, key, value)
            
            logger.info(f"Updated preferences for user {user_id}: {list(preferences.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user preferences for {user_id}: {e}")
            return False
    
    # Additional utility methods
    
    async def get_config_by_category(self, category: str) -> Dict[str, Any]:
        """Get all configuration values for a specific category."""
        category_config = {}
        
        for key, config_def in self._config_definitions.items():
            if config_def.category == category:
                category_config[key] = self._config_cache.get(key)
        
        return category_config
    
    async def get_all_config(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Get all configuration values."""
        all_config = {}
        
        for key, value in self._config_cache.items():
            config_def = self._config_definitions.get(key)
            
            if config_def and config_def.sensitive and not include_sensitive:
                all_config[key] = "[MASKED]"
            else:
                all_config[key] = value
        
        return all_config
    
    async def get_all_feature_flags(self) -> Dict[str, Dict[str, Any]]:
        """Get all feature flags."""
        return {
            name: {
                "enabled": flag.enabled,
                "description": flag.description,
                "rollout_percentage": flag.rollout_percentage,
                "updated_at": flag.updated_at.isoformat()
            }
            for name, flag in self._feature_flags_cache.items()
        }
    
    async def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate all configuration values."""
        validation_errors = {}
        
        for key, config_def in self._config_definitions.items():
            errors = []
            
            # Check if required config is present
            if config_def.required and key not in self._config_cache:
                errors.append(f"Required configuration {key} is missing")
            
            # Validate current value if present
            if key in self._config_cache:
                try:
                    self._validate_and_convert(key, self._config_cache[key])
                except Exception as e:
                    errors.append(str(e))
            
            if errors:
                validation_errors[key] = errors
        
        return validation_errors