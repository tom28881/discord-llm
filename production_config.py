"""
Production configuration and deployment settings for Discord monitoring assistant.

Provides comprehensive configuration management, environment-specific settings,
and production deployment utilities for reliable 24/7 operation.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from pathlib import Path
from dotenv import load_dotenv

from lib.exceptions import ConfigurationError


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    path: str = "data/db.sqlite"
    connection_timeout: float = 30.0
    max_retries: int = 3
    backup_interval_hours: int = 6
    max_backups: int = 48
    wal_mode: bool = True
    synchronous: str = "NORMAL"
    journal_size_limit: int = 100 * 1024 * 1024  # 100MB
    auto_vacuum: str = "INCREMENTAL"
    page_size: int = 4096
    connection_pool_size: int = 10


@dataclass
class DiscordConfig:
    """Discord API configuration settings."""
    token: str = ""
    base_url: str = "https://discord.com/api/v9"
    rate_limit_requests_per_minute: int = 50
    max_retries: int = 3
    connection_timeout: float = 30.0
    read_timeout: float = 60.0
    max_message_batch_size: int = 100
    channel_delay_seconds: float = 1.0
    server_delay_seconds: float = 10.0
    error_retry_delay_seconds: float = 30.0


@dataclass
class LLMConfig:
    """LLM integration configuration settings."""
    google_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    daily_cost_limit: float = 10.0
    default_provider: str = "gemini"
    fallback_providers: List[str] = field(default_factory=lambda: ["gemini-1.5-flash", "gemini-2.5-pro"])
    max_retries: int = 2
    timeout_seconds: float = 60.0
    max_tokens: int = 8192


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_max_size_mb: int = 100
    log_backup_count: int = 5
    metrics_retention_hours: int = 24
    health_check_interval_seconds: int = 30
    alert_cooldown_minutes: int = 5
    enable_json_logging: bool = True
    enable_console_logging: bool = True


@dataclass
class ProcessingConfig:
    """Processing pipeline configuration."""
    max_queue_size: int = 10000
    worker_count: int = 4
    task_timeout_seconds: float = 300.0
    max_task_retries: int = 3
    memory_limit_mb: float = 1024.0
    deadlock_check_interval_seconds: int = 30
    memory_check_interval_seconds: int = 60


@dataclass
class AlertConfig:
    """Alert configuration settings."""
    email_enabled: bool = False
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)
    
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProductionConfig:
    """Complete production configuration."""
    environment: str = "production"
    debug_mode: bool = False
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    
    # Health check settings
    health_check_port: int = 8080
    health_check_path: str = "/health"
    
    # Graceful shutdown settings
    shutdown_timeout_seconds: float = 30.0
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Required Discord token
        if not self.discord.token:
            issues.append("Discord token is required")
        
        # Required LLM API key
        if not any([self.llm.google_api_key, self.llm.openai_api_key, self.llm.openrouter_api_key]):
            issues.append("At least one LLM API key is required")
        
        # Database path
        db_dir = Path(self.database.path).parent
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create database directory {db_dir}: {e}")
        
        # Log directory
        log_dir = Path(self.monitoring.log_dir)
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create log directory {log_dir}: {e}")
        
        # Email configuration
        if self.alerts.email_enabled:
            required_email_fields = [
                'email_smtp_host', 'email_username', 'email_password', 
                'email_from', 'email_to'
            ]
            for field in required_email_fields:
                if not getattr(self.alerts, field):
                    issues.append(f"Email alert enabled but {field} not configured")
        
        # Slack configuration
        if self.alerts.slack_enabled:
            if not self.alerts.slack_webhook_url:
                issues.append("Slack alerts enabled but webhook URL not configured")
        
        # Webhook configuration
        if self.alerts.webhook_enabled:
            if not self.alerts.webhook_url:
                issues.append("Webhook alerts enabled but URL not configured")
        
        # Resource limits
        if self.processing.memory_limit_mb < 256:
            issues.append("Memory limit too low, should be at least 256MB")
        
        if self.processing.worker_count < 1:
            issues.append("Worker count must be at least 1")
        
        if self.llm.daily_cost_limit <= 0:
            issues.append("LLM daily cost limit must be positive")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductionConfig':
        """Create configuration from dictionary."""
        # Handle nested dataclasses
        if 'database' in data and isinstance(data['database'], dict):
            data['database'] = DatabaseConfig(**data['database'])
        
        if 'discord' in data and isinstance(data['discord'], dict):
            data['discord'] = DiscordConfig(**data['discord'])
        
        if 'llm' in data and isinstance(data['llm'], dict):
            data['llm'] = LLMConfig(**data['llm'])
        
        if 'monitoring' in data and isinstance(data['monitoring'], dict):
            data['monitoring'] = MonitoringConfig(**data['monitoring'])
        
        if 'processing' in data and isinstance(data['processing'], dict):
            data['processing'] = ProcessingConfig(**data['processing'])
        
        if 'alerts' in data and isinstance(data['alerts'], dict):
            data['alerts'] = AlertConfig(**data['alerts'])
        
        return cls(**data)


class ConfigurationManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None, env_file: Optional[str] = None):
        self.config_path = config_path or "production_config.json"
        self.env_file = env_file or ".env"
        self.logger = logging.getLogger(__name__)
        
    def load_config(self) -> ProductionConfig:
        """Load configuration from files and environment."""
        # Load environment variables
        if os.path.exists(self.env_file):
            load_dotenv(self.env_file)
        
        # Start with default configuration
        config = ProductionConfig()
        
        # Load from JSON file if exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                config = ProductionConfig.from_dict(config_data)
            except Exception as e:
                raise ConfigurationError(f"Failed to load config file {self.config_path}: {e}")
        
        # Override with environment variables
        config = self._apply_environment_overrides(config)
        
        # Validate configuration
        issues = config.validate()
        if issues:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(issues)}")
        
        return config
    
    def _apply_environment_overrides(self, config: ProductionConfig) -> ProductionConfig:
        """Apply environment variable overrides to configuration."""
        
        # Environment mapping
        env_mappings = {
            # Discord
            'DISCORD_TOKEN': ('discord', 'token'),
            'DISCORD_RATE_LIMIT_RPM': ('discord', 'rate_limit_requests_per_minute', int),
            
            # LLM
            'GOOGLE_API_KEY': ('llm', 'google_api_key'),
            'OPENAI_API_KEY': ('llm', 'openai_api_key'),
            'OPENROUTER_API_KEY': ('llm', 'openrouter_api_key'),
            'LLM_DAILY_COST_LIMIT': ('llm', 'daily_cost_limit', float),
            
            # Database
            'DATABASE_PATH': ('database', 'path'),
            'DATABASE_BACKUP_INTERVAL_HOURS': ('database', 'backup_interval_hours', int),
            
            # Monitoring
            'LOG_LEVEL': ('monitoring', 'log_level'),
            'LOG_DIR': ('monitoring', 'log_dir'),
            
            # Processing
            'WORKER_COUNT': ('processing', 'worker_count', int),
            'MAX_QUEUE_SIZE': ('processing', 'max_queue_size', int),
            'MEMORY_LIMIT_MB': ('processing', 'memory_limit_mb', float),
            
            # Alerts
            'EMAIL_ENABLED': ('alerts', 'email_enabled', bool),
            'EMAIL_SMTP_HOST': ('alerts', 'email_smtp_host'),
            'EMAIL_SMTP_PORT': ('alerts', 'email_smtp_port', int),
            'EMAIL_USERNAME': ('alerts', 'email_username'),
            'EMAIL_PASSWORD': ('alerts', 'email_password'),
            'EMAIL_FROM': ('alerts', 'email_from'),
            'EMAIL_TO': ('alerts', 'email_to', lambda x: x.split(',')),
            
            'SLACK_ENABLED': ('alerts', 'slack_enabled', bool),
            'SLACK_WEBHOOK_URL': ('alerts', 'slack_webhook_url'),
            
            'WEBHOOK_ENABLED': ('alerts', 'webhook_enabled', bool),
            'WEBHOOK_URL': ('alerts', 'webhook_url'),
            
            # General
            'ENVIRONMENT': ('environment',),
            'DEBUG_MODE': ('debug_mode', bool),
            'HEALTH_CHECK_PORT': ('health_check_port', int),
        }
        
        for env_var, mapping in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Parse the mapping
                    if len(mapping) == 1:
                        # Direct attribute
                        attr_name = mapping[0]
                        converter = str
                        target = config
                    elif len(mapping) == 2:
                        # Nested attribute
                        section_name, attr_name = mapping
                        converter = str
                        target = getattr(config, section_name)
                    else:
                        # Nested attribute with converter
                        section_name, attr_name, converter = mapping
                        target = getattr(config, section_name)
                    
                    # Convert value
                    if converter == bool:
                        converted_value = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        converted_value = converter(value)
                    
                    # Set the value
                    setattr(target, attr_name, converted_value)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to apply environment override {env_var}: {e}")
        
        return config
    
    def save_config(self, config: ProductionConfig, path: Optional[str] = None):
        """Save configuration to file."""
        save_path = path or self.config_path
        
        try:
            with open(save_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save config to {save_path}: {e}")


def create_sample_config(path: str = "production_config.json.example"):
    """Create a sample configuration file."""
    config = ProductionConfig()
    
    # Set example values
    config.discord.token = "YOUR_DISCORD_TOKEN_HERE"
    config.llm.google_api_key = "YOUR_GOOGLE_API_KEY_HERE"
    config.alerts.email_enabled = False
    config.alerts.email_smtp_host = "smtp.gmail.com"
    config.alerts.email_username = "your-email@gmail.com"
    config.alerts.email_password = "your-app-password"
    config.alerts.email_from = "discord-monitor@yourcompany.com"
    config.alerts.email_to = ["admin@yourcompany.com"]
    
    manager = ConfigurationManager()
    manager.save_config(config, path)
    print(f"Sample configuration saved to {path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create-sample":
        create_sample_config()
    else:
        # Test configuration loading
        try:
            manager = ConfigurationManager()
            config = manager.load_config()
            print("Configuration loaded successfully!")
            print(f"Environment: {config.environment}")
            print(f"Discord API configured: {'Yes' if config.discord.token else 'No'}")
            print(f"LLM API configured: {'Yes' if any([config.llm.google_api_key, config.llm.openai_api_key]) else 'No'}")
            
        except ConfigurationError as e:
            print(f"Configuration error: {e}")
            sys.exit(1)