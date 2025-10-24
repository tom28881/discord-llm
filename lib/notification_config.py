"""
Configuration management and main integration bridge for the notification system.
Provides easy setup, configuration, and management of all notification components.
"""

import logging
import sqlite3
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import threading
import time

from .notification_database import init_notification_db, NotificationRuleManager, NotificationQueue
from .notification_engine import get_notification_engine, SmartNotificationEngine
from .notification_channels import NotificationChannelFactory
from .external_integrations import get_integration_manager
from .api_server import APIServer, create_api_token
from .database import init_db, DB_NAME

logger = logging.getLogger('notification_config')

class NotificationSystemConfig:
    """Main configuration class for the notification system."""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or str(Path(__file__).parent.parent / 'notification_config.json')
        self.config: Dict[str, Any] = {}
        self.engine: Optional[SmartNotificationEngine] = None
        self.api_server: Optional[APIServer] = None
        self._load_config()
        
        # Initialize databases
        init_db()
        init_notification_db()
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
            else:
                # Create default configuration
                self.config = self._get_default_config()
                self.save_config()
                logger.info("Created default configuration")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "notification_engine": {
                "processing_interval_seconds": 10,
                "dedup_window_minutes": 30,
                "batching": {
                    "enabled": True,
                    "batch_size": 5,
                    "timeout_minutes": 15,
                    "batch_channels": ["email"]
                }
            },
            "api_server": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False
            },
            "notification_channels": {
                "email": {
                    "enabled": False,
                    "config": {
                        "smtp_server": "smtp.gmail.com",
                        "smtp_port": 587,
                        "username": "",
                        "password": "",
                        "to_emails": []
                    }
                },
                "telegram": {
                    "enabled": False,
                    "config": {
                        "bot_token": "",
                        "chat_ids": []
                    }
                },
                "desktop": {
                    "enabled": True,
                    "config": {
                        "app_name": "Discord Monitor"
                    }
                },
                "slack": {
                    "enabled": False,
                    "config": {
                        "webhook_url": "",
                        "channel": "#general",
                        "username": "Discord Monitor"
                    }
                }
            },
            "external_integrations": {
                "google_calendar": {
                    "enabled": False,
                    "config": {
                        "client_id": "",
                        "client_secret": "",
                        "access_token": "",
                        "refresh_token": "",
                        "calendar_id": "primary"
                    }
                },
                "todoist": {
                    "enabled": False,
                    "config": {
                        "api_token": "",
                        "project_id": None
                    }
                },
                "notion": {
                    "enabled": False,
                    "config": {
                        "auth_token": "",
                        "database_id": ""
                    }
                },
                "rss": {
                    "enabled": True,
                    "config": {
                        "feed_path": "data/discord_feed.xml",
                        "feed_title": "Discord Notifications",
                        "feed_description": "Latest Discord notifications",
                        "feed_link": "http://localhost:8501",
                        "max_items": 50
                    }
                }
            },
            "default_rules": [
                {
                    "name": "High Priority Keywords",
                    "keywords": ["urgent", "important", "asap", "emergency", "critical"],
                    "priority": 4,
                    "channels": ["email", "telegram", "desktop"],
                    "conditions": {
                        "require_mentions": False,
                        "min_length": 10
                    }
                },
                {
                    "name": "Group Purchases",
                    "keywords": ["group buy", "bulk order", "split cost", "group purchase"],
                    "priority": 3,
                    "channels": ["email", "desktop"],
                    "conditions": {
                        "time_range": {"start": 9, "end": 21}
                    }
                },
                {
                    "name": "Decision Making",
                    "keywords": ["vote", "decision", "poll", "choose", "decide"],
                    "priority": 2,
                    "channels": ["desktop"],
                    "conditions": {
                        "min_length": 20
                    }
                }
            ]
        }
    
    def save_config(self):
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def setup_notification_channels(self):
        """Set up notification channels from configuration."""
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            for channel_type, channel_config in self.config.get('notification_channels', {}).items():
                if not channel_config.get('enabled', False):
                    continue
                
                # Check if channel already exists
                cursor.execute('''
                SELECT id FROM notification_channels 
                WHERE channel_type = ? AND name = ?
                ''', (channel_type, 'default'))
                
                if cursor.fetchone():
                    continue  # Channel already exists
                
                # Create channel
                now = int(datetime.now().timestamp())
                cursor.execute('''
                INSERT INTO notification_channels 
                (channel_type, name, config, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ''', (channel_type, 'default', json.dumps(channel_config['config']), now, now))
                
                logger.info(f"Created notification channel: {channel_type}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to setup notification channels: {e}")
    
    def setup_external_integrations(self):
        """Set up external integrations from configuration."""
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            for integration_type, integration_config in self.config.get('external_integrations', {}).items():
                if not integration_config.get('enabled', False):
                    continue
                
                # Check if integration already exists
                cursor.execute('''
                SELECT id FROM external_integrations 
                WHERE integration_type = ? AND name = ?
                ''', (integration_type, 'default'))
                
                if cursor.fetchone():
                    continue  # Integration already exists
                
                # Create integration
                now = int(datetime.now().timestamp())
                cursor.execute('''
                INSERT INTO external_integrations 
                (integration_type, name, config, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ''', (integration_type, 'default', json.dumps(integration_config['config']), now, now))
                
                logger.info(f"Created external integration: {integration_type}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to setup external integrations: {e}")
    
    def setup_default_rules(self):
        """Set up default notification rules."""
        try:
            rule_manager = NotificationRuleManager()
            
            for rule_config in self.config.get('default_rules', []):
                # Check if rule already exists
                existing_rules = rule_manager.get_active_rules()
                if any(rule['name'] == rule_config['name'] for rule in existing_rules):
                    continue
                
                from .notification_database import NotificationPriority, NotificationChannel
                
                # Create rule
                rule_manager.create_rule(
                    name=rule_config['name'],
                    keywords=rule_config.get('keywords'),
                    priority=NotificationPriority(rule_config['priority']),
                    channels=[NotificationChannel(ch) for ch in rule_config['channels']],
                    conditions=rule_config.get('conditions', {})
                )
                
                logger.info(f"Created default rule: {rule_config['name']}")
        
        except Exception as e:
            logger.error(f"Failed to setup default rules: {e}")
    
    def start_notification_engine(self):
        """Start the notification engine."""
        try:
            engine_config = self.config.get('notification_engine', {})
            self.engine = get_notification_engine(engine_config)
            self.engine.start()
            logger.info("Notification engine started")
        except Exception as e:
            logger.error(f"Failed to start notification engine: {e}")
    
    def start_api_server(self):
        """Start the API server."""
        try:
            api_config = self.config.get('api_server', {})
            if not api_config.get('enabled', True):
                return
            
            self.api_server = APIServer(
                host=api_config.get('host', '0.0.0.0'),
                port=api_config.get('port', 8000)
            )
            self.api_server.start(debug=api_config.get('debug', False))
            logger.info(f"API server started on {api_config.get('host')}:{api_config.get('port')}")
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
    
    def initialize_system(self):
        """Initialize the complete notification system."""
        logger.info("Initializing notification system...")
        
        # Setup components
        self.setup_notification_channels()
        self.setup_external_integrations()
        self.setup_default_rules()
        
        # Start services
        self.start_notification_engine()
        self.start_api_server()
        
        logger.info("Notification system initialized successfully")
    
    def stop_system(self):
        """Stop all notification system components."""
        logger.info("Stopping notification system...")
        
        if self.engine:
            self.engine.stop()
        
        if self.api_server:
            self.api_server.stop()
        
        logger.info("Notification system stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information."""
        status = {
            "notification_engine": {
                "running": self.engine is not None and self.engine.running,
                "statistics": self.engine.get_statistics() if self.engine else {}
            },
            "api_server": {
                "running": self.api_server is not None and self.api_server.running,
                "host": self.config.get('api_server', {}).get('host', 'N/A'),
                "port": self.config.get('api_server', {}).get('port', 'N/A')
            },
            "channels": {},
            "integrations": {}
        }
        
        # Get channel status
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            cursor.execute('SELECT channel_type, COUNT(*) FROM notification_channels WHERE is_active = 1 GROUP BY channel_type')
            for row in cursor.fetchall():
                status["channels"][row[0]] = {"count": row[1], "active": True}
            
            cursor.execute('SELECT integration_type, COUNT(*) FROM external_integrations WHERE is_active = 1 GROUP BY integration_type')
            for row in cursor.fetchall():
                status["integrations"][row[0]] = {"count": row[1], "active": True}
            
            conn.close()
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
        
        return status

class NotificationSystemIntegrator:
    """Integration bridge for connecting the notification system with existing Discord client."""
    
    def __init__(self, config: NotificationSystemConfig):
        self.config = config
        self.engine = config.engine
        self._message_buffer = []
        self._buffer_lock = threading.Lock()
        self._processing_thread: Optional[threading.Thread] = None
        self._running = False
    
    def process_discord_message(self, server_id: int, channel_id: int, message_id: int,
                               content: str, author: str = None, timestamp: datetime = None,
                               server_name: str = None, channel_name: str = None):
        """Process a Discord message for notifications."""
        if not self.engine:
            return
        
        metadata = {
            'server_id': server_id,
            'channel_id': channel_id,
            'message_id': message_id,
            'author': author,
            'timestamp': (timestamp or datetime.now()).isoformat(),
            'server_name': server_name,
            'channel_name': channel_name
        }
        
        # Add to buffer for batch processing
        with self._buffer_lock:
            self._message_buffer.append({
                'server_id': server_id,
                'channel_id': channel_id,
                'message_id': message_id,
                'content': content,
                'author': author,
                'metadata': metadata
            })
    
    def start_processing(self):
        """Start background processing of Discord messages."""
        if self._running:
            return
        
        self._running = True
        self._processing_thread = threading.Thread(target=self._process_message_buffer, daemon=True)
        self._processing_thread.start()
        logger.info("Started Discord message processing")
    
    def stop_processing(self):
        """Stop background processing."""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5)
        logger.info("Stopped Discord message processing")
    
    def _process_message_buffer(self):
        """Process buffered messages."""
        while self._running:
            try:
                messages_to_process = []
                
                # Get messages from buffer
                with self._buffer_lock:
                    if self._message_buffer:
                        messages_to_process = self._message_buffer.copy()
                        self._message_buffer.clear()
                
                # Process messages
                for message in messages_to_process:
                    try:
                        self.engine.process_new_message(
                            server_id=message['server_id'],
                            channel_id=message['channel_id'],
                            message_id=message['message_id'],
                            content=message['content'],
                            author=message['author'],
                            metadata=message['metadata']
                        )
                    except Exception as e:
                        logger.error(f"Error processing message {message['message_id']}: {e}")
                
                # Sleep before next iteration
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
                time.sleep(5)
    
    def create_manual_notification(self, title: str, content: str, priority: int = 2,
                                 channels: List[str] = None, metadata: Dict[str, Any] = None):
        """Create a manual notification (for testing or external triggers)."""
        if not self.engine:
            return
        
        from .notification_database import NotificationPriority, NotificationChannel
        
        # Create a dummy rule for manual notifications
        rule_manager = NotificationRuleManager()
        
        # Check if manual rule exists
        existing_rules = rule_manager.get_active_rules()
        manual_rule = next((rule for rule in existing_rules if rule['name'] == 'Manual Notifications'), None)
        
        if not manual_rule:
            # Create manual rule
            rule_id = rule_manager.create_rule(
                name='Manual Notifications',
                priority=NotificationPriority.MEDIUM,
                channels=[NotificationChannel.EMAIL, NotificationChannel.DESKTOP],
                conditions={'manual': True}
            )
        else:
            rule_id = manual_rule['id']
        
        # Queue notification
        queue = NotificationQueue()
        queue.enqueue_notification(
            rule_id=rule_id,
            message_id=0,  # No message ID for manual notifications
            title=title,
            content=content,
            priority=NotificationPriority(priority),
            channels=[NotificationChannel(ch) for ch in (channels or ['desktop'])],
            metadata=metadata or {}
        )
        
        logger.info(f"Created manual notification: {title}")

# Global system instance
_notification_system: Optional[NotificationSystemConfig] = None
_system_integrator: Optional[NotificationSystemIntegrator] = None

def get_notification_system(config_file: str = None) -> NotificationSystemConfig:
    """Get or create the global notification system instance."""
    global _notification_system
    
    if _notification_system is None:
        _notification_system = NotificationSystemConfig(config_file)
    
    return _notification_system

def get_system_integrator() -> NotificationSystemIntegrator:
    """Get or create the system integrator."""
    global _system_integrator, _notification_system
    
    if _system_integrator is None:
        if _notification_system is None:
            _notification_system = get_notification_system()
        _system_integrator = NotificationSystemIntegrator(_notification_system)
    
    return _system_integrator

def initialize_notification_system(config_file: str = None, auto_start: bool = True):
    """Initialize and optionally start the notification system."""
    system = get_notification_system(config_file)
    
    if auto_start:
        system.initialize_system()
        
        # Start message processing
        integrator = get_system_integrator()
        integrator.start_processing()
    
    return system

def shutdown_notification_system():
    """Shutdown the notification system."""
    global _notification_system, _system_integrator
    
    if _system_integrator:
        _system_integrator.stop_processing()
    
    if _notification_system:
        _notification_system.stop_system()
    
    _notification_system = None
    _system_integrator = None