"""
Database schema extensions for the notification system.
Extends the existing SQLite database with notification-specific tables.
"""

import sqlite3
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

# Import existing database configuration
from .database import DB_NAME, DATA_DIR

logger = logging.getLogger('notification_system')

class NotificationPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class NotificationStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    SCHEDULED = "scheduled"

class NotificationChannel(Enum):
    EMAIL = "email"
    TELEGRAM = "telegram"
    DESKTOP = "desktop"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    SMS = "sms"

def init_notification_db():
    """Initialize notification system database tables."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Notification rules table - defines when and how to notify
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS notification_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        server_id INTEGER,
        channel_id INTEGER,
        keywords TEXT,  -- JSON array of keywords
        priority INTEGER NOT NULL,  -- 1=LOW, 2=MEDIUM, 3=HIGH, 4=URGENT
        channels TEXT NOT NULL,  -- JSON array of notification channels
        conditions TEXT,  -- JSON object with conditions (time_range, user_mentions, etc.)
        is_active BOOLEAN DEFAULT 1,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        FOREIGN KEY (server_id) REFERENCES servers(id),
        FOREIGN KEY (channel_id) REFERENCES channels(id)
    )
    ''')
    
    # Notification queue table - stores pending notifications
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS notification_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rule_id INTEGER NOT NULL,
        message_id INTEGER,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        priority INTEGER NOT NULL,
        channels TEXT NOT NULL,  -- JSON array
        status TEXT DEFAULT 'pending',
        scheduled_at INTEGER,
        created_at INTEGER NOT NULL,
        sent_at INTEGER,
        error_message TEXT,
        metadata TEXT,  -- JSON object for channel-specific data
        FOREIGN KEY (rule_id) REFERENCES notification_rules(id),
        FOREIGN KEY (message_id) REFERENCES messages(id)
    )
    ''')
    
    # Notification channels configuration
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS notification_channels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_type TEXT NOT NULL,  -- email, telegram, etc.
        name TEXT NOT NULL,
        config TEXT NOT NULL,  -- JSON configuration
        is_active BOOLEAN DEFAULT 1,
        rate_limit_per_hour INTEGER DEFAULT 60,
        rate_limit_per_day INTEGER DEFAULT 500,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
    )
    ''')
    
    # User preferences table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_identifier TEXT NOT NULL,  -- email, telegram_id, etc.
        channel_type TEXT NOT NULL,
        preferences TEXT NOT NULL,  -- JSON preferences
        do_not_disturb_start INTEGER,  -- Hour of day (0-23)
        do_not_disturb_end INTEGER,    -- Hour of day (0-23)
        timezone TEXT DEFAULT 'UTC',
        is_active BOOLEAN DEFAULT 1,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
    )
    ''')
    
    # Notification delivery log
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS notification_delivery_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        notification_id INTEGER NOT NULL,
        channel_type TEXT NOT NULL,
        recipient TEXT NOT NULL,
        status TEXT NOT NULL,  -- sent, failed, bounced
        response_data TEXT,  -- JSON response from service
        delivered_at INTEGER,
        error_message TEXT,
        FOREIGN KEY (notification_id) REFERENCES notification_queue(id)
    )
    ''')
    
    # External integrations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS external_integrations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        integration_type TEXT NOT NULL,  -- calendar, todoist, notion, etc.
        name TEXT NOT NULL,
        config TEXT NOT NULL,  -- JSON configuration including auth tokens
        is_active BOOLEAN DEFAULT 1,
        last_sync INTEGER,
        error_count INTEGER DEFAULT 0,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
    )
    ''')
    
    # API tokens and webhooks
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS api_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        token_hash TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        permissions TEXT NOT NULL,  -- JSON array of permissions
        rate_limit_per_hour INTEGER DEFAULT 100,
        rate_limit_per_day INTEGER DEFAULT 1000,
        is_active BOOLEAN DEFAULT 1,
        last_used INTEGER,
        created_at INTEGER NOT NULL,
        expires_at INTEGER
    )
    ''')
    
    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_notification_queue_status ON notification_queue(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_notification_queue_scheduled ON notification_queue(scheduled_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_notification_rules_server ON notification_rules(server_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_notification_rules_active ON notification_rules(is_active)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_delivery_log_notification ON notification_delivery_log(notification_id)')
    
    conn.commit()
    conn.close()
    logger.info("Notification database tables initialized.")

class NotificationRuleManager:
    """Manages notification rules and triggers."""
    
    def create_rule(self, name: str, server_id: Optional[int] = None, 
                   channel_id: Optional[int] = None, keywords: List[str] = None,
                   priority: NotificationPriority = NotificationPriority.MEDIUM,
                   channels: List[NotificationChannel] = None,
                   conditions: Dict[str, Any] = None) -> int:
        """Create a new notification rule."""
        if channels is None:
            channels = [NotificationChannel.EMAIL]
        if conditions is None:
            conditions = {}
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        now = int(datetime.now().timestamp())
        
        cursor.execute('''
        INSERT INTO notification_rules 
        (name, server_id, channel_id, keywords, priority, channels, conditions, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            name, server_id, channel_id, 
            json.dumps(keywords) if keywords else None,
            priority.value,
            json.dumps([c.value for c in channels]),
            json.dumps(conditions),
            now, now
        ))
        
        rule_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Created notification rule '{name}' with ID {rule_id}")
        return rule_id
    
    def get_active_rules(self) -> List[Dict[str, Any]]:
        """Get all active notification rules."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, name, server_id, channel_id, keywords, priority, channels, conditions
        FROM notification_rules
        WHERE is_active = 1
        ORDER BY priority DESC, created_at ASC
        ''')
        
        rules = []
        for row in cursor.fetchall():
            rule = {
                'id': row[0],
                'name': row[1],
                'server_id': row[2],
                'channel_id': row[3],
                'keywords': json.loads(row[4]) if row[4] else [],
                'priority': row[5],
                'channels': json.loads(row[6]),
                'conditions': json.loads(row[7]) if row[7] else {}
            }
            rules.append(rule)
        
        conn.close()
        return rules
    
    def evaluate_message_for_notifications(self, server_id: int, channel_id: int, 
                                         message_content: str, message_id: int) -> List[int]:
        """Evaluate a message against all rules and return matching rule IDs."""
        rules = self.get_active_rules()
        matching_rules = []
        
        for rule in rules:
            if self._message_matches_rule(rule, server_id, channel_id, message_content):
                matching_rules.append(rule['id'])
                logger.debug(f"Message {message_id} matches rule '{rule['name']}'")
        
        return matching_rules
    
    def _message_matches_rule(self, rule: Dict[str, Any], server_id: int, 
                            channel_id: int, message_content: str) -> bool:
        """Check if a message matches a notification rule."""
        # Check server filter
        if rule['server_id'] and rule['server_id'] != server_id:
            return False
        
        # Check channel filter
        if rule['channel_id'] and rule['channel_id'] != channel_id:
            return False
        
        # Check keywords
        if rule['keywords']:
            content_lower = message_content.lower()
            keyword_match = any(keyword.lower() in content_lower for keyword in rule['keywords'])
            if not keyword_match:
                return False
        
        # Check additional conditions
        conditions = rule['conditions']
        
        # Time-based conditions
        if 'time_range' in conditions:
            current_hour = datetime.now().hour
            start_hour = conditions['time_range'].get('start', 0)
            end_hour = conditions['time_range'].get('end', 23)
            
            if start_hour <= end_hour:
                if not (start_hour <= current_hour <= end_hour):
                    return False
            else:  # Crosses midnight
                if not (current_hour >= start_hour or current_hour <= end_hour):
                    return False
        
        # Message length conditions
        if 'min_length' in conditions:
            if len(message_content) < conditions['min_length']:
                return False
        
        # User mention conditions
        if 'require_mentions' in conditions and conditions['require_mentions']:
            if '@' not in message_content:
                return False
        
        return True

class NotificationQueue:
    """Manages the notification queue and delivery."""
    
    def enqueue_notification(self, rule_id: int, message_id: int, title: str, 
                           content: str, priority: NotificationPriority,
                           channels: List[NotificationChannel],
                           schedule_at: Optional[datetime] = None,
                           metadata: Dict[str, Any] = None) -> int:
        """Add a notification to the queue."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        now = int(datetime.now().timestamp())
        scheduled_at = int(schedule_at.timestamp()) if schedule_at else now
        
        cursor.execute('''
        INSERT INTO notification_queue 
        (rule_id, message_id, title, content, priority, channels, scheduled_at, created_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            rule_id, message_id, title, content, priority.value,
            json.dumps([c.value for c in channels]),
            scheduled_at, now,
            json.dumps(metadata) if metadata else None
        ))
        
        notification_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Enqueued notification {notification_id} for rule {rule_id}")
        return notification_id
    
    def get_pending_notifications(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get pending notifications ready for delivery."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        now = int(datetime.now().timestamp())
        
        cursor.execute('''
        SELECT id, rule_id, message_id, title, content, priority, channels, metadata
        FROM notification_queue
        WHERE status = 'pending' AND scheduled_at <= ?
        ORDER BY priority DESC, created_at ASC
        LIMIT ?
        ''', (now, limit))
        
        notifications = []
        for row in cursor.fetchall():
            notification = {
                'id': row[0],
                'rule_id': row[1],
                'message_id': row[2],
                'title': row[3],
                'content': row[4],
                'priority': row[5],
                'channels': json.loads(row[6]),
                'metadata': json.loads(row[7]) if row[7] else {}
            }
            notifications.append(notification)
        
        conn.close()
        return notifications
    
    def mark_notification_sent(self, notification_id: int, error_message: str = None):
        """Mark a notification as sent or failed."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        status = 'failed' if error_message else 'sent'
        sent_at = int(datetime.now().timestamp())
        
        cursor.execute('''
        UPDATE notification_queue 
        SET status = ?, sent_at = ?, error_message = ?
        WHERE id = ?
        ''', (status, sent_at, error_message, notification_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Marked notification {notification_id} as {status}")

# Initialize the notification database when module is imported
try:
    init_notification_db()
except Exception as e:
    logger.error(f"Failed to initialize notification database: {e}")