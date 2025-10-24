"""
Smart notification engine that processes messages and delivers notifications
with priority routing, batching, deduplication, and Do Not Disturb features.
"""

import logging
import sqlite3
import json
import time
import threading
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import hashlib
import re

from .notification_database import (
    NotificationRuleManager, NotificationQueue, NotificationPriority, 
    NotificationStatus, NotificationChannel as DBNotificationChannel, DB_NAME
)
from .notification_channels import NotificationChannelFactory, NotificationChannel
from .database import get_recent_messages

logger = logging.getLogger('notification_engine')

@dataclass
class NotificationBatch:
    """Represents a batch of notifications to be sent together."""
    priority: NotificationPriority
    channel_type: str
    notifications: List[Dict[str, Any]]
    scheduled_at: datetime
    created_at: datetime

class NotificationDeduplicator:
    """Handles deduplication of notifications to prevent spam."""
    
    def __init__(self, time_window_minutes: int = 30):
        self.time_window = timedelta(minutes=time_window_minutes)
        self.recent_notifications: deque = deque()
        self._lock = threading.Lock()
    
    def is_duplicate(self, title: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Check if this notification is a duplicate of a recent one."""
        with self._lock:
            # Clean old notifications
            now = datetime.now()
            while self.recent_notifications and \
                  (now - self.recent_notifications[0]['timestamp']) > self.time_window:
                self.recent_notifications.popleft()
            
            # Create hash for this notification
            notification_hash = self._create_hash(title, content, metadata)
            
            # Check for duplicates
            for recent in self.recent_notifications:
                if recent['hash'] == notification_hash:
                    logger.debug(f"Duplicate notification detected: {title}")
                    return True
            
            # Add to recent notifications
            self.recent_notifications.append({
                'hash': notification_hash,
                'timestamp': now,
                'title': title
            })
            
            return False
    
    def _create_hash(self, title: str, content: str, metadata: Dict[str, Any]) -> str:
        """Create a hash for notification comparison."""
        # Normalize content for comparison
        normalized_content = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', 
                                  'TIMESTAMP', content)  # Remove timestamps
        normalized_content = re.sub(r'message_id:\s*\d+', '', normalized_content)  # Remove message IDs
        
        hash_input = f"{title}|{normalized_content}|{metadata.get('server_id', '')}|{metadata.get('channel_id', '')}"
        return hashlib.md5(hash_input.encode()).hexdigest()

class DoNotDisturbManager:
    """Manages Do Not Disturb schedules and preferences."""
    
    def __init__(self):
        self.global_dnd = False
        self.global_dnd_until: Optional[datetime] = None
        self.user_preferences = {}
        self._load_preferences()
    
    def _load_preferences(self):
        """Load user DND preferences from database."""
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT user_identifier, channel_type, preferences, 
                   do_not_disturb_start, do_not_disturb_end, timezone
            FROM user_preferences
            WHERE is_active = 1
            ''')
            
            for row in cursor.fetchall():
                user_id, channel_type, prefs_json, dnd_start, dnd_end, timezone = row
                
                if user_id not in self.user_preferences:
                    self.user_preferences[user_id] = {}
                
                self.user_preferences[user_id][channel_type] = {
                    'preferences': json.loads(prefs_json),
                    'dnd_start': dnd_start,
                    'dnd_end': dnd_end,
                    'timezone': timezone or 'UTC'
                }
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to load DND preferences: {e}")
    
    def is_dnd_active(self, channel_type: str, priority: NotificationPriority, 
                     user_identifier: str = None) -> bool:
        """Check if Do Not Disturb is active for this notification."""
        # Global DND check
        if self.global_dnd:
            if self.global_dnd_until and datetime.now() < self.global_dnd_until:
                return priority.value < NotificationPriority.URGENT.value
            elif not self.global_dnd_until:
                return priority.value < NotificationPriority.URGENT.value
        
        # User-specific DND check
        if user_identifier and user_identifier in self.user_preferences:
            user_prefs = self.user_preferences[user_identifier].get(channel_type, {})
            
            dnd_start = user_prefs.get('dnd_start')
            dnd_end = user_prefs.get('dnd_end')
            
            if dnd_start is not None and dnd_end is not None:
                current_hour = datetime.now().hour
                
                if dnd_start <= dnd_end:
                    # Same day DND (e.g., 22:00 to 06:00)
                    if dnd_start <= current_hour <= dnd_end:
                        return priority.value < NotificationPriority.HIGH.value
                else:
                    # Cross-midnight DND (e.g., 22:00 to 06:00)
                    if current_hour >= dnd_start or current_hour <= dnd_end:
                        return priority.value < NotificationPriority.HIGH.value
        
        return False
    
    def set_global_dnd(self, enabled: bool, duration_minutes: int = None):
        """Set global Do Not Disturb mode."""
        self.global_dnd = enabled
        if enabled and duration_minutes:
            self.global_dnd_until = datetime.now() + timedelta(minutes=duration_minutes)
        else:
            self.global_dnd_until = None
        
        logger.info(f"Global DND {'enabled' if enabled else 'disabled'}" + 
                   (f" until {self.global_dnd_until}" if self.global_dnd_until else ""))

class NotificationBatcher:
    """Batches notifications to reduce spam and improve user experience."""
    
    def __init__(self, batch_config: Dict[str, Any]):
        self.batch_size = batch_config.get('batch_size', 5)
        self.batch_timeout_minutes = batch_config.get('timeout_minutes', 15)
        self.pending_batches: Dict[str, NotificationBatch] = {}
        self._lock = threading.Lock()
    
    def add_notification(self, notification: Dict[str, Any], channel_type: str, 
                        priority: NotificationPriority) -> Optional[NotificationBatch]:
        """Add notification to batch. Returns completed batch if ready."""
        with self._lock:
            batch_key = f"{channel_type}_{priority.value}"
            
            # Create new batch if needed
            if batch_key not in self.pending_batches:
                self.pending_batches[batch_key] = NotificationBatch(
                    priority=priority,
                    channel_type=channel_type,
                    notifications=[],
                    scheduled_at=datetime.now() + timedelta(minutes=self.batch_timeout_minutes),
                    created_at=datetime.now()
                )
            
            batch = self.pending_batches[batch_key]
            batch.notifications.append(notification)
            
            # Check if batch is ready
            if (len(batch.notifications) >= self.batch_size or 
                priority.value >= NotificationPriority.HIGH.value):
                completed_batch = batch
                del self.pending_batches[batch_key]
                return completed_batch
            
            return None
    
    def get_ready_batches(self) -> List[NotificationBatch]:
        """Get batches that are ready to be sent (timeout expired)."""
        ready_batches = []
        now = datetime.now()
        
        with self._lock:
            keys_to_remove = []
            for key, batch in self.pending_batches.items():
                if now >= batch.scheduled_at:
                    ready_batches.append(batch)
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.pending_batches[key]
        
        return ready_batches
    
    def force_flush_all(self) -> List[NotificationBatch]:
        """Force flush all pending batches."""
        with self._lock:
            batches = list(self.pending_batches.values())
            self.pending_batches.clear()
            return batches

class SmartNotificationEngine:
    """Main notification engine that orchestrates all notification logic."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rule_manager = NotificationRuleManager()
        self.queue = NotificationQueue()
        self.deduplicator = NotificationDeduplicator(
            config.get('dedup_window_minutes', 30)
        )
        self.dnd_manager = DoNotDisturbManager()
        self.batcher = NotificationBatcher(config.get('batching', {}))
        
        # Channel configurations
        self.channels: Dict[str, NotificationChannel] = {}
        self._load_notification_channels()
        
        # Processing thread
        self.running = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'processed': 0,
            'sent': 0,
            'failed': 0,
            'deduplicated': 0,
            'dnd_blocked': 0
        }
    
    def _load_notification_channels(self):
        """Load and initialize notification channels from database."""
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT channel_type, name, config
            FROM notification_channels
            WHERE is_active = 1
            ''')
            
            for row in cursor.fetchall():
                channel_type, name, config_json = row
                try:
                    config = json.loads(config_json)
                    channel = NotificationChannelFactory.create_channel(channel_type, config)
                    self.channels[f"{channel_type}_{name}"] = channel
                    logger.info(f"Loaded notification channel: {channel_type}_{name}")
                except Exception as e:
                    logger.error(f"Failed to load channel {channel_type}_{name}: {e}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to load notification channels: {e}")
    
    def start(self):
        """Start the notification engine."""
        if self.running:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_notifications, daemon=True)
        self.processing_thread.start()
        logger.info("Notification engine started")
    
    def stop(self):
        """Stop the notification engine."""
        if not self.running:
            return
        
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("Notification engine stopped")
    
    def process_new_message(self, server_id: int, channel_id: int, message_id: int, 
                          content: str, author: str = None, metadata: Dict[str, Any] = None):
        """Process a new Discord message for notifications."""
        try:
            # Find matching rules
            matching_rules = self.rule_manager.evaluate_message_for_notifications(
                server_id, channel_id, content, message_id
            )
            
            if not matching_rules:
                return
            
            # Get additional metadata
            enhanced_metadata = metadata or {}
            enhanced_metadata.update({
                'server_id': server_id,
                'channel_id': channel_id,
                'message_id': message_id,
                'author': author,
                'timestamp': datetime.now().isoformat()
            })
            
            # Create notifications for each matching rule
            for rule_id in matching_rules:
                self._create_notification_for_rule(rule_id, content, enhanced_metadata)
            
            self.stats['processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing message {message_id}: {e}")
    
    def _create_notification_for_rule(self, rule_id: int, content: str, 
                                    metadata: Dict[str, Any]):
        """Create a notification for a specific rule."""
        try:
            # Get rule details
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT name, priority, channels, conditions
            FROM notification_rules
            WHERE id = ? AND is_active = 1
            ''', (rule_id,))
            
            rule = cursor.fetchone()
            conn.close()
            
            if not rule:
                return
            
            rule_name, priority_value, channels_json, conditions_json = rule
            priority = NotificationPriority(priority_value)
            channels = json.loads(channels_json)
            conditions = json.loads(conditions_json) if conditions_json else {}
            
            # Create notification title and content
            title = f"Discord Alert: {rule_name}"
            
            # Apply content formatting based on conditions
            if conditions.get('include_context'):
                # Get some context messages
                context_messages = get_recent_messages(
                    metadata['server_id'], 
                    hours=1, 
                    channel_id=metadata['channel_id']
                )
                if len(context_messages) > 1:
                    content = f"Recent context:\n" + "\n".join(context_messages[-3:-1]) + f"\n\n**New message:**\n{content}"
            
            # Truncate content if too long
            max_length = conditions.get('max_content_length', 500)
            if len(content) > max_length:
                content = content[:max_length-3] + "..."
            
            # Check for deduplication
            if self.deduplicator.is_duplicate(title, content, metadata):
                self.stats['deduplicated'] += 1
                return
            
            # Queue notification for each channel
            for channel_type in channels:
                # Check DND status
                if self.dnd_manager.is_dnd_active(channel_type, priority):
                    self.stats['dnd_blocked'] += 1
                    logger.debug(f"Notification blocked by DND: {title}")
                    continue
                
                # Add to queue
                notification_id = self.queue.enqueue_notification(
                    rule_id=rule_id,
                    message_id=metadata['message_id'],
                    title=title,
                    content=content,
                    priority=priority,
                    channels=[DBNotificationChannel(channel_type)],
                    metadata=metadata
                )
                
                logger.debug(f"Queued notification {notification_id} for {channel_type}")
        
        except Exception as e:
            logger.error(f"Error creating notification for rule {rule_id}: {e}")
    
    def _process_notifications(self):
        """Main processing loop for notifications."""
        logger.info("Starting notification processing loop")
        
        while self.running:
            try:
                # Get pending notifications
                pending = self.queue.get_pending_notifications(limit=50)
                
                for notification in pending:
                    self._handle_notification(notification)
                
                # Process ready batches
                ready_batches = self.batcher.get_ready_batches()
                for batch in ready_batches:
                    self._send_batch(batch)
                
                # Sleep before next iteration
                time.sleep(self.config.get('processing_interval_seconds', 10))
                
            except Exception as e:
                logger.error(f"Error in notification processing loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _handle_notification(self, notification: Dict[str, Any]):
        """Handle a single notification."""
        try:
            channels = notification['channels']
            priority = NotificationPriority(notification['priority'])
            
            # Try to batch notifications for better UX
            for channel_type in channels:
                # Check if we should batch this notification
                batch_config = self.config.get('batching', {})
                should_batch = (
                    priority.value < NotificationPriority.HIGH.value and
                    batch_config.get('enabled', True) and
                    channel_type in batch_config.get('batch_channels', ['email'])
                )
                
                if should_batch:
                    completed_batch = self.batcher.add_notification(
                        notification, channel_type, priority
                    )
                    if completed_batch:
                        self._send_batch(completed_batch)
                else:
                    # Send immediately
                    self._send_notification(notification, channel_type)
        
        except Exception as e:
            logger.error(f"Error handling notification {notification.get('id')}: {e}")
            self.queue.mark_notification_sent(notification['id'], str(e))
    
    def _send_notification(self, notification: Dict[str, Any], channel_type: str):
        """Send a single notification."""
        notification_id = notification['id']
        
        try:
            # Find appropriate channel
            channel = self._find_channel(channel_type)
            if not channel:
                error_msg = f"No active channel found for type: {channel_type}"
                logger.warning(error_msg)
                self.queue.mark_notification_sent(notification_id, error_msg)
                return
            
            # Send notification
            success, error_msg = channel.send(
                title=notification['title'],
                content=notification['content'],
                metadata=notification['metadata']
            )
            
            # Update notification status
            self.queue.mark_notification_sent(notification_id, error_msg if not success else None)
            
            # Update statistics
            if success:
                self.stats['sent'] += 1
                logger.info(f"Sent notification {notification_id} via {channel_type}")
            else:
                self.stats['failed'] += 1
                logger.error(f"Failed to send notification {notification_id}: {error_msg}")
        
        except Exception as e:
            error_msg = f"Unexpected error sending notification: {str(e)}"
            logger.error(error_msg)
            self.queue.mark_notification_sent(notification_id, error_msg)
            self.stats['failed'] += 1
    
    def _send_batch(self, batch: NotificationBatch):
        """Send a batch of notifications."""
        try:
            channel = self._find_channel(batch.channel_type)
            if not channel:
                logger.warning(f"No active channel found for batch type: {batch.channel_type}")
                return
            
            # Create batch content
            title = f"Discord Digest: {len(batch.notifications)} notifications"
            content = self._format_batch_content(batch)
            
            # Combined metadata
            combined_metadata = {
                'batch_size': len(batch.notifications),
                'priority': batch.priority.name,
                'notifications': batch.notifications
            }
            
            # Send batch
            success, error_msg = channel.send(title, content, combined_metadata)
            
            # Mark all notifications as sent
            for notification in batch.notifications:
                self.queue.mark_notification_sent(
                    notification['id'], 
                    error_msg if not success else None
                )
            
            # Update statistics
            if success:
                self.stats['sent'] += len(batch.notifications)
                logger.info(f"Sent batch of {len(batch.notifications)} notifications via {batch.channel_type}")
            else:
                self.stats['failed'] += len(batch.notifications)
                logger.error(f"Failed to send batch: {error_msg}")
        
        except Exception as e:
            logger.error(f"Error sending batch: {e}")
    
    def _find_channel(self, channel_type: str) -> Optional[NotificationChannel]:
        """Find an active channel of the specified type."""
        for key, channel in self.channels.items():
            if key.startswith(f"{channel_type}_"):
                return channel
        return None
    
    def _format_batch_content(self, batch: NotificationBatch) -> str:
        """Format content for a batch notification."""
        content = f"You have {len(batch.notifications)} new Discord notifications:\n\n"
        
        for i, notification in enumerate(batch.notifications[:10], 1):  # Limit to 10
            content += f"{i}. **{notification['title']}**\n"
            # Truncate individual notification content
            notif_content = notification['content']
            if len(notif_content) > 100:
                notif_content = notif_content[:97] + "..."
            content += f"   {notif_content}\n\n"
        
        if len(batch.notifications) > 10:
            content += f"... and {len(batch.notifications) - 10} more notifications\n"
        
        return content
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification engine statistics."""
        stats = self.stats.copy()
        stats['channels_active'] = len(self.channels)
        stats['rules_active'] = len(self.rule_manager.get_active_rules())
        return stats
    
    def reload_channels(self):
        """Reload notification channels from database."""
        self.channels.clear()
        self._load_notification_channels()
        logger.info("Notification channels reloaded")

# Global notification engine instance
_notification_engine: Optional[SmartNotificationEngine] = None

def get_notification_engine(config: Dict[str, Any] = None) -> SmartNotificationEngine:
    """Get or create the global notification engine instance."""
    global _notification_engine
    
    if _notification_engine is None:
        if config is None:
            config = {
                'processing_interval_seconds': 10,
                'dedup_window_minutes': 30,
                'batching': {
                    'enabled': True,
                    'batch_size': 5,
                    'timeout_minutes': 15,
                    'batch_channels': ['email']
                }
            }
        _notification_engine = SmartNotificationEngine(config)
    
    return _notification_engine