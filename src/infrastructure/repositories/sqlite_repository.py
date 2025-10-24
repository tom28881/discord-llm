"""
SQLite implementation of repository interfaces.
This provides concrete implementations for data persistence using SQLite.
"""
import asyncio
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import aiosqlite

from ...core.domain.models import (
    Server, Channel, User, Message, MonitoringEvent, 
    ImportanceLevel, EventType, MessageType
)
from ...core.interfaces.repositories import (
    IServerRepository, IChannelRepository, IUserRepository, 
    IMessageRepository, IEventRepository, IConfigRepository
)

logger = logging.getLogger(__name__)


class SqliteRepositoryBase:
    """Base class for SQLite repositories with common functionality."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_db_directory()
    
    def _ensure_db_directory(self) -> None:
        """Ensure database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    async def _execute_query(
        self, 
        query: str, 
        params: tuple = (), 
        fetch_one: bool = False,
        fetch_all: bool = False
    ) -> Optional[Union[sqlite3.Row, List[sqlite3.Row]]]:
        """Execute a database query safely."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, params) as cursor:
                    if fetch_one:
                        return await cursor.fetchone()
                    elif fetch_all:
                        return await cursor.fetchall()
                    else:
                        await db.commit()
                        return None
        except Exception as e:
            logger.error(f"Database query failed: {query[:100]}... Error: {e}")
            raise


class SqliteServerRepository(SqliteRepositoryBase, IServerRepository):
    """SQLite implementation of server repository."""
    
    async def initialize_tables(self) -> None:
        """Initialize server-related tables."""
        create_servers_table = """
        CREATE TABLE IF NOT EXISTS servers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            metadata TEXT DEFAULT '{}'
        )
        """
        await self._execute_query(create_servers_table)
    
    async def get_by_id(self, server_id: str) -> Optional[Server]:
        """Get server by ID."""
        query = "SELECT * FROM servers WHERE id = ?"
        row = await self._execute_query(query, (server_id,), fetch_one=True)
        
        if row:
            return Server(
                id=row['id'],
                name=row['name'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.utcnow(),
                is_active=bool(row['is_active']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
        return None
    
    async def get_all_active(self) -> List[Server]:
        """Get all active servers."""
        query = "SELECT * FROM servers WHERE is_active = 1 ORDER BY name"
        rows = await self._execute_query(query, fetch_all=True)
        
        servers = []
        for row in rows:
            servers.append(Server(
                id=row['id'],
                name=row['name'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.utcnow(),
                is_active=bool(row['is_active']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))
        
        return servers
    
    async def save(self, server: Server) -> Server:
        """Save or update server."""
        query = """
        INSERT OR REPLACE INTO servers (id, name, created_at, is_active, metadata)
        VALUES (?, ?, ?, ?, ?)
        """
        
        await self._execute_query(
            query, 
            (
                server.id,
                server.name,
                server.created_at.isoformat(),
                server.is_active,
                json.dumps(server.metadata)
            )
        )
        
        return server
    
    async def delete(self, server_id: str) -> bool:
        """Delete server by ID."""
        try:
            query = "DELETE FROM servers WHERE id = ?"
            await self._execute_query(query, (server_id,))
            return True
        except Exception as e:
            logger.error(f"Failed to delete server {server_id}: {e}")
            return False


class SqliteChannelRepository(SqliteRepositoryBase, IChannelRepository):
    """SQLite implementation of channel repository."""
    
    async def initialize_tables(self) -> None:
        """Initialize channel-related tables."""
        create_channels_table = """
        CREATE TABLE IF NOT EXISTS channels (
            id TEXT PRIMARY KEY,
            server_id TEXT NOT NULL,
            name TEXT NOT NULL,
            channel_type TEXT DEFAULT 'text',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_monitored BOOLEAN DEFAULT 1,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (server_id) REFERENCES servers(id)
        )
        """
        
        create_index = "CREATE INDEX IF NOT EXISTS idx_channels_server_id ON channels(server_id)"
        
        await self._execute_query(create_channels_table)
        await self._execute_query(create_index)
    
    async def get_by_id(self, channel_id: str) -> Optional[Channel]:
        """Get channel by ID."""
        query = "SELECT * FROM channels WHERE id = ?"
        row = await self._execute_query(query, (channel_id,), fetch_one=True)
        
        if row:
            return Channel(
                id=row['id'],
                server_id=row['server_id'],
                name=row['name'],
                channel_type=row['channel_type'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.utcnow(),
                is_monitored=bool(row['is_monitored']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
        return None
    
    async def get_by_server(self, server_id: str) -> List[Channel]:
        """Get all channels for a server."""
        query = "SELECT * FROM channels WHERE server_id = ? ORDER BY name"
        rows = await self._execute_query(query, (server_id,), fetch_all=True)
        
        channels = []
        for row in rows:
            channels.append(Channel(
                id=row['id'],
                server_id=row['server_id'],
                name=row['name'],
                channel_type=row['channel_type'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.utcnow(),
                is_monitored=bool(row['is_monitored']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))
        
        return channels
    
    async def get_monitored_channels(self) -> List[Channel]:
        """Get all monitored channels."""
        query = "SELECT * FROM channels WHERE is_monitored = 1 ORDER BY server_id, name"
        rows = await self._execute_query(query, fetch_all=True)
        
        channels = []
        for row in rows:
            channels.append(Channel(
                id=row['id'],
                server_id=row['server_id'],
                name=row['name'],
                channel_type=row['channel_type'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.utcnow(),
                is_monitored=bool(row['is_monitored']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))
        
        return channels
    
    async def save(self, channel: Channel) -> Channel:
        """Save or update channel."""
        query = """
        INSERT OR REPLACE INTO channels (id, server_id, name, channel_type, created_at, is_monitored, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        await self._execute_query(
            query,
            (
                channel.id,
                channel.server_id,
                channel.name,
                channel.channel_type,
                channel.created_at.isoformat(),
                channel.is_monitored,
                json.dumps(channel.metadata)
            )
        )
        
        return channel
    
    async def update_monitoring_status(self, channel_id: str, is_monitored: bool) -> bool:
        """Update channel monitoring status."""
        try:
            query = "UPDATE channels SET is_monitored = ? WHERE id = ?"
            await self._execute_query(query, (is_monitored, channel_id))
            return True
        except Exception as e:
            logger.error(f"Failed to update monitoring status for channel {channel_id}: {e}")
            return False


class SqliteUserRepository(SqliteRepositoryBase, IUserRepository):
    """SQLite implementation of user repository."""
    
    async def initialize_tables(self) -> None:
        """Initialize user-related tables."""
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            display_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}'
        )
        """
        await self._execute_query(create_users_table)
    
    async def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        query = "SELECT * FROM users WHERE id = ?"
        row = await self._execute_query(query, (user_id,), fetch_one=True)
        
        if row:
            return User(
                id=row['id'],
                username=row['username'],
                display_name=row['display_name'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.utcnow(),
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
        return None
    
    async def save(self, user: User) -> User:
        """Save or update user."""
        query = """
        INSERT OR REPLACE INTO users (id, username, display_name, created_at, metadata)
        VALUES (?, ?, ?, ?, ?)
        """
        
        await self._execute_query(
            query,
            (
                user.id,
                user.username,
                user.display_name,
                user.created_at.isoformat(),
                json.dumps(user.metadata)
            )
        )
        
        return user
    
    async def get_or_create(self, user_id: str, username: str, display_name: Optional[str] = None) -> User:
        """Get existing user or create new one."""
        existing_user = await self.get_by_id(user_id)
        if existing_user:
            return existing_user
        
        new_user = User(
            id=user_id,
            username=username,
            display_name=display_name
        )
        
        return await self.save(new_user)


class SqliteMessageRepository(SqliteRepositoryBase, IMessageRepository):
    """SQLite implementation of message repository."""
    
    async def initialize_tables(self) -> None:
        """Initialize message-related tables."""
        create_messages_table = """
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            server_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            content TEXT,
            message_type TEXT DEFAULT 'text',
            sent_at TIMESTAMP NOT NULL,
            importance_level TEXT DEFAULT 'medium',
            importance_score REAL DEFAULT 0.5,
            keywords_matched TEXT DEFAULT '[]',
            mentions TEXT DEFAULT '[]',
            attachments TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            processed_at TIMESTAMP,
            FOREIGN KEY (server_id) REFERENCES servers(id),
            FOREIGN KEY (channel_id) REFERENCES channels(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
        
        # Create indexes for better query performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_messages_server_id ON messages(server_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_channel_id ON messages(channel_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_sent_at ON messages(sent_at)",
            "CREATE INDEX IF NOT EXISTS idx_messages_importance ON messages(importance_level, importance_score)",
            "CREATE INDEX IF NOT EXISTS idx_messages_processed_at ON messages(processed_at)"
        ]
        
        await self._execute_query(create_messages_table)
        for index_query in indexes:
            await self._execute_query(index_query)
    
    async def get_by_id(self, message_id: str) -> Optional[Message]:
        """Get message by ID."""
        query = "SELECT * FROM messages WHERE id = ?"
        row = await self._execute_query(query, (message_id,), fetch_one=True)
        
        if row:
            return self._row_to_message(row)
        return None
    
    async def get_recent_messages(
        self,
        server_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        hours: int = 24,
        importance_level: Optional[ImportanceLevel] = None,
        keywords: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[Message]:
        """Get recent messages with filters."""
        
        # Build dynamic query
        base_query = "SELECT * FROM messages WHERE sent_at >= ?"
        params = [datetime.utcnow() - timedelta(hours=hours)]
        
        if server_id:
            base_query += " AND server_id = ?"
            params.append(server_id)
        
        if channel_id:
            base_query += " AND channel_id = ?"
            params.append(channel_id)
        
        if importance_level:
            # Get messages at or above the specified importance level
            importance_order = {
                ImportanceLevel.NOISE: 1,
                ImportanceLevel.LOW: 2,
                ImportanceLevel.MEDIUM: 3,
                ImportanceLevel.HIGH: 4,
                ImportanceLevel.CRITICAL: 5
            }
            
            min_level_value = importance_order[importance_level]
            level_conditions = []
            for level, value in importance_order.items():
                if value >= min_level_value:
                    level_conditions.append(f"importance_level = ?")
                    params.append(level.value)
            
            if level_conditions:
                base_query += f" AND ({' OR '.join(level_conditions)})"
        
        if keywords:
            # Search for keywords in content (case-insensitive)
            keyword_conditions = []
            for keyword in keywords:
                keyword_conditions.append("content LIKE ? COLLATE NOCASE")
                params.append(f"%{keyword}%")
            
            if keyword_conditions:
                base_query += f" AND ({' OR '.join(keyword_conditions)})"
        
        base_query += " ORDER BY sent_at DESC LIMIT ?"
        params.append(limit)
        
        rows = await self._execute_query(base_query, tuple(params), fetch_all=True)
        
        messages = []
        for row in rows:
            messages.append(self._row_to_message(row))
        
        return messages
    
    async def get_last_message_id(self, channel_id: str) -> Optional[str]:
        """Get the ID of the last message in a channel."""
        query = "SELECT id FROM messages WHERE channel_id = ? ORDER BY sent_at DESC LIMIT 1"
        row = await self._execute_query(query, (channel_id,), fetch_one=True)
        
        return row['id'] if row else None
    
    async def save(self, message: Message) -> Message:
        """Save message."""
        query = """
        INSERT OR REPLACE INTO messages (
            id, server_id, channel_id, user_id, content, message_type,
            sent_at, importance_level, importance_score, keywords_matched,
            mentions, attachments, metadata, processed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        await self._execute_query(
            query,
            (
                message.id,
                message.server_id,
                message.channel_id,
                message.user_id,
                message.content,
                message.message_type.value,
                message.sent_at.isoformat(),
                message.importance_level.value,
                message.importance_score,
                json.dumps(message.keywords_matched),
                json.dumps(message.mentions),
                json.dumps(message.attachments),
                json.dumps(message.metadata),
                message.processed_at.isoformat() if message.processed_at else None
            )
        )
        
        return message
    
    async def save_batch(self, messages: List[Message]) -> List[Message]:
        """Save multiple messages efficiently."""
        if not messages:
            return []
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                query = """
                INSERT OR REPLACE INTO messages (
                    id, server_id, channel_id, user_id, content, message_type,
                    sent_at, importance_level, importance_score, keywords_matched,
                    mentions, attachments, metadata, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                batch_data = []
                for message in messages:
                    batch_data.append((
                        message.id,
                        message.server_id,
                        message.channel_id,
                        message.user_id,
                        message.content,
                        message.message_type.value,
                        message.sent_at.isoformat(),
                        message.importance_level.value,
                        message.importance_score,
                        json.dumps(message.keywords_matched),
                        json.dumps(message.mentions),
                        json.dumps(message.attachments),
                        json.dumps(message.metadata),
                        message.processed_at.isoformat() if message.processed_at else None
                    ))
                
                await db.executemany(query, batch_data)
                await db.commit()
                
            return messages
            
        except Exception as e:
            logger.error(f"Failed to save message batch: {e}")
            raise
    
    async def update_importance(self, message_id: str, importance_level: ImportanceLevel, score: float) -> bool:
        """Update message importance."""
        try:
            query = """
            UPDATE messages 
            SET importance_level = ?, importance_score = ?, processed_at = ?
            WHERE id = ?
            """
            
            await self._execute_query(
                query, 
                (importance_level.value, score, datetime.utcnow().isoformat(), message_id)
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to update importance for message {message_id}: {e}")
            return False
    
    async def get_unprocessed_messages(self, limit: int = 100) -> List[Message]:
        """Get messages that haven't been processed for importance."""
        query = """
        SELECT * FROM messages 
        WHERE processed_at IS NULL 
        ORDER BY sent_at ASC 
        LIMIT ?
        """
        
        rows = await self._execute_query(query, (limit,), fetch_all=True)
        
        messages = []
        for row in rows:
            messages.append(self._row_to_message(row))
        
        return messages
    
    def _row_to_message(self, row) -> Message:
        """Convert database row to Message object."""
        return Message(
            id=row['id'],
            server_id=row['server_id'],
            channel_id=row['channel_id'],
            user_id=row['user_id'],
            content=row['content'] or "",
            message_type=MessageType(row['message_type']) if row['message_type'] else MessageType.TEXT,
            sent_at=datetime.fromisoformat(row['sent_at']),
            importance_level=ImportanceLevel(row['importance_level']) if row['importance_level'] else ImportanceLevel.MEDIUM,
            importance_score=float(row['importance_score']) if row['importance_score'] is not None else 0.5,
            keywords_matched=json.loads(row['keywords_matched']) if row['keywords_matched'] else [],
            mentions=json.loads(row['mentions']) if row['mentions'] else [],
            attachments=json.loads(row['attachments']) if row['attachments'] else [],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            processed_at=datetime.fromisoformat(row['processed_at']) if row['processed_at'] else None
        )


class SqliteEventRepository(SqliteRepositoryBase, IEventRepository):
    """SQLite implementation of event repository."""
    
    async def initialize_tables(self) -> None:
        """Initialize event-related tables."""
        create_events_table = """
        CREATE TABLE IF NOT EXISTS monitoring_events (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            message_id TEXT NOT NULL,
            triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            trigger_rules TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
        """
        
        create_index = "CREATE INDEX IF NOT EXISTS idx_events_triggered_at ON monitoring_events(triggered_at)"
        create_type_index = "CREATE INDEX IF NOT EXISTS idx_events_type ON monitoring_events(event_type)"
        
        await self._execute_query(create_events_table)
        await self._execute_query(create_index)
        await self._execute_query(create_type_index)
    
    async def save(self, event: MonitoringEvent) -> MonitoringEvent:
        """Save monitoring event."""
        query = """
        INSERT INTO monitoring_events (id, event_type, message_id, triggered_at, trigger_rules, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        await self._execute_query(
            query,
            (
                str(event.id),
                event.event_type.value,
                event.message.id,
                event.triggered_at.isoformat(),
                json.dumps(event.trigger_rules),
                json.dumps(event.metadata)
            )
        )
        
        return event
    
    async def get_recent_events(
        self,
        hours: int = 24,
        event_types: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[MonitoringEvent]:
        """Get recent monitoring events."""
        
        base_query = "SELECT * FROM monitoring_events WHERE triggered_at >= ?"
        params = [datetime.utcnow() - timedelta(hours=hours)]
        
        if event_types:
            type_conditions = []
            for event_type in event_types:
                type_conditions.append("event_type = ?")
                params.append(event_type)
            
            if type_conditions:
                base_query += f" AND ({' OR '.join(type_conditions)})"
        
        base_query += " ORDER BY triggered_at DESC LIMIT ?"
        params.append(limit)
        
        rows = await self._execute_query(base_query, tuple(params), fetch_all=True)
        
        # Note: This is a simplified version. In a real implementation,
        # you'd want to join with the messages table to get full message data
        events = []
        for row in rows:
            # For now, create a minimal message object - in practice you'd join the tables
            from uuid import UUID
            
            events.append(MonitoringEvent(
                id=UUID(row['id']),
                event_type=EventType(row['event_type']),
                message=None,  # Would be populated from join in real implementation
                triggered_at=datetime.fromisoformat(row['triggered_at']),
                trigger_rules=json.loads(row['trigger_rules']) if row['trigger_rules'] else [],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))
        
        return events
    
    async def get_events_for_message(self, message_id: str) -> List[MonitoringEvent]:
        """Get all events triggered by a specific message."""
        query = "SELECT * FROM monitoring_events WHERE message_id = ? ORDER BY triggered_at DESC"
        rows = await self._execute_query(query, (message_id,), fetch_all=True)
        
        events = []
        for row in rows:
            from uuid import UUID
            
            events.append(MonitoringEvent(
                id=UUID(row['id']),
                event_type=EventType(row['event_type']),
                message=None,  # Would be populated from join in real implementation
                triggered_at=datetime.fromisoformat(row['triggered_at']),
                trigger_rules=json.loads(row['trigger_rules']) if row['trigger_rules'] else [],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))
        
        return events


class SqliteConfigRepository(SqliteRepositoryBase, IConfigRepository):
    """SQLite implementation of configuration repository."""
    
    async def initialize_tables(self) -> None:
        """Initialize configuration tables."""
        create_config_table = """
        CREATE TABLE IF NOT EXISTS configuration (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        create_user_prefs_table = """
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT,
            key TEXT,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, key)
        )
        """
        
        await self._execute_query(create_config_table)
        await self._execute_query(create_user_prefs_table)
    
    async def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        query = "SELECT value FROM configuration WHERE key = ?"
        row = await self._execute_query(query, (key,), fetch_one=True)
        
        if row and row['value'] is not None:
            try:
                return json.loads(row['value'])
            except json.JSONDecodeError:
                return row['value']
        
        return default
    
    async def set_config(self, key: str, value: Any) -> bool:
        """Set configuration value."""
        try:
            json_value = json.dumps(value) if not isinstance(value, str) else value
            
            query = """
            INSERT OR REPLACE INTO configuration (key, value, updated_at)
            VALUES (?, ?, ?)
            """
            
            await self._execute_query(query, (key, json_value, datetime.utcnow().isoformat()))
            return True
            
        except Exception as e:
            logger.error(f"Failed to set configuration {key}: {e}")
            return False
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user-specific preferences."""
        query = "SELECT key, value FROM user_preferences WHERE user_id = ?"
        rows = await self._execute_query(query, (user_id,), fetch_all=True)
        
        preferences = {}
        for row in rows:
            try:
                preferences[row['key']] = json.loads(row['value'])
            except json.JSONDecodeError:
                preferences[row['key']] = row['value']
        
        return preferences
    
    async def set_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """Set user preference."""
        try:
            json_value = json.dumps(value) if not isinstance(value, str) else value
            
            query = """
            INSERT OR REPLACE INTO user_preferences (user_id, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            """
            
            await self._execute_query(
                query, 
                (user_id, key, json_value, datetime.utcnow().isoformat())
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to set user preference {user_id}.{key}: {e}")
            return False
    
    async def get_forbidden_channels(self) -> List[str]:
        """Get list of forbidden channel IDs."""
        forbidden_channels = await self.get_config("forbidden_channels", [])
        return forbidden_channels if isinstance(forbidden_channels, list) else []
    
    async def add_forbidden_channel(self, channel_id: str) -> bool:
        """Add channel to forbidden list."""
        try:
            forbidden_channels = await self.get_forbidden_channels()
            
            if channel_id not in forbidden_channels:
                forbidden_channels.append(channel_id)
                return await self.set_config("forbidden_channels", forbidden_channels)
            
            return True  # Already in list
            
        except Exception as e:
            logger.error(f"Failed to add forbidden channel {channel_id}: {e}")
            return False


class SqliteRepositoryManager:
    """Manager for all SQLite repositories."""
    
    def __init__(self, db_path: str = "data/discord_monitor.db"):
        self.db_path = db_path
        
        # Initialize repositories
        self.server_repository = SqliteServerRepository(db_path)
        self.channel_repository = SqliteChannelRepository(db_path)
        self.user_repository = SqliteUserRepository(db_path)
        self.message_repository = SqliteMessageRepository(db_path)
        self.event_repository = SqliteEventRepository(db_path)
        self.config_repository = SqliteConfigRepository(db_path)
    
    async def initialize_all_tables(self) -> bool:
        """Initialize all database tables."""
        try:
            await self.server_repository.initialize_tables()
            await self.channel_repository.initialize_tables()
            await self.user_repository.initialize_tables()
            await self.message_repository.initialize_tables()
            await self.event_repository.initialize_tables()
            await self.config_repository.initialize_tables()
            
            logger.info(f"Initialized all database tables at {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database tables: {e}")
            return False