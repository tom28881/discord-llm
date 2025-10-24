"""
Repository interfaces defining data access contracts.
These define the contract for data persistence without implementation details.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any

from ..domain.models import Server, Channel, User, Message, MonitoringEvent, ImportanceLevel


class IServerRepository(ABC):
    """Interface for server data access."""
    
    @abstractmethod
    async def get_by_id(self, server_id: str) -> Optional[Server]:
        """Get server by ID."""
        pass
    
    @abstractmethod
    async def get_all_active(self) -> List[Server]:
        """Get all active servers."""
        pass
    
    @abstractmethod
    async def save(self, server: Server) -> Server:
        """Save or update server."""
        pass
    
    @abstractmethod
    async def delete(self, server_id: str) -> bool:
        """Delete server by ID."""
        pass


class IChannelRepository(ABC):
    """Interface for channel data access."""
    
    @abstractmethod
    async def get_by_id(self, channel_id: str) -> Optional[Channel]:
        """Get channel by ID."""
        pass
    
    @abstractmethod
    async def get_by_server(self, server_id: str) -> List[Channel]:
        """Get all channels for a server."""
        pass
    
    @abstractmethod
    async def get_monitored_channels(self) -> List[Channel]:
        """Get all monitored channels."""
        pass
    
    @abstractmethod
    async def save(self, channel: Channel) -> Channel:
        """Save or update channel."""
        pass
    
    @abstractmethod
    async def update_monitoring_status(self, channel_id: str, is_monitored: bool) -> bool:
        """Update channel monitoring status."""
        pass


class IUserRepository(ABC):
    """Interface for user data access."""
    
    @abstractmethod
    async def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        pass
    
    @abstractmethod
    async def save(self, user: User) -> User:
        """Save or update user."""
        pass
    
    @abstractmethod
    async def get_or_create(self, user_id: str, username: str, display_name: Optional[str] = None) -> User:
        """Get existing user or create new one."""
        pass


class IMessageRepository(ABC):
    """Interface for message data access."""
    
    @abstractmethod
    async def get_by_id(self, message_id: str) -> Optional[Message]:
        """Get message by ID."""
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def get_last_message_id(self, channel_id: str) -> Optional[str]:
        """Get the ID of the last message in a channel."""
        pass
    
    @abstractmethod
    async def save(self, message: Message) -> Message:
        """Save message."""
        pass
    
    @abstractmethod
    async def save_batch(self, messages: List[Message]) -> List[Message]:
        """Save multiple messages efficiently."""
        pass
    
    @abstractmethod
    async def update_importance(self, message_id: str, importance_level: ImportanceLevel, score: float) -> bool:
        """Update message importance."""
        pass
    
    @abstractmethod
    async def get_unprocessed_messages(self, limit: int = 100) -> List[Message]:
        """Get messages that haven't been processed for importance."""
        pass


class IEventRepository(ABC):
    """Interface for monitoring event data access."""
    
    @abstractmethod
    async def save(self, event: MonitoringEvent) -> MonitoringEvent:
        """Save monitoring event."""
        pass
    
    @abstractmethod
    async def get_recent_events(
        self, 
        hours: int = 24, 
        event_types: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[MonitoringEvent]:
        """Get recent monitoring events."""
        pass
    
    @abstractmethod
    async def get_events_for_message(self, message_id: str) -> List[MonitoringEvent]:
        """Get all events triggered by a specific message."""
        pass


class IConfigRepository(ABC):
    """Interface for configuration data access."""
    
    @abstractmethod
    async def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        pass
    
    @abstractmethod
    async def set_config(self, key: str, value: Any) -> bool:
        """Set configuration value."""
        pass
    
    @abstractmethod
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user-specific preferences."""
        pass
    
    @abstractmethod
    async def set_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """Set user preference."""
        pass
    
    @abstractmethod
    async def get_forbidden_channels(self) -> List[str]:
        """Get list of forbidden channel IDs."""
        pass
    
    @abstractmethod
    async def add_forbidden_channel(self, channel_id: str) -> bool:
        """Add channel to forbidden list."""
        pass