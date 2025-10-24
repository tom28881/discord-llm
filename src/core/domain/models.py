"""
Domain models for the Discord monitoring system.
These represent the core business entities and should be framework-agnostic.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4


class MessageType(Enum):
    TEXT = "text"
    EMBED = "embed"
    ATTACHMENT = "attachment"
    SYSTEM = "system"


class ImportanceLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOISE = "noise"


class EventType(Enum):
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_UPDATED = "message_updated"
    MESSAGE_DELETED = "message_deleted"
    USER_MENTIONED = "user_mentioned"
    KEYWORD_TRIGGERED = "keyword_triggered"
    IMPORTANCE_THRESHOLD_EXCEEDED = "importance_threshold_exceeded"


@dataclass(frozen=True)
class Server:
    """Represents a Discord server."""
    id: str
    name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Channel:
    """Represents a Discord channel."""
    id: str
    server_id: str
    name: str
    channel_type: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_monitored: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class User:
    """Represents a Discord user."""
    id: str
    username: str
    display_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """Represents a Discord message with importance scoring."""
    id: str
    server_id: str
    channel_id: str
    user_id: str
    content: str
    message_type: MessageType
    sent_at: datetime
    importance_level: ImportanceLevel = ImportanceLevel.MEDIUM
    importance_score: float = 0.5
    keywords_matched: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed_at: Optional[datetime] = None
    
    @property
    def is_important(self) -> bool:
        """Check if message meets importance threshold."""
        return self.importance_level in [ImportanceLevel.CRITICAL, ImportanceLevel.HIGH]


@dataclass(frozen=True)
class MonitoringEvent:
    """Represents an event in the monitoring system."""
    id: UUID = field(default_factory=uuid4)
    event_type: EventType
    message: Message
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    trigger_rules: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportanceResult:
    """Result of importance analysis."""
    score: float
    level: ImportanceLevel
    reasons: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Result of message processing."""
    message: Message
    importance: ImportanceResult
    events_triggered: List[MonitoringEvent]
    processing_time_ms: float
    success: bool
    error_message: Optional[str] = None