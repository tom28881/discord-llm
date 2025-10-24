"""
Service interfaces defining business logic contracts.
These define the contract for core business services without implementation details.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncIterator
from datetime import datetime

from ..domain.models import (
    Server, Channel, User, Message, MonitoringEvent, 
    ImportanceResult, ProcessingResult, ImportanceLevel, EventType
)


class IDiscordService(ABC):
    """Interface for Discord API interactions."""
    
    @abstractmethod
    async def get_servers(self) -> List[Server]:
        """Get all available servers."""
        pass
    
    @abstractmethod
    async def get_channels(self, server_id: str) -> List[Channel]:
        """Get channels for a server."""
        pass
    
    @abstractmethod
    async def fetch_messages(
        self, 
        channel_id: str, 
        since_message_id: Optional[str] = None, 
        limit: int = 100
    ) -> List[Message]:
        """Fetch messages from a channel."""
        pass
    
    @abstractmethod
    async def start_real_time_monitoring(self) -> AsyncIterator[Message]:
        """Start real-time message monitoring across all monitored channels."""
        pass
    
    @abstractmethod
    async def stop_real_time_monitoring(self) -> None:
        """Stop real-time monitoring."""
        pass


class IImportanceAnalyzer(ABC):
    """Interface for message importance analysis."""
    
    @abstractmethod
    async def analyze_importance(self, message: Message) -> ImportanceResult:
        """Analyze the importance of a message."""
        pass
    
    @abstractmethod
    async def update_scoring_rules(self, rules: Dict[str, Any]) -> bool:
        """Update importance scoring rules."""
        pass
    
    @abstractmethod
    async def get_scoring_explanation(self, message: Message) -> Dict[str, Any]:
        """Get detailed explanation of how importance score was calculated."""
        pass


class ILLMService(ABC):
    """Interface for Large Language Model interactions."""
    
    @abstractmethod
    async def summarize_messages(
        self, 
        messages: List[Message], 
        context: Optional[str] = None
    ) -> str:
        """Summarize a collection of messages."""
        pass
    
    @abstractmethod
    async def analyze_sentiment(self, message: Message) -> Dict[str, float]:
        """Analyze sentiment of a message."""
        pass
    
    @abstractmethod
    async def extract_key_topics(self, messages: List[Message]) -> List[str]:
        """Extract key topics from messages."""
        pass
    
    @abstractmethod
    async def answer_query(
        self, 
        query: str, 
        context_messages: List[Message],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Answer a query based on message context."""
        pass


class INotificationService(ABC):
    """Interface for sending notifications."""
    
    @abstractmethod
    async def send_notification(
        self, 
        message: str, 
        importance_level: ImportanceLevel,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a notification."""
        pass
    
    @abstractmethod
    async def register_notification_channel(self, channel_name: str, config: Dict[str, Any]) -> bool:
        """Register a new notification channel."""
        pass
    
    @abstractmethod
    async def get_available_channels(self) -> List[str]:
        """Get list of available notification channels."""
        pass


class IMessageProcessingService(ABC):
    """Interface for message processing pipeline."""
    
    @abstractmethod
    async def process_message(self, message: Message) -> ProcessingResult:
        """Process a single message through the entire pipeline."""
        pass
    
    @abstractmethod
    async def process_message_batch(self, messages: List[Message]) -> List[ProcessingResult]:
        """Process multiple messages efficiently."""
        pass
    
    @abstractmethod
    async def reprocess_message(self, message_id: str) -> ProcessingResult:
        """Reprocess an existing message with updated rules."""
        pass


class IEventService(ABC):
    """Interface for event handling and triggering."""
    
    @abstractmethod
    async def trigger_event(self, event: MonitoringEvent) -> bool:
        """Trigger a monitoring event."""
        pass
    
    @abstractmethod
    async def register_event_handler(
        self, 
        event_type: EventType, 
        handler_name: str,
        handler_config: Dict[str, Any]
    ) -> bool:
        """Register an event handler."""
        pass
    
    @abstractmethod
    async def get_recent_events(
        self, 
        hours: int = 24, 
        event_types: Optional[List[EventType]] = None
    ) -> List[MonitoringEvent]:
        """Get recent events."""
        pass


class IConfigService(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    async def get_feature_flag(self, flag_name: str) -> bool:
        """Get feature flag status."""
        pass
    
    @abstractmethod
    async def set_feature_flag(self, flag_name: str, enabled: bool) -> bool:
        """Set feature flag status."""
        pass
    
    @abstractmethod
    async def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        pass
    
    @abstractmethod
    async def update_monitoring_config(self, config: Dict[str, Any]) -> bool:
        """Update monitoring configuration."""
        pass
    
    @abstractmethod
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences."""
        pass
    
    @abstractmethod
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        pass


class IHealthCheckService(ABC):
    """Interface for system health monitoring."""
    
    @abstractmethod
    async def check_discord_connection(self) -> bool:
        """Check if Discord connection is healthy."""
        pass
    
    @abstractmethod
    async def check_database_connection(self) -> bool:
        """Check if database connection is healthy."""
        pass
    
    @abstractmethod
    async def check_llm_service(self) -> bool:
        """Check if LLM service is healthy."""
        pass
    
    @abstractmethod
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        pass
    
    @abstractmethod
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        pass