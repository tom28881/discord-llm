"""
Custom exception classes for Discord monitoring assistant.

Provides specific error types for different failure scenarios to enable
targeted error handling and recovery strategies.
"""

import time
from typing import Optional, Dict, Any
from datetime import datetime


class DiscordMonitorException(Exception):
    """Base exception for Discord monitoring assistant."""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and monitoring."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }


class DiscordAPIError(DiscordMonitorException):
    """Discord API related errors."""
    
    def __init__(self, message: str, status_code: int = None, response_data: Dict = None, 
                 retry_after: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response_data = response_data or {}
        self.retry_after = retry_after
        self.context.update({
            'status_code': status_code,
            'response_data': response_data,
            'retry_after': retry_after
        })


class DiscordRateLimitError(DiscordAPIError):
    """Discord API rate limit exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None, **kwargs):
        super().__init__(message, status_code=429, retry_after=retry_after, **kwargs)
        self.recommended_wait = retry_after or 60  # Default 60 seconds


class DiscordConnectionError(DiscordAPIError):
    """Discord connection related errors."""
    
    def __init__(self, message: str, original_error: Exception = None, **kwargs):
        super().__init__(message, **kwargs)
        self.original_error = original_error
        self.context['original_error'] = str(original_error) if original_error else None


class DiscordAuthError(DiscordAPIError):
    """Discord authentication/authorization errors."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, status_code=401, **kwargs)


class DiscordForbiddenError(DiscordAPIError):
    """Discord access forbidden errors."""
    
    def __init__(self, message: str, channel_id: str = None, server_id: str = None, **kwargs):
        super().__init__(message, status_code=403, **kwargs)
        self.channel_id = channel_id
        self.server_id = server_id
        self.context.update({
            'channel_id': channel_id,
            'server_id': server_id
        })


class LLMError(DiscordMonitorException):
    """LLM integration related errors."""
    
    def __init__(self, message: str, provider: str = None, model: str = None, 
                 token_count: int = None, cost: float = None, **kwargs):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.model = model
        self.token_count = token_count
        self.cost = cost
        self.context.update({
            'provider': provider,
            'model': model,
            'token_count': token_count,
            'cost': cost
        })


class LLMRateLimitError(LLMError):
    """LLM API rate limit exceeded."""
    
    def __init__(self, message: str = "LLM rate limit exceeded", retry_after: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.context['retry_after'] = retry_after


class LLMTokenLimitError(LLMError):
    """LLM token limit exceeded."""
    
    def __init__(self, message: str, requested_tokens: int = None, max_tokens: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.requested_tokens = requested_tokens
        self.max_tokens = max_tokens
        self.context.update({
            'requested_tokens': requested_tokens,
            'max_tokens': max_tokens
        })


class LLMCostLimitError(LLMError):
    """LLM cost limit exceeded."""
    
    def __init__(self, message: str, current_cost: float = None, max_cost: float = None, **kwargs):
        super().__init__(message, **kwargs)
        self.current_cost = current_cost
        self.max_cost = max_cost
        self.context.update({
            'current_cost': current_cost,
            'max_cost': max_cost
        })


class DatabaseError(DiscordMonitorException):
    """Database related errors."""
    
    def __init__(self, message: str, operation: str = None, table: str = None, 
                 original_error: Exception = None, **kwargs):
        super().__init__(message, **kwargs)
        self.operation = operation
        self.table = table
        self.original_error = original_error
        self.context.update({
            'operation': operation,
            'table': table,
            'original_error': str(original_error) if original_error else None
        })


class DatabaseConnectionError(DatabaseError):
    """Database connection errors."""
    
    def __init__(self, message: str = "Database connection failed", **kwargs):
        super().__init__(message, **kwargs)


class DatabaseLockError(DatabaseError):
    """Database lock/timeout errors."""
    
    def __init__(self, message: str = "Database locked", timeout: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.timeout = timeout
        self.context['timeout'] = timeout


class DatabaseIntegrityError(DatabaseError):
    """Database integrity/constraint errors."""
    
    def __init__(self, message: str, constraint: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.constraint = constraint
        self.context['constraint'] = constraint


class ProcessingError(DiscordMonitorException):
    """Real-time processing errors."""
    
    def __init__(self, message: str, component: str = None, queue_size: int = None, 
                 memory_usage: float = None, **kwargs):
        super().__init__(message, **kwargs)
        self.component = component
        self.queue_size = queue_size
        self.memory_usage = memory_usage
        self.context.update({
            'component': component,
            'queue_size': queue_size,
            'memory_usage': memory_usage
        })


class QueueOverflowError(ProcessingError):
    """Message queue overflow error."""
    
    def __init__(self, message: str = "Queue overflow detected", **kwargs):
        super().__init__(message, **kwargs)


class MemoryError(ProcessingError):
    """Memory limit exceeded error."""
    
    def __init__(self, message: str, current_usage: float = None, max_usage: float = None, **kwargs):
        super().__init__(message, **kwargs)
        self.current_usage = current_usage
        self.max_usage = max_usage
        self.context.update({
            'current_usage': current_usage,
            'max_usage': max_usage
        })


class DeadlockError(ProcessingError):
    """Deadlock detection error."""
    
    def __init__(self, message: str = "Deadlock detected", involved_components: list = None, **kwargs):
        super().__init__(message, **kwargs)
        self.involved_components = involved_components or []
        self.context['involved_components'] = self.involved_components


class ConfigurationError(DiscordMonitorException):
    """Configuration related errors."""
    
    def __init__(self, message: str, config_key: str = None, config_file: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.config_file = config_file
        self.context.update({
            'config_key': config_key,
            'config_file': config_file
        })


# Exception mapping for HTTP status codes
HTTP_STATUS_TO_EXCEPTION = {
    401: DiscordAuthError,
    403: DiscordForbiddenError,
    429: DiscordRateLimitError,
    500: DiscordConnectionError,
    502: DiscordConnectionError,
    503: DiscordConnectionError,
    504: DiscordConnectionError,
}


def create_discord_api_exception(status_code: int, message: str, **kwargs) -> DiscordAPIError:
    """Create appropriate Discord API exception based on status code."""
    exception_class = HTTP_STATUS_TO_EXCEPTION.get(status_code, DiscordAPIError)
    return exception_class(message, status_code=status_code, **kwargs)