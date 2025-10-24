"""
Comprehensive logging setup with structured logging, error handling, and performance tracking.
This module provides enterprise-grade logging capabilities for the Discord Monitor.
"""
import logging
import logging.handlers
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import asyncio
import contextlib
import functools
import time


@dataclass
class LogContext:
    """Context information for structured logging."""
    component: str
    operation: str
    user_id: Optional[str] = None
    message_id: Optional[str] = None
    server_id: Optional[str] = None
    channel_id: Optional[str] = None
    correlation_id: Optional[str] = None
    extra_data: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        try:
            log_data = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Add exception information if present
            if record.exc_info:
                log_data["exception"] = {
                    "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                    "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                    "traceback": self.formatException(record.exc_info)
                }
            
            # Add extra context if present
            if hasattr(record, 'context') and record.context:
                if isinstance(record.context, LogContext):
                    log_data["context"] = record.context.to_dict()
                else:
                    log_data["context"] = record.context
            
            # Add performance data if present
            if hasattr(record, 'performance'):
                log_data["performance"] = record.performance
            
            # Add custom fields from extra
            for key, value in record.__dict__.items():
                if (key not in log_data and 
                    not key.startswith('_') and 
                    key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                               'pathname', 'filename', 'module', 'lineno', 'funcName',
                               'created', 'msecs', 'relativeCreated', 'thread',
                               'threadName', 'processName', 'process', 'getMessage',
                               'exc_info', 'exc_text', 'stack_info', 'context',
                               'performance']):
                    log_data[key] = value
            
            return json.dumps(log_data, default=str, ensure_ascii=False)
            
        except Exception as e:
            # Fallback to basic formatting if JSON serialization fails
            return f"LOGGING_ERROR: {str(e)} | Original: {record.getMessage()}"


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability during development."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[41m', # Red background
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console output."""
        try:
            color = self.COLORS.get(record.levelname, '')
            reset = self.RESET
            
            # Format timestamp
            timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
            
            # Format basic log line
            log_line = f"{color}[{timestamp}] {record.levelname:8} {record.name:20} | {record.getMessage()}{reset}"
            
            # Add context information if present
            if hasattr(record, 'context') and record.context:
                if isinstance(record.context, LogContext):
                    context_str = " | ".join([f"{k}={v}" for k, v in record.context.to_dict().items() if v])
                    log_line += f"\n    Context: {context_str}"
            
            # Add performance information if present
            if hasattr(record, 'performance'):
                perf_str = " | ".join([f"{k}={v}" for k, v in record.performance.items()])
                log_line += f"\n    Performance: {perf_str}"
            
            # Add exception information
            if record.exc_info:
                log_line += f"\n{self.formatException(record.exc_info)}"
            
            return log_line
            
        except Exception as e:
            return f"FORMATTING_ERROR: {str(e)} | Original: {record.getMessage()}"


class PerformanceLogger:
    """Context manager and decorator for performance logging."""
    
    def __init__(self, logger: logging.Logger, operation: str, context: Optional[LogContext] = None):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.debug(f"Starting {self.operation}", extra={"context": self.context})
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration_ms = (self.end_time - self.start_time) * 1000
        
        performance_data = {
            "operation": self.operation,
            "duration_ms": round(duration_ms, 2),
            "success": exc_type is None
        }
        
        if exc_type:
            performance_data["error_type"] = exc_type.__name__
            performance_data["error_message"] = str(exc_val)
        
        log_level = logging.DEBUG if exc_type is None else logging.ERROR
        self.logger.log(
            log_level,
            f"Completed {self.operation} in {duration_ms:.2f}ms",
            extra={
                "context": self.context,
                "performance": performance_data
            },
            exc_info=(exc_type, exc_val, exc_tb) if exc_type else None
        )
    
    def __call__(self, func):
        """Use as decorator."""
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                context = self.context or LogContext(
                    component=func.__module__,
                    operation=func.__name__
                )
                
                with PerformanceLogger(self.logger, f"{func.__name__}()", context):
                    return await func(*args, **kwargs)
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                context = self.context or LogContext(
                    component=func.__module__,
                    operation=func.__name__
                )
                
                with PerformanceLogger(self.logger, f"{func.__name__}()", context):
                    return func(*args, **kwargs)
            
            return sync_wrapper


class DiscordMonitorLogger:
    """Enhanced logger with Discord Monitor specific functionality."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._context_stack = []
    
    @contextlib.contextmanager
    def context(self, **context_kwargs):
        """Context manager for adding structured context to logs."""
        if isinstance(context_kwargs.get('context'), LogContext):
            context = context_kwargs['context']
        else:
            context = LogContext(**context_kwargs)
        
        self._context_stack.append(context)
        try:
            yield context
        finally:
            if self._context_stack:
                self._context_stack.pop()
    
    def _log_with_context(self, level: int, message: str, *args, **kwargs):
        """Log with current context."""
        extra = kwargs.get('extra', {})
        
        # Add current context if available
        if self._context_stack:
            extra['context'] = self._context_stack[-1]
        
        # Allow override of context
        if 'context' in kwargs:
            extra['context'] = kwargs.pop('context')
        
        kwargs['extra'] = extra
        self.logger.log(level, message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message with context."""
        self._log_with_context(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message with context."""
        self._log_with_context(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log exception with context."""
        kwargs['exc_info'] = True
        self.error(message, *args, **kwargs)
    
    def performance(self, operation: str, context: Optional[LogContext] = None):
        """Get performance logger for operation."""
        return PerformanceLogger(self.logger, operation, context)
    
    def message_processing(self, message_id: str, server_id: str, channel_id: str):
        """Context manager for message processing logs."""
        return self.context(
            component="message_processor",
            operation="process_message",
            message_id=message_id,
            server_id=server_id,
            channel_id=channel_id
        )
    
    def discord_api(self, operation: str, server_id: Optional[str] = None, channel_id: Optional[str] = None):
        """Context manager for Discord API logs."""
        return self.context(
            component="discord_service",
            operation=operation,
            server_id=server_id,
            channel_id=channel_id
        )
    
    def notification(self, channel: str, importance_level: str):
        """Context manager for notification logs."""
        return self.context(
            component="notification_service",
            operation="send_notification",
            extra_data={"channel": channel, "importance": importance_level}
        )


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    structured_logging: bool = True,
    console_colors: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Setup comprehensive logging for the application.
    
    Args:
        level: Logging level
        log_file: Path to log file (if None, only console logging)
        structured_logging: Use structured JSON logging for files
        console_colors: Use colored console output
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if console_colors:
        console_formatter = ColoredConsoleFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        
        if structured_logging:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific log levels for noisy libraries
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('discord').setLevel(logging.WARNING)
    logging.getLogger('sqlite').setLevel(logging.WARNING)
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {logging.getLevelName(level)}, File: {log_file}")


def get_logger(name: str) -> DiscordMonitorLogger:
    """
    Get enhanced logger instance for Discord Monitor.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        DiscordMonitorLogger: Enhanced logger instance
    """
    return DiscordMonitorLogger(name)


# Performance decorators for common use cases
def log_performance(logger: Optional[logging.Logger] = None, operation: Optional[str] = None):
    """Decorator for automatic performance logging."""
    def decorator(func):
        nonlocal logger, operation
        
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        if operation is None:
            operation = func.__name__
        
        perf_logger = PerformanceLogger(logger, operation)
        return perf_logger(func)
    
    return decorator


def log_exceptions(logger: Optional[logging.Logger] = None, reraise: bool = True):
    """Decorator for automatic exception logging."""
    def decorator(func):
        nonlocal logger
        
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.exception(f"Exception in {func.__name__}: {str(e)}")
                    if reraise:
                        raise
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.exception(f"Exception in {func.__name__}: {str(e)}")
                    if reraise:
                        raise
            
            return sync_wrapper
    
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Test logging setup
    setup_logging(level="DEBUG", log_file="logs/test.log")
    
    logger = get_logger(__name__)
    
    # Test basic logging
    logger.info("Testing basic logging")
    
    # Test contextual logging
    with logger.context(component="test", operation="demo", user_id="12345"):
        logger.info("Testing contextual logging")
        
        with logger.performance("test_operation"):
            import time
            time.sleep(0.1)  # Simulate work
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception:
        logger.exception("Testing exception logging")
    
    print("Logging test completed - check logs/test.log for structured output")