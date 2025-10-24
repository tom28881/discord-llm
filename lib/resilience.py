"""
Resilience and recovery framework for Discord monitoring assistant.

Implements retry logic, circuit breakers, rate limiting, and fallback strategies
for reliable 24/7 operation.
"""

import time
import asyncio
import random
from typing import Callable, Any, Optional, Dict, List, Union
from functools import wraps
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import threading
import logging
from .exceptions import (
    DiscordRateLimitError, DiscordConnectionError, LLMRateLimitError,
    DatabaseLockError, QueueOverflowError, MemoryError
)

logger = logging.getLogger('discord_bot.resilience')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit open, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: List[type] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception


class ExponentialBackoff:
    """Implements exponential backoff with jitter."""
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, 
                 exponential_base: float = 2.0, jitter: bool = True):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker for {func.__name__} moving to HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return True
        return (datetime.utcnow() - self.last_failure_time).seconds >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            logger.debug("Circuit breaker reset to CLOSED")
    
    def _on_failure(self):
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, burst: int = None):
        self.rate = rate  # tokens per second
        self.burst = burst or int(rate)  # bucket size
        self.tokens = self.burst
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, tokens: int = 1, timeout: float = None) -> bool:
        """Acquire tokens from the bucket."""
        start_time = time.time()
        
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                
                # Add tokens based on elapsed time
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
            
            # Check timeout
            if timeout is not None and (time.time() - start_time) >= timeout:
                return False
            
            # Wait a bit before retrying
            time.sleep(0.1)
    
    def wait_time(self, tokens: int = 1) -> float:
        """Calculate time to wait for tokens to be available."""
        with self.lock:
            if self.tokens >= tokens:
                return 0.0
            needed_tokens = tokens - self.tokens
            return needed_tokens / self.rate


def retry_with_backoff(config: RetryConfig = None):
    """Decorator for retry logic with exponential backoff."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            backoff = ExponentialBackoff(
                base_delay=config.base_delay,
                max_delay=config.max_delay,
                exponential_base=config.exponential_base,
                jitter=config.jitter
            )
            
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry on this exception
                    if config.retry_on and not any(isinstance(e, exc_type) for exc_type in config.retry_on):
                        raise
                    
                    # Don't retry on the last attempt
                    if attempt == config.max_attempts - 1:
                        raise
                    
                    delay = backoff.delay(attempt)
                    logger.warning(f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}: {e}. "
                                 f"Retrying in {delay:.2f} seconds...")
                    
                    # Handle specific rate limit delays
                    if isinstance(e, (DiscordRateLimitError, LLMRateLimitError)):
                        if hasattr(e, 'retry_after') and e.retry_after:
                            delay = max(delay, e.retry_after)
                    
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


class FallbackHandler:
    """Handles fallback strategies when primary methods fail."""
    
    def __init__(self):
        self.fallback_methods = {}
    
    def register_fallback(self, primary_method: str, fallback_method: Callable):
        """Register a fallback method for a primary method."""
        if primary_method not in self.fallback_methods:
            self.fallback_methods[primary_method] = []
        self.fallback_methods[primary_method].append(fallback_method)
    
    def execute_with_fallback(self, primary_method: str, primary_func: Callable, 
                            *args, **kwargs) -> Any:
        """Execute primary function with fallback options."""
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary method {primary_method} failed: {e}")
            
            fallbacks = self.fallback_methods.get(primary_method, [])
            for i, fallback_func in enumerate(fallbacks):
                try:
                    logger.info(f"Attempting fallback {i + 1}/{len(fallbacks)} for {primary_method}")
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.warning(f"Fallback {i + 1} failed for {primary_method}: {fallback_error}")
            
            # If all fallbacks failed, raise the original exception
            raise e


class HealthChecker:
    """Monitors system health and triggers recovery actions."""
    
    def __init__(self):
        self.health_checks = {}
        self.recovery_actions = {}
        self.check_interval = 30  # seconds
        self.running = False
        self.thread = None
    
    def register_health_check(self, name: str, check_func: Callable[[], bool], 
                            recovery_action: Callable = None):
        """Register a health check with optional recovery action."""
        self.health_checks[name] = check_func
        if recovery_action:
            self.recovery_actions[name] = recovery_action
    
    def start_monitoring(self):
        """Start health monitoring in background thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            for name, check_func in self.health_checks.items():
                try:
                    if not check_func():
                        logger.warning(f"Health check failed: {name}")
                        
                        # Attempt recovery if available
                        if name in self.recovery_actions:
                            try:
                                logger.info(f"Executing recovery action for: {name}")
                                self.recovery_actions[name]()
                            except Exception as e:
                                logger.error(f"Recovery action failed for {name}: {e}")
                except Exception as e:
                    logger.error(f"Health check error for {name}: {e}")
            
            time.sleep(self.check_interval)


# Global instances for easy access
fallback_handler = FallbackHandler()
health_checker = HealthChecker()

# Common rate limiters
discord_rate_limiter = RateLimiter(rate=50/60, burst=50)  # 50 requests per minute
llm_rate_limiter = RateLimiter(rate=60/60, burst=10)      # 60 requests per minute

# Common circuit breakers
discord_circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=120,
    expected_exception=DiscordConnectionError
))

llm_circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=60,
    expected_exception=(LLMRateLimitError, Exception)
))

database_circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=30,
    expected_exception=DatabaseLockError
))


def resilient_discord_call(func):
    """Decorator combining retry, rate limiting, and circuit breaking for Discord calls."""
    @retry_with_backoff(RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=300.0,
        retry_on=[DiscordRateLimitError, DiscordConnectionError]
    ))
    @discord_circuit_breaker
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Rate limiting
        if not discord_rate_limiter.acquire(timeout=30):
            raise DiscordRateLimitError("Rate limiter timeout")
        
        return func(*args, **kwargs)
    
    return wrapper


def resilient_llm_call(func):
    """Decorator combining retry, rate limiting, and circuit breaking for LLM calls."""
    @retry_with_backoff(RetryConfig(
        max_attempts=2,
        base_delay=2.0,
        max_delay=60.0,
        retry_on=[LLMRateLimitError]
    ))
    @llm_circuit_breaker
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Rate limiting
        if not llm_rate_limiter.acquire(timeout=60):
            raise LLMRateLimitError("Rate limiter timeout")
        
        return func(*args, **kwargs)
    
    return wrapper


def resilient_database_call(func):
    """Decorator combining retry and circuit breaking for database calls."""
    @retry_with_backoff(RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        max_delay=5.0,
        retry_on=[DatabaseLockError]
    ))
    @database_circuit_breaker
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return wrapper