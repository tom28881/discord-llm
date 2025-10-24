"""
Enhanced LLM client with comprehensive error handling and fallback strategies.

Supports multiple providers (OpenAI, Google Gemini, OpenRouter) with automatic
fallback, cost tracking, token management, and robust error handling.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import google.generativeai as genai
from dotenv import load_dotenv

from .exceptions import (
    LLMError, LLMRateLimitError, LLMTokenLimitError, LLMCostLimitError,
    ConfigurationError
)
from .resilience import resilient_llm_call, FallbackHandler

load_dotenv()
logger = logging.getLogger('discord_bot.llm')


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    provider: str
    model: str
    max_tokens: int
    cost_per_1k_tokens: float
    rate_limit_rpm: int  # requests per minute
    supports_streaming: bool = False
    context_window: int = 4096


@dataclass
class UsageStats:
    """Track usage statistics for cost and rate limiting."""
    total_tokens: int = 0
    total_cost: float = 0.0
    requests_today: int = 0
    last_request: Optional[datetime] = None
    daily_reset: datetime = None
    
    def __post_init__(self):
        if self.daily_reset is None:
            self.daily_reset = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        self.usage_stats = UsageStats()
    
    @abstractmethod
    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion for given prompt."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is currently available."""
        pass
    
    def update_usage(self, tokens: int, cost: float):
        """Update usage statistics."""
        now = datetime.now()
        
        # Reset daily stats if needed
        if now.date() > self.usage_stats.daily_reset.date():
            self.usage_stats.requests_today = 0
            self.usage_stats.daily_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        self.usage_stats.total_tokens += tokens
        self.usage_stats.total_cost += cost
        self.usage_stats.requests_today += 1
        self.usage_stats.last_request = now
    
    def check_rate_limits(self) -> bool:
        """Check if within rate limits."""
        if not self.usage_stats.last_request:
            return True
        
        time_since_last = datetime.now() - self.usage_stats.last_request
        if time_since_last < timedelta(minutes=1):
            # Rough rate limiting check
            if self.usage_stats.requests_today >= self.config.rate_limit_rpm:
                return False
        
        return True


class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config, api_key)
        genai.configure(api_key=api_key)
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Gemini model."""
        try:
            self.model = genai.GenerativeModel(model_name=self.config.model)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise ConfigurationError(f"Gemini model initialization failed: {e}")
    
    @resilient_llm_call
    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion using Gemini."""
        if not self.is_available():
            raise LLMError("Gemini provider not available")
        
        if not self.check_rate_limits():
            raise LLMRateLimitError("Gemini rate limit exceeded")
        
        try:
            # Estimate tokens (rough approximation)
            estimated_tokens = self.count_tokens(prompt)
            
            # Check token limits
            if estimated_tokens > self.config.max_tokens:
                raise LLMTokenLimitError(
                    f"Prompt too long: {estimated_tokens} > {self.config.max_tokens}",
                    requested_tokens=estimated_tokens,
                    max_tokens=self.config.max_tokens
                )
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                raise LLMError("Empty response from Gemini")
            
            # Calculate actual usage
            response_tokens = self.count_tokens(response.text)
            total_tokens = estimated_tokens + response_tokens
            cost = (total_tokens / 1000) * self.config.cost_per_1k_tokens
            
            # Update usage stats
            self.update_usage(total_tokens, cost)
            
            return response.text
            
        except Exception as e:
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                raise LLMRateLimitError(f"Gemini rate limit: {e}")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise LLMTokenLimitError(f"Gemini token limit: {e}")
            else:
                raise LLMError(f"Gemini generation failed: {e}")
    
    def count_tokens(self, text: str) -> int:
        """Rough token counting for Gemini (approximation)."""
        # Gemini doesn't have a direct token counting API in the client
        # This is a rough approximation: ~4 characters per token
        return len(text) // 4
    
    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return self.model is not None and self.api_key is not None


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation (placeholder for future implementation)."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config, api_key)
        # Would implement OpenAI client here
    
    def generate_completion(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("OpenAI provider not implemented yet")
    
    def count_tokens(self, text: str) -> int:
        # Would use tiktoken or similar for accurate counting
        return len(text) // 4
    
    def is_available(self) -> bool:
        return False  # Not implemented yet


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider implementation (placeholder for future implementation)."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config, api_key)
        # Would implement OpenRouter client here
    
    def generate_completion(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("OpenRouter provider not implemented yet")
    
    def count_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def is_available(self) -> bool:
        return False  # Not implemented yet


class EnhancedLLMClient:
    """Enhanced LLM client with multiple providers and fallback strategies."""
    
    def __init__(self, cost_limit_daily: float = 10.0):
        self.providers = {}
        self.fallback_order = []
        self.cost_limit_daily = cost_limit_daily
        self.fallback_handler = FallbackHandler()
        
        # Load configuration
        self._load_model_configs()
        self._initialize_providers()
        self._setup_fallbacks()
    
    def _load_model_configs(self):
        """Load model configurations."""
        self.model_configs = {
            'gemini-2.5-flash': ModelConfig(
                provider='gemini',
                model='gemini-2.5-flash',
                max_tokens=8192,
                cost_per_1k_tokens=0.0005,  # Cheapest - primary choice
                rate_limit_rpm=120
            ),
            'gemini-2.5-pro': ModelConfig(
                provider='gemini',
                model='gemini-2.5-pro',
                max_tokens=8192,
                cost_per_1k_tokens=0.003,  # More expensive
                rate_limit_rpm=60
            ),
            'gemini-1.5-flash': ModelConfig(
                provider='gemini',
                model='gemini-1.5-flash',
                max_tokens=4096,
                cost_per_1k_tokens=0.001,  # Fallback
                rate_limit_rpm=100
            ),
            # Would add more models here
        }
    
    def _initialize_providers(self):
        """Initialize available providers."""
        # Initialize Gemini
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            try:
                for model_name, config in self.model_configs.items():
                    if config.provider == 'gemini':
                        provider = GeminiProvider(config, google_api_key)
                        if provider.is_available():
                            self.providers[model_name] = provider
                            logger.info(f"Initialized {model_name} provider")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini provider: {e}")
        
        # Initialize OpenAI (when implemented)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            # Would initialize OpenAI providers here
            pass
        
        # Initialize OpenRouter (when implemented)
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_api_key:
            # Would initialize OpenRouter providers here
            pass
        
        if not self.providers:
            raise ConfigurationError("No LLM providers available")
    
    def _setup_fallbacks(self):
        """Setup fallback strategies."""
        # Order providers by preference (cost, reliability, speed)
        primary_models = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-2.5-pro']  # Cheapest first
        
        self.fallback_order = [model for model in primary_models if model in self.providers]
        
        if not self.fallback_order:
            raise ConfigurationError("No available models for fallback")
        
        logger.info(f"Fallback order: {self.fallback_order}")
    
    def _check_daily_cost_limit(self) -> bool:
        """Check if daily cost limit would be exceeded."""
        total_daily_cost = sum(p.usage_stats.total_cost for p in self.providers.values())
        return total_daily_cost < self.cost_limit_daily
    
    def _get_best_available_provider(self, prompt: str) -> Tuple[str, LLMProvider]:
        """Get the best available provider for the given prompt."""
        prompt_tokens = len(prompt) // 4  # Rough estimate
        
        for model_name in self.fallback_order:
            provider = self.providers.get(model_name)
            if not provider:
                continue
            
            # Check availability
            if not provider.is_available():
                continue
            
            # Check rate limits
            if not provider.check_rate_limits():
                continue
            
            # Check token limits
            if prompt_tokens > provider.config.max_tokens:
                continue
            
            # Check cost limits
            if not self._check_daily_cost_limit():
                logger.warning("Daily cost limit reached, using cheapest available model")
                # Find cheapest available model
                cheapest = min(
                    [(name, p) for name, p in self.providers.items() if p.is_available()],
                    key=lambda x: x[1].config.cost_per_1k_tokens,
                    default=(None, None)
                )
                if cheapest[0]:
                    return cheapest
                else:
                    raise LLMCostLimitError("Daily cost limit exceeded and no cheap alternatives")
            
            return model_name, provider
        
        raise LLMError("No available providers")
    
    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion with automatic fallback."""
        if not prompt or not prompt.strip():
            raise LLMError("Empty prompt provided")
        
        last_exception = None
        
        for attempt in range(len(self.fallback_order)):
            try:
                model_name, provider = self._get_best_available_provider(prompt)
                logger.debug(f"Using {model_name} for completion")
                
                return provider.generate_completion(prompt, **kwargs)
                
            except LLMRateLimitError as e:
                logger.warning(f"Rate limit hit for {model_name}: {e}")
                last_exception = e
                # Remove this provider temporarily and try next
                continue
                
            except LLMTokenLimitError as e:
                logger.warning(f"Token limit exceeded for {model_name}: {e}")
                # Try to truncate prompt if possible
                if len(prompt) > 1000:
                    truncated_prompt = prompt[:1000] + "...[truncated]"
                    logger.info("Truncating prompt and retrying")
                    try:
                        model_name, provider = self._get_best_available_provider(truncated_prompt)
                        return provider.generate_completion(truncated_prompt, **kwargs)
                    except Exception:
                        pass
                
                last_exception = e
                continue
                
            except LLMCostLimitError as e:
                logger.error(f"Cost limit exceeded: {e}")
                last_exception = e
                break  # Don't continue if cost limit reached
                
            except LLMError as e:
                logger.warning(f"LLM error with {model_name}: {e}")
                last_exception = e
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error with {model_name}: {e}")
                last_exception = LLMError(f"Unexpected error: {e}")
                continue
        
        # All providers failed
        if last_exception:
            raise last_exception
        else:
            raise LLMError("All LLM providers failed")
    
    def get_simple_completion(self, prompt: str) -> str:
        """Simple wrapper that returns empty string on failure for non-critical use."""
        try:
            return self.generate_completion(prompt)
        except Exception as e:
            logger.warning(f"LLM completion failed, returning empty string: {e}")
            return ""
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all providers."""
        stats = {
            'total_cost_today': 0.0,
            'total_tokens_today': 0,
            'total_requests_today': 0,
            'cost_limit_daily': self.cost_limit_daily,
            'providers': {}
        }
        
        for name, provider in self.providers.items():
            provider_stats = asdict(provider.usage_stats)
            stats['providers'][name] = provider_stats
            
            stats['total_cost_today'] += provider_stats['total_cost']
            stats['total_tokens_today'] += provider_stats['total_tokens']
            stats['total_requests_today'] += provider_stats['requests_today']
        
        stats['cost_limit_remaining'] = max(0, self.cost_limit_daily - stats['total_cost_today'])
        stats['cost_limit_exceeded'] = stats['total_cost_today'] >= self.cost_limit_daily
        
        return stats
    
    def reset_daily_stats(self):
        """Reset daily usage statistics for all providers."""
        for provider in self.providers.values():
            provider.usage_stats.requests_today = 0
            provider.usage_stats.daily_reset = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        logger.info("Daily LLM usage statistics reset")
    
    def add_fallback_strategy(self, strategy_name: str, strategy_func: Callable):
        """Add custom fallback strategy."""
        self.fallback_handler.register_fallback(strategy_name, strategy_func)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all providers."""
        health = {
            'healthy_providers': [],
            'unhealthy_providers': [],
            'total_providers': len(self.providers),
            'fallback_order': self.fallback_order
        }
        
        for name, provider in self.providers.items():
            try:
                if provider.is_available() and provider.check_rate_limits():
                    health['healthy_providers'].append(name)
                else:
                    health['unhealthy_providers'].append(name)
            except Exception as e:
                health['unhealthy_providers'].append(f"{name} (error: {e})")
        
        health['overall_healthy'] = len(health['healthy_providers']) > 0
        
        return health


# Global instance for backward compatibility and easy access
enhanced_llm_client = None

def get_enhanced_llm_client(cost_limit_daily: float = 10.0) -> EnhancedLLMClient:
    """Get or create the global enhanced LLM client."""
    global enhanced_llm_client
    if enhanced_llm_client is None:
        enhanced_llm_client = EnhancedLLMClient(cost_limit_daily)
    return enhanced_llm_client


def get_completion(prompt: str, model: str = None, **kwargs) -> str:
    """Backward compatible function that uses enhanced client."""
    client = get_enhanced_llm_client()
    return client.generate_completion(prompt, **kwargs)


def get_simple_completion(prompt: str) -> str:
    """Get completion with graceful fallback to empty string."""
    client = get_enhanced_llm_client()
    return client.get_simple_completion(prompt)