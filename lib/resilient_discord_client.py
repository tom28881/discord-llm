"""
Enhanced Discord client with comprehensive error handling and resilience.

Implements robust error handling, automatic recovery, rate limiting,
and connection management for 24/7 reliability.
"""

import logging
import requests
import time
import json
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import (
    DiscordAPIError, DiscordRateLimitError, DiscordConnectionError,
    DiscordAuthError, DiscordForbiddenError, create_discord_api_exception
)
from .resilience import resilient_discord_call, RateLimiter

logger = logging.getLogger('discord_bot.client')


class EnhancedDiscordClient:
    """Enhanced Discord client with comprehensive error handling."""
    
    BASE_URL = 'https://discord.com/api/v9'
    ALLOWED_CHANNEL_TYPES = {0, 5, 13}  # Text, News, Stage channels
    
    def __init__(self, token: str, server_id: str = None, max_retries: int = 3):
        if not token or len(token) <= 10:
            raise DiscordAuthError("Valid Discord token must be provided.")
            
        self.server_id = server_id
        self.token = token
        self.max_retries = max_retries
        
        # Setup session with retry strategy
        self.session = self._create_session()
        
        # Rate limiting - Discord allows 50 requests per second
        self.global_rate_limiter = RateLimiter(rate=45, burst=50)  # Slightly under limit
        self.route_rate_limiters = {}  # Per-route rate limiters
        
        # Connection health tracking
        self.last_successful_call = datetime.utcnow()
        self.consecutive_failures = 0
        self.connection_healthy = True
        
        # Request tracking for debugging
        self.request_history = []
        self.max_history_size = 100
        
    def _create_session(self) -> requests.Session:
        """Create session with retry strategy and proper headers."""
        session = requests.Session()
        
        # Retry strategy for connection errors
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["GET", "POST"],
            backoff_factor=1,
            raise_on_status=False  # We handle status codes manually
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            'Authorization': self.token,
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'User-Agent': (
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                'AppleWebKit/605.1.15 (KHTML, like Gecko) '
                'Version/18.0.1 Safari/605.1.15'
            ),
            'X-Discord-Client': 'Discord-Monitor/1.0'
        })
        
        return session
    
    def _record_request(self, method: str, url: str, status_code: int, 
                       response_time: float, error: str = None):
        """Record request for debugging and monitoring."""
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'method': method,
            'url': url,
            'status_code': status_code,
            'response_time': response_time,
            'error': error
        }
        
        self.request_history.append(record)
        
        # Limit history size
        if len(self.request_history) > self.max_history_size:
            self.request_history = self.request_history[-self.max_history_size:]
    
    def _get_route_rate_limiter(self, route: str) -> RateLimiter:
        """Get or create rate limiter for specific route."""
        if route not in self.route_rate_limiters:
            # Different routes have different limits
            if 'messages' in route:
                # Message endpoints: 5 requests per 5 seconds
                self.route_rate_limiters[route] = RateLimiter(rate=1, burst=5)
            else:
                # General endpoints: 50 requests per second
                self.route_rate_limiters[route] = RateLimiter(rate=50, burst=50)
        
        return self.route_rate_limiters[route]
    
    def _handle_rate_limit_headers(self, response: requests.Response, route: str):
        """Process rate limit headers and update rate limiters."""
        try:
            remaining = response.headers.get('X-RateLimit-Remaining')
            reset_after = response.headers.get('X-RateLimit-Reset-After')
            
            if remaining is not None and int(remaining) == 0 and reset_after:
                reset_time = float(reset_after)
                logger.warning(f"Rate limit reached for {route}. Reset in {reset_time} seconds.")
                
                # Update route-specific rate limiter
                route_limiter = self._get_route_rate_limiter(route)
                route_limiter.tokens = 0  # Force wait
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing rate limit headers: {e}")
    
    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with comprehensive error handling."""
        start_time = time.time()
        route = url.replace(self.BASE_URL, '').split('?')[0]  # Extract route for rate limiting
        
        try:
            # Global rate limiting
            if not self.global_rate_limiter.acquire(timeout=30):
                raise DiscordRateLimitError("Global rate limit timeout")
            
            # Route-specific rate limiting
            route_limiter = self._get_route_rate_limiter(route)
            if not route_limiter.acquire(timeout=60):
                raise DiscordRateLimitError(f"Route rate limit timeout for {route}")
            
            # Make the request
            response = self.session.request(method, url, **kwargs)
            response_time = time.time() - start_time
            
            # Record successful request
            self._record_request(method, url, response.status_code, response_time)
            
            # Handle rate limit headers
            self._handle_rate_limit_headers(response, route)
            
            # Check for errors
            self._check_response_errors(response)
            
            # Update health status
            self.last_successful_call = datetime.utcnow()
            self.consecutive_failures = 0
            self.connection_healthy = True
            
            return response
            
        except requests.RequestException as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            
            # Record failed request
            self._record_request(method, url, 0, response_time, error_msg)
            
            # Update health status
            self.consecutive_failures += 1
            if self.consecutive_failures >= 5:
                self.connection_healthy = False
            
            # Convert to appropriate exception
            if isinstance(e, requests.exceptions.Timeout):
                raise DiscordConnectionError(f"Request timeout: {error_msg}", original_error=e)
            elif isinstance(e, requests.exceptions.ConnectionError):
                raise DiscordConnectionError(f"Connection error: {error_msg}", original_error=e)
            else:
                raise DiscordConnectionError(f"Request failed: {error_msg}", original_error=e)
    
    def _check_response_errors(self, response: requests.Response):
        """Check response for errors and raise appropriate exceptions."""
        if response.status_code == 200:
            return
        
        try:
            error_data = response.json()
        except (ValueError, json.JSONDecodeError):
            error_data = {}
        
        error_message = error_data.get('message', f'HTTP {response.status_code}')
        
        # Handle specific status codes
        if response.status_code == 401:
            raise DiscordAuthError(f"Authentication failed: {error_message}")
        
        elif response.status_code == 403:
            raise DiscordForbiddenError(f"Access forbidden: {error_message}")
        
        elif response.status_code == 429:
            retry_after = error_data.get('retry_after', 60)
            global_limit = error_data.get('global', False)
            
            raise DiscordRateLimitError(
                f"Rate limit exceeded: {error_message}",
                retry_after=int(retry_after),
                context={'global': global_limit, 'error_data': error_data}
            )
        
        elif 500 <= response.status_code < 600:
            raise DiscordConnectionError(
                f"Server error ({response.status_code}): {error_message}",
                status_code=response.status_code,
                response_data=error_data
            )
        
        else:
            # Generic API error
            raise create_discord_api_exception(
                response.status_code,
                error_message,
                response_data=error_data
            )
    
    @resilient_discord_call
    def get_server_ids(self) -> List[Tuple[int, str]]:
        """Get list of accessible servers with enhanced error handling."""
        logger.info("Retrieving server IDs...")
        
        try:
            url = f"{self.BASE_URL}/users/@me/guilds"
            response = self._make_request('GET', url)
            servers = response.json()
            
            if not isinstance(servers, list):
                raise DiscordAPIError("Invalid server list response")
            
            server_info = []
            for server in servers:
                try:
                    server_id = int(server['id'])
                    server_name = server.get('name', f'Server-{server_id}')
                    server_info.append((server_id, server_name))
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid server data: {server}, error: {e}")
                    continue
            
            logger.info(f"Found {len(server_info)} servers")
            return server_info
            
        except DiscordAPIError:
            raise
        except Exception as e:
            raise DiscordAPIError(f"Failed to retrieve servers: {e}", context={'original_error': str(e)})
    
    @resilient_discord_call
    def get_channel_ids(self) -> List[Tuple[int, str]]:
        """Get list of accessible channels with enhanced error handling."""
        if not self.server_id:
            raise DiscordAPIError("Server ID not set")
        
        logger.info(f"Retrieving channels for server {self.server_id}...")
        
        try:
            url = f"{self.BASE_URL}/guilds/{self.server_id}/channels"
            response = self._make_request('GET', url)
            channels = response.json()
            
            if not isinstance(channels, list):
                raise DiscordAPIError("Invalid channel list response")
            
            channel_info = []
            for channel in channels:
                try:
                    channel_type = channel.get('type')
                    if channel_type not in self.ALLOWED_CHANNEL_TYPES:
                        continue
                    
                    channel_id = int(channel['id'])
                    channel_name = channel.get('name', f'Channel-{channel_id}')
                    
                    # Skip channels without read permissions
                    permissions = channel.get('permission_overwrites', [])
                    if self._is_channel_readable(permissions):
                        channel_info.append((channel_id, channel_name))
                        
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid channel data: {channel}, error: {e}")
                    continue
            
            logger.info(f"Found {len(channel_info)} accessible channels")
            return channel_info
            
        except DiscordForbiddenError as e:
            # Add server context to forbidden error
            e.server_id = self.server_id
            raise e
        except DiscordAPIError:
            raise
        except Exception as e:
            raise DiscordAPIError(
                f"Failed to retrieve channels for server {self.server_id}: {e}",
                context={'server_id': self.server_id, 'original_error': str(e)}
            )
    
    def _is_channel_readable(self, permission_overwrites: List[Dict]) -> bool:
        """Check if channel is readable based on permission overwrites."""
        # Simplified permission check - in practice you'd want more sophisticated logic
        # This is a basic implementation that assumes readability unless explicitly denied
        for overwrite in permission_overwrites:
            deny = overwrite.get('deny', 0)
            # Check for VIEW_CHANNEL (1024) and READ_MESSAGE_HISTORY (65536) permissions
            if (int(deny) & 1024) or (int(deny) & 65536):
                return False
        return True
    
    @resilient_discord_call
    def fetch_messages(self, channel_id: int, since_message_id: str = None, 
                      limit: int = 100) -> List[Tuple[str, str, int]]:
        """Fetch messages with enhanced error handling and validation."""
        logger.debug(f"Fetching up to {limit} messages from channel {channel_id}")
        
        try:
            all_messages = []
            params = {'limit': min(100, limit)}  # Discord API limit per request
            
            if since_message_id:
                params['after'] = since_message_id
            
            url = f"{self.BASE_URL}/channels/{channel_id}/messages"
            batch_number = 1
            
            while len(all_messages) < limit:
                try:
                    response = self._make_request('GET', url, params=params)
                    messages = response.json()
                    
                    if not isinstance(messages, list):
                        raise DiscordAPIError("Invalid messages response format")
                    
                    if not messages:
                        break
                    
                    # Process messages
                    for message in messages:
                        if len(all_messages) >= limit:
                            break
                        
                        try:
                            message_data = self._process_message(message)
                            if message_data:
                                all_messages.append(message_data)
                        except Exception as e:
                            logger.warning(f"Skipping invalid message: {e}")
                            continue
                    
                    logger.debug(f"Batch {batch_number}: Retrieved {len(all_messages)} total messages")
                    batch_number += 1
                    
                    if len(messages) < params['limit']:
                        break  # No more messages to retrieve
                    
                    # Update params for next batch
                    params['before'] = messages[-1]['id']
                    
                    # Small delay between batches to be respectful
                    time.sleep(0.1)
                    
                except DiscordRateLimitError as e:
                    # Handle rate limit with proper delay
                    wait_time = getattr(e, 'retry_after', 5)
                    logger.warning(f"Rate limited on channel {channel_id}, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                
            logger.info(f"Fetched {len(all_messages)} messages from channel {channel_id}")
            return all_messages
            
        except DiscordForbiddenError as e:
            # Add channel context to forbidden error
            e.channel_id = str(channel_id)
            raise e
        except DiscordAPIError:
            raise
        except Exception as e:
            raise DiscordAPIError(
                f"Failed to fetch messages from channel {channel_id}: {e}",
                context={'channel_id': channel_id, 'since_message_id': since_message_id, 'original_error': str(e)}
            )
    
    def _process_message(self, message: Dict[str, Any]) -> Optional[Tuple[str, str, int]]:
        """Process and validate a single message."""
        try:
            message_id = message['id']
            content = message.get('content', '')
            
            # Parse timestamp
            timestamp_str = message['timestamp']
            timestamp = int(datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).timestamp())
            
            # Basic validation
            if not message_id:
                return None
            
            return (message_id, content, timestamp)
            
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Invalid message format: {e}")
            return None
    
    def get_connection_health(self) -> Dict[str, Any]:
        """Get connection health information."""
        time_since_success = (datetime.utcnow() - self.last_successful_call).total_seconds()
        
        return {
            'healthy': self.connection_healthy,
            'consecutive_failures': self.consecutive_failures,
            'time_since_last_success': time_since_success,
            'last_successful_call': self.last_successful_call.isoformat(),
            'request_history_size': len(self.request_history)
        }
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get rate limiting status."""
        status = {
            'global_tokens': self.global_rate_limiter.tokens,
            'global_burst': self.global_rate_limiter.burst,
            'route_limiters': {}
        }
        
        for route, limiter in self.route_rate_limiters.items():
            status['route_limiters'][route] = {
                'tokens': limiter.tokens,
                'burst': limiter.burst,
                'wait_time': limiter.wait_time()
            }
        
        return status
    
    def reset_connection(self):
        """Reset connection and clear error state."""
        logger.info("Resetting Discord connection...")
        
        self.session.close()
        self.session = self._create_session()
        self.consecutive_failures = 0
        self.connection_healthy = True
        self.request_history.clear()
        
        logger.info("Discord connection reset complete")