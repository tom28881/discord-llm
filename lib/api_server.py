"""
RESTful API server for Discord monitoring system with OAuth, webhooks,
rate limiting, and comprehensive endpoints for external access.
"""

import logging
import sqlite3
import json
import hashlib
import secrets
import time
import hmac
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from functools import wraps
import threading
from dataclasses import dataclass
from pathlib import Path

# FastAPI for modern API development
try:
    from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    # Fallback to Flask if FastAPI not available
    try:
        from flask import Flask, request, jsonify, abort
        from flask_cors import CORS
        FASTAPI_AVAILABLE = False
        FLASK_AVAILABLE = True
    except ImportError:
        FASTAPI_AVAILABLE = False
        FLASK_AVAILABLE = False

from .notification_database import DB_NAME, NotificationRuleManager, NotificationQueue
from .notification_engine import get_notification_engine
from .external_integrations import get_integration_manager
from .database import get_recent_messages, get_servers, get_channels

logger = logging.getLogger('api_server')

# Pydantic models for FastAPI (if available)
if FASTAPI_AVAILABLE:
    class NotificationRuleCreate(BaseModel):
        name: str = Field(..., description="Rule name")
        server_id: Optional[int] = Field(None, description="Discord server ID")
        channel_id: Optional[int] = Field(None, description="Discord channel ID")
        keywords: Optional[List[str]] = Field(None, description="Keywords to match")
        priority: int = Field(2, ge=1, le=4, description="Priority level (1-4)")
        channels: List[str] = Field(..., description="Notification channels")
        conditions: Optional[Dict[str, Any]] = Field(None, description="Additional conditions")

    class NotificationChannelCreate(BaseModel):
        channel_type: str = Field(..., description="Channel type (email, telegram, etc.)")
        name: str = Field(..., description="Channel name")
        config: Dict[str, Any] = Field(..., description="Channel configuration")
        rate_limit_per_hour: Optional[int] = Field(60, description="Hourly rate limit")
        rate_limit_per_day: Optional[int] = Field(500, description="Daily rate limit")

    class WebhookPayload(BaseModel):
        event_type: str = Field(..., description="Event type")
        data: Dict[str, Any] = Field(..., description="Event data")
        timestamp: Optional[str] = Field(None, description="Event timestamp")

@dataclass
class APIToken:
    """Represents an API token with permissions and rate limits."""
    token_hash: str
    name: str
    permissions: List[str]
    rate_limit_per_hour: int
    rate_limit_per_day: int
    is_active: bool
    last_used: Optional[datetime]
    created_at: datetime
    expires_at: Optional[datetime]

class RateLimiter:
    """Rate limiter for API endpoints."""
    
    def __init__(self):
        self.requests: Dict[str, List[datetime]] = {}
        self._lock = threading.Lock()
    
    def is_allowed(self, token_hash: str, hourly_limit: int, daily_limit: int) -> bool:
        """Check if request is allowed based on rate limits."""
        with self._lock:
            now = datetime.now()
            
            # Initialize if not exists
            if token_hash not in self.requests:
                self.requests[token_hash] = []
            
            requests = self.requests[token_hash]
            
            # Clean old requests
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)
            
            requests[:] = [req_time for req_time in requests if req_time > day_ago]
            
            # Count recent requests
            hourly_count = sum(1 for req_time in requests if req_time > hour_ago)
            daily_count = len(requests)
            
            # Check limits
            if hourly_count >= hourly_limit or daily_count >= daily_limit:
                return False
            
            # Record this request
            requests.append(now)
            return True

class APITokenManager:
    """Manages API tokens and authentication."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
    
    def create_token(self, name: str, permissions: List[str], 
                    rate_limit_per_hour: int = 100, rate_limit_per_day: int = 1000,
                    expires_days: Optional[int] = None) -> str:
        """Create a new API token."""
        # Generate secure token
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        now = int(datetime.now().timestamp())
        expires_at = None
        if expires_days:
            expires_at = int((datetime.now() + timedelta(days=expires_days)).timestamp())
        
        cursor.execute('''
        INSERT INTO api_tokens 
        (token_hash, name, permissions, rate_limit_per_hour, rate_limit_per_day, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (token_hash, name, json.dumps(permissions), rate_limit_per_hour, 
              rate_limit_per_day, now, expires_at))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created API token '{name}' with permissions: {permissions}")
        return token
    
    def validate_token(self, token: str) -> Optional[APIToken]:
        """Validate token and return token info."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT token_hash, name, permissions, rate_limit_per_hour, rate_limit_per_day,
               is_active, last_used, created_at, expires_at
        FROM api_tokens
        WHERE token_hash = ? AND is_active = 1
        ''', (token_hash,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
        
        # Check expiration
        if row[8] and row[8] < int(datetime.now().timestamp()):
            conn.close()
            return None
        
        # Update last used
        now = int(datetime.now().timestamp())
        cursor.execute('''
        UPDATE api_tokens SET last_used = ? WHERE token_hash = ?
        ''', (now, token_hash))
        
        conn.commit()
        conn.close()
        
        return APIToken(
            token_hash=row[0],
            name=row[1],
            permissions=json.loads(row[2]),
            rate_limit_per_hour=row[3],
            rate_limit_per_day=row[4],
            is_active=bool(row[5]),
            last_used=datetime.fromtimestamp(row[6]) if row[6] else None,
            created_at=datetime.fromtimestamp(row[7]),
            expires_at=datetime.fromtimestamp(row[8]) if row[8] else None
        )
    
    def check_permission(self, token: APIToken, required_permission: str) -> bool:
        """Check if token has required permission."""
        return required_permission in token.permissions or 'admin' in token.permissions
    
    def check_rate_limit(self, token: APIToken) -> bool:
        """Check if token is within rate limits."""
        return self.rate_limiter.is_allowed(
            token.token_hash, 
            token.rate_limit_per_hour, 
            token.rate_limit_per_day
        )

# Initialize token manager
token_manager = APITokenManager()

def require_auth(permission: str = None):
    """Decorator to require authentication and optional permission."""
    def decorator(func):
        if FASTAPI_AVAILABLE:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract token from request
                request = kwargs.get('request') or args[0] if args else None
                if not request:
                    raise HTTPException(status_code=500, detail="Internal error: no request object")
                
                auth_header = request.headers.get('authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
                
                token = auth_header[7:]  # Remove 'Bearer ' prefix
                
                # Validate token
                token_info = token_manager.validate_token(token)
                if not token_info:
                    raise HTTPException(status_code=401, detail="Invalid or expired token")
                
                # Check rate limit
                if not token_manager.check_rate_limit(token_info):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                # Check permission
                if permission and not token_manager.check_permission(token_info, permission):
                    raise HTTPException(status_code=403, detail=f"Missing permission: {permission}")
                
                # Add token info to kwargs
                kwargs['token_info'] = token_info
                return await func(*args, **kwargs)
            return wrapper
        else:
            # Flask version
            @wraps(func)
            def wrapper(*args, **kwargs):
                auth_header = request.headers.get('authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    abort(401, "Missing or invalid authorization header")
                
                token = auth_header[7:]
                token_info = token_manager.validate_token(token)
                if not token_info:
                    abort(401, "Invalid or expired token")
                
                if not token_manager.check_rate_limit(token_info):
                    abort(429, "Rate limit exceeded")
                
                if permission and not token_manager.check_permission(token_info, permission):
                    abort(403, f"Missing permission: {permission}")
                
                kwargs['token_info'] = token_info
                return func(*args, **kwargs)
            return wrapper
    return decorator

# FastAPI Application
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Discord Monitor API",
        description="API for Discord monitoring and notification system",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure as needed
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    # Authentication endpoints
    @app.post("/auth/token")
    async def create_api_token(request: Request, token_info: APIToken = Depends(require_auth('admin'))):
        """Create a new API token (admin only)."""
        data = await request.json()
        
        token = token_manager.create_token(
            name=data['name'],
            permissions=data['permissions'],
            rate_limit_per_hour=data.get('rate_limit_per_hour', 100),
            rate_limit_per_day=data.get('rate_limit_per_day', 1000),
            expires_days=data.get('expires_days')
        )
        
        return {"token": token, "message": "Token created successfully"}
    
    # Notification rules endpoints
    @app.get("/rules")
    async def list_notification_rules(request: Request, token_info: APIToken = Depends(require_auth('read'))):
        """List all notification rules."""
        rule_manager = NotificationRuleManager()
        rules = rule_manager.get_active_rules()
        return {"rules": rules}
    
    @app.post("/rules")
    async def create_notification_rule(rule: NotificationRuleCreate, request: Request, 
                                     token_info: APIToken = Depends(require_auth('write'))):
        """Create a new notification rule."""
        rule_manager = NotificationRuleManager()
        
        from .notification_database import NotificationPriority, NotificationChannel
        
        priority = NotificationPriority(rule.priority)
        channels = [NotificationChannel(ch) for ch in rule.channels]
        
        rule_id = rule_manager.create_rule(
            name=rule.name,
            server_id=rule.server_id,
            channel_id=rule.channel_id,
            keywords=rule.keywords,
            priority=priority,
            channels=channels,
            conditions=rule.conditions
        )
        
        return {"rule_id": rule_id, "message": "Rule created successfully"}
    
    # Messages endpoints
    @app.get("/messages")
    async def get_messages(request: Request, server_id: int, hours: int = 24, 
                          channel_id: Optional[int] = None, keywords: Optional[str] = None,
                          token_info: APIToken = Depends(require_auth('read'))):
        """Get recent messages."""
        keyword_list = keywords.split(',') if keywords else None
        
        messages = get_recent_messages(
            server_id=server_id,
            hours=hours,
            keywords=keyword_list,
            channel_id=channel_id
        )
        
        return {"messages": messages, "count": len(messages)}
    
    @app.get("/servers")
    async def list_servers(request: Request, token_info: APIToken = Depends(require_auth('read'))):
        """List available Discord servers."""
        servers = get_servers()
        return {"servers": [{"id": s[0], "name": s[1]} for s in servers]}
    
    @app.get("/servers/{server_id}/channels")
    async def list_channels(server_id: int, request: Request, 
                           token_info: APIToken = Depends(require_auth('read'))):
        """List channels for a specific server."""
        channels = get_channels(server_id)
        return {"channels": [{"id": c[0], "name": c[1]} for c in channels]}
    
    # Notification channels endpoints
    @app.post("/notification-channels")
    async def create_notification_channel(channel: NotificationChannelCreate, request: Request,
                                        token_info: APIToken = Depends(require_auth('admin'))):
        """Create a new notification channel."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        now = int(datetime.now().timestamp())
        
        cursor.execute('''
        INSERT INTO notification_channels 
        (channel_type, name, config, rate_limit_per_hour, rate_limit_per_day, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (channel.channel_type, channel.name, json.dumps(channel.config),
              channel.rate_limit_per_hour, channel.rate_limit_per_day, now, now))
        
        channel_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Reload channels in notification engine
        engine = get_notification_engine()
        engine.reload_channels()
        
        return {"channel_id": channel_id, "message": "Notification channel created"}
    
    # Webhook endpoints
    @app.post("/webhooks/discord")
    async def discord_webhook(payload: WebhookPayload, request: Request,
                             token_info: APIToken = Depends(require_auth('webhook'))):
        """Webhook endpoint for Discord events."""
        try:
            # Process the webhook payload
            engine = get_notification_engine()
            
            if payload.event_type == "message_create":
                data = payload.data
                engine.process_new_message(
                    server_id=data.get('server_id'),
                    channel_id=data.get('channel_id'),
                    message_id=data.get('message_id'),
                    content=data.get('content', ''),
                    author=data.get('author'),
                    metadata=data.get('metadata', {})
                )
            
            return {"status": "processed", "event_type": payload.event_type}
            
        except Exception as e:
            logger.error(f"Webhook processing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Statistics endpoint
    @app.get("/stats")
    async def get_statistics(request: Request, token_info: APIToken = Depends(require_auth('read'))):
        """Get system statistics."""
        engine = get_notification_engine()
        stats = engine.get_statistics()
        
        # Add database stats
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM messages')
        stats['total_messages'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM notification_rules WHERE is_active = 1')
        stats['active_rules'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM notification_queue WHERE status = "pending"')
        stats['pending_notifications'] = cursor.fetchone()[0]
        
        conn.close()
        
        return {"statistics": stats}
    
    # Testing endpoints
    @app.post("/test/notification")
    async def test_notification(request: Request, token_info: APIToken = Depends(require_auth('write'))):
        """Send a test notification."""
        data = await request.json()
        
        engine = get_notification_engine()
        engine.process_new_message(
            server_id=data.get('server_id', 0),
            channel_id=data.get('channel_id', 0),
            message_id=data.get('message_id', 0),
            content=data.get('content', 'Test notification'),
            author=data.get('author', 'API Test'),
            metadata=data.get('metadata', {'test': True})
        )
        
        return {"message": "Test notification sent"}
    
    @app.get("/test/integrations")
    async def test_integrations(request: Request, token_info: APIToken = Depends(require_auth('admin'))):
        """Test all external integrations."""
        integration_manager = get_integration_manager()
        results = integration_manager.test_all_integrations()
        return {"test_results": results}

# Flask Application (fallback)
elif FLASK_AVAILABLE:
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
    
    @app.route('/rules', methods=['GET'])
    @require_auth('read')
    def list_notification_rules(token_info):
        rule_manager = NotificationRuleManager()
        rules = rule_manager.get_active_rules()
        return jsonify({"rules": rules})
    
    @app.route('/messages', methods=['GET'])
    @require_auth('read')
    def get_messages(token_info):
        server_id = int(request.args.get('server_id'))
        hours = int(request.args.get('hours', 24))
        channel_id = request.args.get('channel_id', type=int)
        keywords = request.args.get('keywords')
        
        keyword_list = keywords.split(',') if keywords else None
        
        messages = get_recent_messages(
            server_id=server_id,
            hours=hours,
            keywords=keyword_list,
            channel_id=channel_id
        )
        
        return jsonify({"messages": messages, "count": len(messages)})
    
    @app.route('/stats', methods=['GET'])
    @require_auth('read')
    def get_statistics(token_info):
        engine = get_notification_engine()
        stats = engine.get_statistics()
        return jsonify({"statistics": stats})

else:
    logger.error("Neither FastAPI nor Flask is available. API server cannot be started.")
    app = None

class APIServer:
    """API server manager."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.server_thread: Optional[threading.Thread] = None
        self.running = False
    
    def start(self, debug: bool = False):
        """Start the API server."""
        if not app:
            logger.error("No web framework available. Cannot start API server.")
            return
        
        if self.running:
            logger.warning("API server is already running")
            return
        
        self.running = True
        
        def run_server():
            try:
                if FASTAPI_AVAILABLE:
                    uvicorn.run(app, host=self.host, port=self.port, 
                               log_level="info" if not debug else "debug")
                elif FLASK_AVAILABLE:
                    app.run(host=self.host, port=self.port, debug=debug, threaded=True)
            except Exception as e:
                logger.error(f"API server error: {e}")
                self.running = False
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        logger.info(f"API server started on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the API server."""
        self.running = False
        logger.info("API server stopped")

# Utility functions for external access
def create_api_token(name: str, permissions: List[str], **kwargs) -> str:
    """Create an API token programmatically."""
    return token_manager.create_token(name, permissions, **kwargs)

def setup_default_tokens():
    """Set up default API tokens for development."""
    try:
        # Check if admin token exists
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM api_tokens WHERE name = "admin"')
        if cursor.fetchone()[0] == 0:
            admin_token = create_api_token(
                name="admin",
                permissions=["admin", "read", "write", "webhook"],
                rate_limit_per_hour=1000,
                rate_limit_per_day=10000
            )
            logger.info(f"Created admin token: {admin_token}")
            
            # Create read-only token for general use
            readonly_token = create_api_token(
                name="readonly",
                permissions=["read"],
                rate_limit_per_hour=100,
                rate_limit_per_day=1000
            )
            logger.info(f"Created readonly token: {readonly_token}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Failed to setup default tokens: {e}")

# Initialize default tokens on import
setup_default_tokens()